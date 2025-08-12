#!/usr/bin/env python3
"""
RAG Demo (HotpotQA + ChromaDB) — runtime selection WITHOUT saving to config

- Dataset: hotpot_qa (config="distractor", split="train")
- Vector store: ChromaDB (persistent), cosine metric (hnsw:space)
- Embeddings (selected at runtime, loaded from config):
    * OpenAI: text-embedding-3-small (dimensions via config, <=1024)
    * Google: gemini-embedding-001 (output_dimensionality via config, <=1024)
    * Chroma default: all-MiniLM-L6-v2 (fixed dims)
- LLMs (selected at runtime, loaded from config):
    * gpt-4o-mini (OpenAI), gemini-1.5-flash (Google), claude-sonnet-4-20250514 (Anthropic),
      llama3.2 (local via Ollama)
- NO config writes: user choices affect only the in-memory cfg for this run.
- Pluggable factories + clear TODOs where specs can vary by SDK/version.

Install (example):
  pip install --upgrade pip
  pip install datasets chromadb pyyaml
  pip install openai anthropic google-generativeai requests
  pip install sentence-transformers
"""

import os
import sys
import json
import uuid
import yaml
import requests
from typing import List, Dict, Any, Tuple, Optional

# Vector DB
import chromadb
from chromadb.config import Settings

# Hugging Face datasets
from datasets import load_dataset

# Optional embedding function from Chroma (SentenceTransformers)
from chromadb.utils import embedding_functions

# OpenAI
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None  # Checked at runtime

# Anthropic (Claude)
try:
    import anthropic
except Exception:
    anthropic = None  # Checked at runtime

# Google / Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None  # Checked at runtime

from abc import ABC, abstractmethod # RRM Code change to import ABC and abstractmethod for defining abstract classes
from rameshm.llmeng.utils import init_utils # RRM Code change to import init_utils from rameshm.llmeng.utils

# Initialize the logger and sets environment variables
logger = init_utils.set_environment_logger() # RRM Code change to set logger and environment variables


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_CONFIG = {
    "chromadb": {
        "persist_directory": "./chroma_rag_store_gpt5Thinking",  # directory for ChromaDB persistence
        "collection_name": "hotpot_distractor_train",
        "distance_metric": "cosine",  # sets hnsw:space
        "batch_size": 256
    },
    "dataset": {
        "name": "hotpot_qa",
        "config": "distractor",
        "split": "train",
        "ingest_limit": 10,  # 1500,  # keep modest for demo speed, RRM Code Change from 1500 to 10
        "top_k": 5
    },
    # All supported embedding options live in config. We only read from here.
    "embedding": {
        "default_key": "openai",   # used if user selection fails
        "options": [
            {
                "key": "openai",
                "label": "OpenAI — text-embedding-3-small",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1024
            },
            {
                "key": "google",
                "label": "Google Gemini — gemini-embedding-001",
                "provider": "google",
                "google_model": "gemini-embedding-001",
                "dimensions": 1024
            },
            {
                "key": "chroma_default",
                "label": "Chroma Default — all-MiniLM-L6-v2",
                "provider": "chroma_default",
                "chroma_model": "all-MiniLM-L6-v2"
                # fixed dims (≈384), no 'dimensions' field needed
            }
        ]
    },
    # All supported LLM options live in config. We only read from here.
    "llms": {
        "max_output_tokens": 600,
        "options": [
            {"key": "gpt-4o-mini", "label": "OpenAI — gpt-4o-mini", "provider": "openai"},
            {"key": "gemini-1.5-flash", "label": "Google — gemini-1.5-flash", "provider": "google"},
            {"key": "claude-sonnet-4-20250514", "label": "Anthropic — claude-sonnet-4-20250514", "provider": "anthropic"},
            {"key": "llama3.2", "label": "Local (Ollama) — llama3.2", "provider": "ollama"}
        ]
    },
    "ollama": {
        "host": "http://localhost:11434"
    }
}


def ensure_config(path: str) -> Dict[str, Any]: # RRM Code change to ensure config is loaded from a specified path
    """Create default config if missing; always return the file content (no writes after)."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
        print(f"[INFO] Wrote default config to {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# Embedding functions
# -----------------------------

class EmbeddingFunction():   # Adding ABC to define an abstract base class for embedding functions
    """Interface expected by Chroma: callable(List[str]) -> List[List[float]]."""
    #@abstractmethod
    def __call__(self, input: List[str]) -> List[List[float]]: # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
        raise NotImplementedError


class OpenAIEmbeddingFn(EmbeddingFunction):
    """OpenAI embedding function with optional dimensionality control."""
    def __init__(self, model: str, dimensions: Optional[int] = None, api_key: Optional[str] = None):
        if OpenAIClient is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        self.client = OpenAIClient(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimensions = dimensions
        print(f"[INFO] Initialized OpenAIEmbeddingFn with model={model}, dimensions={dimensions}")

    #def __call__(self, texts: List[str]) -> List[List[float]]: # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
    def __call__(self, input: List[str]) -> List[List[float]]:  # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
        resp = self.client.embeddings.create(
            model=self.model,
            input=input,    # RRM Code changes from texts to input
            **({"dimensions": self.dimensions} if self.dimensions else {})
        )
        return [d.embedding for d in resp.data]


class GoogleGeminiEmbeddingFn(EmbeddingFunction):
    """
    Google Gemini embeddings via google-generativeai.
    - Model: 'gemini-embedding-001'
    - Dimensionality control: output_dimensionality (e.g., 1024)
    """
    def __init__(self, model: str, dimensions: Optional[int] = None, api_key: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.dimensions = dimensions

    #def __call__(self, texts: List[str]) -> List[List[float]]:  # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
    def __call__(self, input: List[str]) -> List[List[float]]:  # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
        out = []
        for text in input: # RRM Code change (changed texts to input and t to text)
            try:
                resp = genai.embed_content(
                    model=self.model,              # e.g., "gemini-embedding-001"
                    content=text,   # RRM Code change (changed t to text)
                    output_dimensionality=self.dimensions  # when supported
                )
                # SDKs differ slightly in shape; handle common cases:
                if isinstance(resp, dict) and "embedding" in resp:
                    out.append(resp["embedding"])   # RRM Code change response is a dict so this line is correct
                elif hasattr(resp, "embedding"):
                    out.append(resp.embedding)
                else:
                    raise RuntimeError("Unexpected Gemini embedding response format")
            except Exception as e:
                # TODO: Human input required here — verify SDK/model support (may need 'text-embedding-004' etc.)
                raise RuntimeError(
                    f"Gemini embedding call failed. Check model/SDK. Original error: {e}"
                )
        return out


class ChromaSentenceTransformerFn(EmbeddingFunction):
    """Chroma's SentenceTransformer wrapper (all-MiniLM-L6-v2). Fixed dimension (≈384)."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:  # RRM Code change to match ChromaDB expected signature (changed texts to input)
        return self._fn(input) # RRM Code change to match ChromaDB expected signature (changed texts to input)


def build_embedding_function_from_option(opt: Dict[str, Any]) -> EmbeddingFunction:
    provider = opt["provider"].lower()
    logger.info(f"Building embedding function for provider: {provider} (model={opt.get('model')})")
    if provider == "openai":
        return OpenAIEmbeddingFn(
            model=opt["model"],
            dimensions=opt.get("dimensions"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "google":
        return GoogleGeminiEmbeddingFn(
            model=opt["google_model"],
            dimensions=opt.get("dimensions"),
            api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "chroma_default":
        return ChromaSentenceTransformerFn(model_name=opt.get("chroma_model", "all-MiniLM-L6-v2"))
    else:
        # TODO: Human input required here — extend with new providers.
        raise ValueError(f"Unsupported embedding provider: {provider}")


def pick_embedding_option(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ask user to pick an embedding option from cfg; return the chosen option dict (no config writes)."""
    opts = cfg["embedding"]["options"]
    print("\nSelect an Embedding Provider/Model:")
    for i, o in enumerate(opts, 1):
        print(f"  {i}. {o['label']}  [key={o['key']}]")
    sel = input("Enter the number of the embedding to use: ").strip()
    try:
        idx = int(sel) - 1
        return opts[idx]
    except Exception:
        print("[WARN] Invalid selection. Falling back to default.")
        def_key = cfg["embedding"].get("default_key", opts[0]["key"])
        return next((o for o in opts if o["key"] == def_key), opts[0])


# -----------------------------
# LLM clients
# -----------------------------

class LLM:
    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        raise NotImplementedError


class OpenAILLM(LLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if OpenAIClient is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        self.client = OpenAIClient(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly uses the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


class GeminiLLM(LLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text if hasattr(resp, "text") else str(resp)


class ClaudeLLM(LLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("CLAUDE_API_KEY"))
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            system="You are a helpful assistant that strictly uses the provided context.",
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return "".join(block.text for block in msg.content if hasattr(block, "text"))
        except Exception:
            return str(msg)


class OllamaLLM(LLM):
    def __init__(self, model: str, host: str, api_key: Optional[str] = None):
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")

    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        url = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that strictly uses the provided context."},
                {"role": "user", "content": prompt},
            ],
            "stream": False
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"]
            if "response" in data:
                return data["response"]
        return json.dumps(data, indent=2)


def build_llm(model_key: str, cfg: Dict[str, Any]) -> LLM:
    if model_key == "gpt-4o-mini":
        return OpenAILLM(model=model_key, api_key=os.getenv("OPENAI_API_KEY"))
    elif model_key == "gemini-1.5-flash":
        return GeminiLLM(model=model_key, api_key=os.getenv("GOOGLE_API_KEY"))
    elif model_key == "claude-sonnet-4-20250514":
        # TODO: Human input required here — verify this Claude model name is enabled for your account.
        return ClaudeLLM(model=model_key, api_key=os.getenv("CLAUDE_API_KEY"))
    elif model_key == "llama3.2":
        return OllamaLLM(model="llama3.2", host=cfg["ollama"]["host"], api_key=os.getenv("OLLAMA_API_KEY"))
    else:
        # TODO: Human input required here — add new LLMs via config + this factory.
        raise ValueError(f"Unsupported LLM choice: {model_key}")


def select_llm_key(cfg: Dict[str, Any]) -> str:
    opts = cfg["llms"]["options"]
    print("\nSelect an LLM:")
    for i, o in enumerate(opts, 1):
        print(f"  {i}. {o['label']}  [key={o['key']}]")
    choice = input("Enter the number of the LLM to use: ").strip()
    try:
        idx = int(choice) - 1
        return opts[idx]["key"]
    except Exception:
        print("[ERROR] Invalid selection.")
        sys.exit(1)


# -----------------------------
# Dataset ingestion utilities
# -----------------------------

def format_context_entry(title: str, sentences: List[str]) -> str:
    """
    Format: 'title : sentence0, title : sentence1, ...'
    (Matches the requested formatting using repeated 'title : sentence' pairs.)
    """
    if not sentences:
        return f"{title} :"
    return ", ".join([f"{title} : {s}" for s in sentences])


def extract_docs_from_example(example: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Produce (doc_id, document_text, metadata) per 'context' entry.

    NOTE on IDs:
    - Requirement says: "Use `id` from `context` for collection.ids".
      HotpotQA 'context' entries lack their own IDs.
      # TODO: Human input required here — we synthesize IDs from example 'id' + context index.
    """
    out = []
    example_id = example.get("id", str(uuid.uuid4()))
    level = example.get("level")
    ex_type = example.get("type")

    #context = example.get("context", [])   # RRM Code change to use Dict as default
    context = example.get("context", {})  # RRM Code change to use Dict as default
    #print(f"[DEBUG] Example ID: {example_id} has title count: {len(context['title'])} Sentences Length: {len(context['sentences'])}") # RRM Code change added print statement to debug titles and sentences length
    for idx, (title, sentences) in enumerate(zip(context.get("title", []), context.get("sentences", []))):  # RRM Code change to use titles and sentences from example
    # for idx, item in enumerate(context):  # RRM Code change - Commenting out the original context extraction
    #     if isinstance(item, (list, tuple)) and len(item) == 2:
    #         title, sentences = item[0], item[1]
    #     elif isinstance(item, dict) and "title" in item and "sentences" in item:
    #         title, sentences = item.get("title"), item.get("sentences")
    #     else:
    #         continue
    #
    #     title = title if isinstance(title, str) else str(title)
    #     sentences = [s if isinstance(s, str) else str(s) for s in (sentences or [])]
    #
        doc_text = format_context_entry(title, sentences)
        doc_id = f"{example_id}__ctx_{idx}"  # synthesized
        metadata = {"level": level, "type": ex_type}
        out.append((doc_id, doc_text, metadata))
    #print(f"Processed example {example_id}: extracted {len(out)} documents.")  # RRM Code change to print processed example and extracted documents count
    return out


def ingest_dataset_into_chroma(cfg: Dict[str, Any], collection) -> None:
    """Load HotpotQA 'distractor/train' and add documents to Chroma (batched)."""
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    config = ds_cfg["config"]
    split = ds_cfg["split"]
    ingest_limit = int(ds_cfg.get("ingest_limit") or 0)
    batch_size = int(cfg["chromadb"]["batch_size"])

    print(f"[INFO] Loading dataset: {name} / {config} / {split}")
    dataset = load_dataset(name, config, split=split)
    total_examples = len(dataset)
    print(f"[INFO] Dataset loaded with {total_examples} examples")

    to_process = min(ingest_limit, total_examples) if ingest_limit else total_examples
    print(f"[INFO] Ingesting up to {to_process} examples...")
    logger.info(f"Ingesting {to_process} documents...")   # RRM Code change to make log entry

    ids, docs, metas = [], [], []
    count = 0

    for example in dataset.select(range(to_process)):
        items = extract_docs_from_example(example)
        #print(f"[INFO] Extracted {len(items)} context entries from example {example.get('id', 'unknown')}")
        for doc_id, doc_text, meta in items:
            ids.append(doc_id)
            docs.append(doc_text)
            metas.append(meta)
            count += 1

            if len(ids) >= batch_size:
                collection.add(ids=ids, documents=docs, metadatas=metas)
                print(f"[INFO] Added {len(ids)} docs (total so far: {count})")
                ids, docs, metas = [], [], []

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"[INFO] Added final batch into ChromaDB: {len(ids)} docs")
    print(f"[INFO] Ingestion complete. Total docs added into ChromaDB: {count}")
    logger.info(f"Ingestion complete. Total docs added into ChromaDB: {count}")   # RRM Code change to make log entry


# -----------------------------
# Chroma helpers
# -----------------------------

def get_or_create_collection(client, cfg: Dict[str, Any], embedding_fn: EmbeddingFunction):
    col_name = cfg["chromadb"]["collection_name"]
    metric = cfg["chromadb"]["distance_metric"]
    metadata = {"hnsw:space": metric}

    try:    # RRM Code change - Deleting existing collection for demo purposes
        # If collection exists, delete it for a fresh start (for demo purposes)
        client.delete_collection(name=col_name)
        logger.info(f"Deleted existing collection: {col_name}")
    except Exception:
        pass  # Collection doesn't exist

    # try:    # RRM Code change - Commenting out existing collection retrieval for demo purposes
    #     col = client.get_collection(name=col_name, embedding_function=embedding_fn)
    #     print(f"[INFO] Using existing collection '{col_name}'")
    #     return col
    # except Exception:
    #     pass

    print(f"[INFO] Creating collection '{col_name}' with metric={metric}")
    return client.create_collection(
        name=col_name,
        metadata=metadata,
        embedding_function=embedding_fn
    )
    logger.info(f"Created collection '{col_name}' with metric={metric}")


def ensure_persistent_client(cfg: Dict[str, Any]):
    persist_dir = cfg["chromadb"]["persist_directory"]
    os.makedirs(persist_dir, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=persist_dir)
    except AttributeError:
        return chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))


# -----------------------------
# Retrieval + Prompt
# -----------------------------

def retrieve_context(collection, query: str, top_k: int) -> Dict[str, Any]:
    """Use Chroma to retrieve top_k docs for the user query."""
    return collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"] # RRM Code change: "ids" isn't a supported attribute hence removed.
    )


def build_prompt(user_query: str, retrieved: Dict[str, Any]) -> str:
    docs = retrieved.get("documents", [[]])[0]
    ids = retrieved.get("ids", [[]])[0]
    dists = retrieved.get("distances", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]

    ctx_blocks = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        dist = dists[i] if i < len(dists) else None
        doc_id = ids[i] if i < len(ids) else f"doc_{i}"
        ctx_blocks.append(f"[DOC {i+1} | id={doc_id} | distance={dist} | meta={meta}]\n{doc}")
    context_text = "\n\n".join(ctx_blocks)

    prompt = f"""You are a retrieval-augmented assistant. Use ONLY the context below to answer the user's question.
If the answer isn't in the context, say you don't have enough information.

# Context
{context_text}

# Question
{user_query}

# Instructions
- Cite the doc ids you used when relevant (e.g., "Based on DOC 2 and DOC 4").
- Be concise and accurate.
"""
    logger.debug(f"Prompt for LLM: {prompt[:500]}... (truncated for debug)")  # RRM Code change to log the prompt

    return prompt   # RRM Code change to return the prompt for LLM generation


# -----------------------------
# Main flow
# -----------------------------

def main():
    #CONFIG_PATH = "rag_gpt5Thinking_config.yaml"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # RRM Code change to set config path relative to script directory
    config_path = os.path.join(script_dir, 'rag_config_gpt5thinking.yaml')  # RRM Code change to set config path relative to script directory

    cfg = ensure_config(config_path)    # RRM Code change to ensure config is loaded from a specified path

    # 1) Embedding selection (no config writes)
    emb_opt = pick_embedding_option(cfg)
    embedding_fn = build_embedding_function_from_option(emb_opt)

    # 2) Persistent Chroma + collection
    client = ensure_persistent_client(cfg)
    collection = get_or_create_collection(client, cfg, embedding_fn)

    # 3) Ingest dataset if empty
    try:
        num = collection.count()
    except Exception:
        num = 0
    if num == 0:
        ingest_dataset_into_chroma(cfg, collection)
        try:
            client.persist()
        except Exception:
            pass
    else:
        print(f"[INFO] Collection already has {num} vectors")

    # 4) LLM selection (no config writes)
    model_key = select_llm_key(cfg)
    llm = build_llm(model_key, cfg)

    # Interactive query loop
    print("\n=== Ready for queries ===")
    print("Type 'quit' to exit")

    while True: # RRM Code change to run in a loop for user queries
        # 5) User query
        user_query = input("\nEnter your question: ").strip()
        if not user_query:
            print("[ERROR] Empty query.")
            sys.exit(1)
        elif user_query.lower() == "quit":
            print("[INFO] Exiting.")
            sys.exit(0)

        print("Processing your query...")
        logger.info(f"User query: {user_query} for LLM: {model_key}")  # RRM Code change to log the user query and LLM model

        # 6) Retrieve + Generate
        top_k = int(cfg["dataset"]["top_k"])
        retrieved = retrieve_context(collection, user_query, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved['documents'][0])} documents/ids/metadatas")  # RRM Code change to log the number of retrieved documents
        logger.debug(f"Ids Retrieved: {retrieved['ids']}")  # RRM Code change to log the retrieved ids
        prompt = build_prompt(user_query, retrieved)

        print("\n[INFO] Generating answer...\n")
        answer = llm.generate(prompt, max_tokens=int(cfg["llms"]["max_output_tokens"]))

        print("\n===== RAG ANSWER =====\n")
        print(answer)
        print("\n======================\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
