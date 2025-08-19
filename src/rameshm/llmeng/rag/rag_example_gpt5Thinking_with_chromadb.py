#!/usr/bin/env python3
"""
RAG over HotpotQA (distractor/train) with pluggable embeddings & LLMs.

- Dataset: hotpot_qa, subset "distractor", split "train"
- Documents: for each sample, concatenate each context entry as "title[i] : sentences[i]" joined by ", "
- IDs: use the sample's id field (fallback to _id, then to index if missing)
- Optional metadata: type, level
- Vector store: ChromaDB (persisted, cosine similarity)
- Embeddings: selectable at runtime
- LLM: selectable at runtime
- Config externalized to rag_config.yaml (auto-created with defaults if absent)
"""

import os
import sys
import json
import yaml
from typing import List, Dict, Any, Optional

# Vector store
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Dataset
from datasets import load_dataset

# Embedding backends
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import requests

from dotenv import load_dotenv  # RRM Code Change: To Load environment variables from .env file


# ---------------------------
# Config helpers
# ---------------------------

DEFAULT_CONFIG = {
    "dataset": {
        "name": "hotpot_qa",
        "subset": "distractor",
        "split": "train",
        # Limit ingestion for demo speed; raise if you want more docs
        "ingest_limit": 10,  # Set to 0 for no limit
    },
    "vectorstore": {
        "persist_dir": "./chroma_hotpot_qa_gpt5Thinking",  # Directory to persist ChromaDB
        "collection_name": "hotpotqa_distractor_train_gpt5Thinking",  # Collection name in ChromaDB
        "metric": "cosine",  # hnsw:space
        "query_top_k": 5,
        "batch_size": 64
    },
    "embeddings": {
        # Choose max dims when the provider supports it (e.g., OpenAI, Gemini)
        "max_dimensions": 1024
    },
    "llms": {
        # Listing provided models exactly as requested; ensure availability in your environment
        "supported": [
            {"key": "gpt-4o-mini", "provider": "openai"},
            {"key": "gemini-1.5-flash", "provider": "google"},
            {"key": "llama3.2", "provider": "ollama"},
            {"key": "claude-sonnet-4-20250514", "provider": "anthropic"}
        ],
        "system_prompt": (
            "You are a helpful assistant that answers using only the provided context. "
            "If the answer is not contained in the context, say you don't know."
        ),
        # Safety: truncate context to roughly this many characters in prompt
        "max_context_chars": 12000
    }
}


def load_or_create_config(path: str = "rag_config.yaml") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
    print(f"[info] Created default config at {path}")
    return DEFAULT_CONFIG


# ---------------------------
# Embedding function factory
# ---------------------------

class OpenAIEmbeddingFn(EmbeddingFunction):
    """Embeds with OpenAI 'text-embedding-3-small' (or any model you set) with optional dimensionality."""
    def __init__(self, model: str, dimensions: Optional[int] = None, api_key: Optional[str] = None):
        self.model = model
        self.dimensions = dimensions
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")
        self.client = OpenAI(api_key=key)

    def __call__(self, input: Documents) -> Embeddings:
        # OpenAI supports batching; here we just call once with the whole list.
        # If you hit rate/size limits, split into batches.
        resp = self.client.embeddings.create(
            model=self.model,
            input=list(input),
            dimensions=self.dimensions if self.dimensions else None
        )
        return [d.embedding for d in resp.data]


class GeminiEmbeddingFn(EmbeddingFunction):
    """
    Embeds with Google 'gemini-embedding-001' with optional output_dimensionality.
    Note: exact model naming and parameters can vary by SDK version.
    """
    def __init__(self, model: str, output_dimensionality: Optional[int] = None, api_key: Optional[str] = None):
        self.model = model
        self.output_dimensionality = output_dimensionality
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY in environment.")
        genai.configure(api_key=key)

    def __call__(self, input: Documents) -> Embeddings:
        print(f"DELETE THIS PRINT [DEBUG] Embedding {len(input)} texts with Gemini model {self.model}...")   # RRM Code Change: Added print statement to debug
        print(f"DELETE THIS PRINT [DEBUG] Sample text to embed: {input[0]}...")   # RRM Code Change: Added print statement to debug
        out: Embeddings = []
        # Prefer batch API if available; fall back to per-item.
        # google.generativeai provides batch_embed_contents in some versions.
        # RRM Code Change: genai.batch_embed_contents is not available in the version I have. Hence switching to genai.embed_content
        # try:    # RRM Code Change: Using the genai.embed_content method to embed contents
        #     # Attempt batch call
        #     # NOTE: The SDK's exact method/args may change; keep a TODO to verify.
        #     # TODO: Verify the 'task_type' or additional args if required in your environment.
        #     batch = genai.batch_embed_contents(
        #         model=self.model,
        #         requests=[
        #             {
        #                 "content": {"parts": [{"text": text}]},
        #                 "output_dimensionality": self.output_dimensionality
        #             }
        #             for text in input
        #         ]
        #     )
        #     for r in batch:
        #         out.append(r["embedding"])
        #     return out
        # except Exception:
        # Fallback to single-item calls
        for text in input:
            # Older/newer SDKs may require "model='models/embedding-001'" or similar.
            # We pass through self.model exactly as requested by the prompt:
            # 'gemini-embedding-001'. If your SDK needs a "models/" prefix, add it in config.
            try:
                r = genai.embed_content(
                    model=self.model,
                    content=text,
                    output_dimensionality=self.output_dimensionality
                )
                out.append(r["embedding"])
            except Exception as e:
                # If your runtime needs "models/gemini-embedding-001", try that:
                # This is intentionally a TODO to avoid hallucinating the exact name.
                # TODO: Human input required here — verify correct Gemini embedding model id for your SDK version.
                raise
        return out


class STEmbeddingFn(EmbeddingFunction):
    """Local SentenceTransformer embedding, defaulting to all-MiniLM-L6-v2."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def __call__(self, input: Documents) -> Embeddings:
        vecs = self.model.encode(
            list(input),
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        return [v.tolist() for v in vecs]


def choose_embedding_fn(cfg: Dict[str, Any]) -> EmbeddingFunction:
    print("\nChoose an embedding model:")
    choices = [
        "gemini-embedding-001 (Google)",
        "text-embedding-3-small (OpenAI)",
        "all-MiniLM-L6-v2 (local SentenceTransformer)",
        "Chroma default all-MiniLM-L6-v2 (alias to local ST)"  # aligned to prompt wording
    ]
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    sel = input("Enter 1-4: ").strip()
    dims = cfg.get("embeddings", {}).get("max_dimensions", 1024)

    if sel == "1":
        return GeminiEmbeddingFn(model="gemini-embedding-001", output_dimensionality=dims)
    elif sel == "2":
        return OpenAIEmbeddingFn(model="text-embedding-3-small", dimensions=dims)
    elif sel == "3":
        return STEmbeddingFn(model_name="all-MiniLM-L6-v2")
    elif sel == "4":
        # Chroma’s “default” is not guaranteed; we alias to the local ST model for reliability.
        # This meets the spirit of using all-MiniLM-L6-v2 as the default.
        return STEmbeddingFn(model_name="all-MiniLM-L6-v2")
    else:
        print("[warn] Invalid selection, defaulting to all-MiniLM-L6-v2 (local).")
        return STEmbeddingFn(model_name="all-MiniLM-L6-v2")


# ---------------------------
# Chroma Collection helpers
# ---------------------------

def get_chroma_collection(cfg: Dict[str, Any], embedding_fn: EmbeddingFunction):
    persist_dir = cfg["vectorstore"]["persist_dir"]
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # RRM Code Change: Delete collection if it exists so that we can create new ones with same name but different embeddings
    # RRM Code Change: Below try-except block is for demo purposes to ensure a fresh start
    try:  # RRM Code change - Deleting existing collection for demo purposes
        # If collection exists, delete it for a fresh start (for demo purposes)
        client.delete_collection(
            name=cfg['vectorstore']['collection_name'])  # RRM Code change - Deleting existing collection for demo purposes
        print(
            f"Deleted existing collection: {cfg['vectorstore']['collection_name']}")  # RRM Code change - Print statement for deleted collection
    except Exception:
        pass  # Collection doesn't exist

    collection = client.get_or_create_collection(
        name=cfg["vectorstore"]["collection_name"],
        embedding_function=embedding_fn,
        metadata={"hnsw:space": cfg["vectorstore"]["metric"]}
    )
    return client, collection


# ---------------------------
# Dataset ingestion & formatting
# ---------------------------

def format_document_from_sample(sample: Dict[str, Any]) -> str:
    """
    Build a single string:
    "title[0] : sentences[0], title[1] : sentences[1], …"
    HotpotQA 'context' is typically List[[title, [sentences...]], ...]
    We flatten sentences[i] by joining its list into a single space-separated string.
    """

    ctx = sample.get("context", [])
    parts = []
    for i, entry in enumerate(ctx):
        # Entry is often [title, [sent1, sent2, ...]]. Support dict-style too if present.
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title = entry[0]
            sents = entry[1]
        elif isinstance(entry, dict):
            title = entry.get("title", f"title[{i}]")
            sents = entry.get("sentences", [])
        else:
            title = f"title[{i}]"
            sents = []

        if isinstance(sents, list):
            sent_text = " ".join(sents)
        else:
            sent_text = str(sents)
        parts.append(f"{title} : {sent_text}")
    return ", ".join(parts)


def derive_id(sample: Dict[str, Any], fallback_idx: int) -> str:
    # Instruction: Use id from 'document' (interpreted here as the sample).
    # HotpotQA commonly has 'id' or '_id'. We try both, else fallback to index.
    return str(sample.get("id") or sample.get("_id") or f"hotpot_{fallback_idx}")


def optional_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = {}
    # Optional per instructions:
    if "type" in sample:
        meta["type"] = sample["type"]
    if "level" in sample:
        meta["level"] = sample["level"]
    return meta


def ingest_dataset(cfg: Dict[str, Any], collection) -> None:
    ds_name = cfg["dataset"]["name"]
    subset = cfg["dataset"]["subset"]
    split = cfg["dataset"]["split"]
    limit = int(cfg["dataset"].get("ingest_limit", 500))
    batch_size = int(cfg["vectorstore"].get("batch_size", 64))

    print(f"[info] Loading dataset: {ds_name} / {subset} / {split}")
    dataset = load_dataset(ds_name, subset, split=split)

    # Quick check: skip if already populated
    existing_count = 0
    try:
        # Not all Chroma builds expose count; attempt a cheap query to detect presence.
        ping = collection.query(query_texts=["ping"], n_results=1)
        existing_count = len(ping.get("ids", [[]])[0])
    except Exception:
        pass

    # A more reliable way is to track our own ingestion marker; for demo we proceed anyway.
    print(f"[info] Beginning ingestion (limit={limit}, batch={batch_size})")
    ids, docs, metas = [], [], []
    added = 0

    docs_processed = 0
    for idx, sample in enumerate(dataset):
        if limit and idx >= limit: # RRM Code Change: Stop processing if limit is reached. CHanged from "added < limit" to "idx >= limit"
            break
        doc_id = derive_id(sample, idx)
        #document_text = format_document_from_sample(sample)    # RRM Code Change: Use the new function to format document text
        meta = optional_metadata(sample)

        ctx_entry = sample.get("context", [])   # RRM Code Change: Use 'context' field directly from sample
        for ctx_id, (title, sentences) in enumerate(zip(ctx_entry['title'], ctx_entry['sentences']), 1): # RRM Code Change: New Block
            # RRM Code Change: Use enumerate to get context ID and title/sentences
            document_text = f"{title} : {' '.join(sentences)}" # RRM Code Change: Use f-string for better readability
            chroma_id = f"{doc_id}_ctx{ctx_id}" # RRM Code Change: Create a unique ID for each context entry
            ids.append(chroma_id)
            docs.append(document_text)
            metas.append(meta if meta else None)  # Chroma allows None

        # ids.append(doc_id)
        # docs.append(document_text)
        # metas.append(meta if meta else None)  # Chroma allows None

        if len(ids) >= batch_size:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            added += len(ids)
            print(f"[ingest] Added {added}/{limit}")
            ids, docs, metas = [], [], []

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)
        added += len(ids)
        print(f"[ingest] Added {added}/{limit}")

    print("[info] Ingestion complete.")


# ---------------------------
# Retrieval
# ---------------------------

def retrieve_context(cfg: Dict[str, Any], collection, query: str) -> Dict[str, List[str]]:
    top_k = cfg["vectorstore"]["query_top_k"]
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"] # RRM Code Change: "ids" is  not supported in ChromaDB version I have. Removiving it.
        #include=["documents", "distances", "metadatas", "ids"] # RRM Code Change: "ids" is  not supported in ChromaDB. Removiving it.
    )
    # Flatten outputs for simplicity
    out = {
        #"ids": res.get("ids", [[]])[0], # RRM Code Change: "ids" is actually just a list not list of lists.
        "ids": res.get("ids", [])[0], # RRM Code Change: "ids" is actually just a list not list of lists.
        "documents": res.get("documents", [[]])[0],
        "distances": res.get("distances", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0]
    }
    return out


# ---------------------------
# LLM clients
# ---------------------------

def choose_llm(cfg: Dict[str, Any]) -> Dict[str, str]:
    print("\nChoose an LLM:")
    supported = cfg["llms"]["supported"]
    for i, item in enumerate(supported, 1):
        print(f"  {i}. {item['key']} ({item['provider']})")
    sel = input("Enter 1-4: ").strip()
    try:
        idx = int(sel) - 1
        if not (0 <= idx < len(supported)):
            raise ValueError
    except Exception:
        print("[warn] Invalid selection, defaulting to option 1.")
        idx = 0
    return supported[idx]


def build_prompt(cfg: Dict[str, Any], user_query: str, retrieved_docs: List[str]) -> str:
    system = cfg["llms"]["system_prompt"]
    ctx_joined = "\n\n".join(retrieved_docs)
    max_chars = cfg["llms"]["max_context_chars"]
    if len(ctx_joined) > max_chars:
        ctx_joined = ctx_joined[:max_chars] + "\n[Context truncated]"
    prompt = (
        f"{system}\n\n"
        f"=== Retrieved Context ===\n{ctx_joined}\n\n"
        f"=== Task ===\n"
        f"Question: {user_query}\n"
        f"Answer using ONLY the retrieved context above.\n"
    )
    return prompt


def call_openai_chat(model: str, prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def call_gemini_chat(model: str, prompt: str) -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment.")
    genai.configure(api_key=key)
    # Using GenerativeModel for chat-style generation
    # NOTE: model naming can vary by SDK version; we pass exactly as chosen.
    gm = genai.GenerativeModel(model)
    resp = gm.generate_content(prompt)
    # In some SDKs, use resp.text; in others, flatten candidates/parts.
    try:
        return resp.text
    except Exception:
        # TODO: Human input required here — extract text from the response structure for your SDK version.
        return str(resp)


def call_anthropic_chat(model: str, prompt: str) -> str:
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in environment.")
    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    # messages API returns content as a list of blocks; join text blocks.
    parts = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "\n".join(parts) if parts else str(msg)


def call_ollama_chat(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    ollama_key = os.getenv("OLLAMA_API_KEY")
    if ollama_key:
        headers["Authorization"] = f"Bearer {ollama_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=600)
    r.raise_for_status()
    data = r.json()
    # Ollama chat output usually in data["message"]["content"]
    try:
        return data["message"]["content"]
    except Exception:
        return str(data)


def run_llm_choice(choice: Dict[str, str], prompt: str) -> str:
    model = choice["key"]
    provider = choice["provider"]
    if provider == "openai":
        return call_openai_chat(model, prompt)
    elif provider == "google":
        return call_gemini_chat(model, prompt)
    elif provider == "anthropic":
        return call_anthropic_chat(model, prompt)
    elif provider == "ollama":
        # Assumes local Ollama running and model pulled: `ollama pull llama3.2`
        return call_ollama_chat(model, prompt)
    else:
        # Should not happen with our menu; keep a TODO to avoid hallucination beyond spec.
        # TODO: Human input required here — unsupported provider selected.
        raise RuntimeError(f"Unsupported provider: {provider}")


# ---------------------------
# Main
# ---------------------------

def main():
    load_dotenv()   # RRM Code Change: Load environment variables from .env file

    script_dir = os.path.dirname(os.path.abspath(__file__))  # RRM Code change to set config path relative to script directory
    config_path = os.path.join(script_dir, 'rag_example_config_gpt5thinking_with_chromadb.yaml')  # RRM Code change to set config path relative to script directory
    cfg = load_or_create_config(config_path)  # RRM Code Change: Load config from rag_example_config_gpt5thinking_with_chromadb.yaml
    #cfg = load_or_create_config()  # RRM Code Change: Load config from rag_example_config_gpt5thinking_with_chromadb.yaml

    # 1) Choose embedding model
    embedding_fn = choose_embedding_fn(cfg)

    # 2) Init Chroma collection with cosine distance and embedding fn
    client, collection = get_chroma_collection(cfg, embedding_fn)

    # 3) Ingest HotpotQA (if not already)
    #    For simplicity we ingest up to the configured limit every run; in a real app,
    #    maintain an ingestion marker to avoid duplicate adds.
    ingest_dataset(cfg, collection)

    # 4) Accept user query
    print("\nEnter your question (User Query):")
    user_query = input("> ").strip()
    if not user_query:
        print("[error] Empty query. Exiting.")
        sys.exit(1)

    # 5) Retrieve context from Chroma
    retrieved = retrieve_context(cfg, collection, user_query)
    docs = retrieved["documents"]
    ids = retrieved["ids"]
    print(f"[info] Retrieved {len(docs)} docs: {ids}")

    # 6) Build prompt
    prompt = build_prompt(cfg, user_query, docs)

    # 7) Choose LLM and generate answer
    choice = choose_llm(cfg)
    print(f"[info] Using LLM: {choice['key']} ({choice['provider']})")
    try:
        answer = run_llm_choice(choice, prompt)
    except Exception as e:
        print("[error] LLM call failed:", repr(e))
        # Provide a clear TODO so the user can adjust configuration/environment
        print("# TODO: Human input required here — verify API keys, model availability, and SDK versions.")
        sys.exit(2)

    # 8) Display answer
    print("\n=== Answer ===")
    print(answer)

    # 9) Optionally persist the DB explicitly (some clients auto-persist)
    try:
        client.persist()
    except Exception:
        # Some chroma builds persist automatically; ignore if not available.
        pass


if __name__ == "__main__":
    main()
