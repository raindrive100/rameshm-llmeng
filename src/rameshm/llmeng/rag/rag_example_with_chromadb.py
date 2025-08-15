import os
import yaml
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datasets import load_dataset
import openai
import google.generativeai as genai
import anthropic
import requests

from dotenv import load_dotenv
import logging
from typing import List, Dict

# --- Config Loader ---
class ConfigUtils:
    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def select_embedding_model(config_data):
        embedding_options = config_data["available_embedding_models"]
        print("\nSelect Embedding Model:")
        for i, (k, v) in enumerate(embedding_options.items(), 1):
            print(f"{i}: {k} ({v['description']})")
        emb_choice = input("Enter your choice number: ").strip()
        emb_keys = list(embedding_options.keys())
        try:
            return list(embedding_options.keys())[int(emb_choice)-1]
        except Exception:
            print("Invalid choice. Exiting.")
            exit(1)


    @staticmethod
    def update_config_with_embedding_model(config_data, selected_emb):
        config_data["embedding_model"] = selected_emb
        return config_data

class RAGConfig:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


# --- Embedding Providers ---
class EmbeddingProvider:
    def get_embedding_function(self, config: RAGConfig):
        raise NotImplementedError

class ChromaDBDefaultEmbeddingProvider(EmbeddingProvider):
    def get_embedding_function(self, config: RAGConfig):
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def get_embedding_function(self, config: RAGConfig):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )

class GoogleEmbeddingFunction:
    def __init__(self, api_key, model_name, dimensions):
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        genai.configure(api_key=api_key)
    def __call__(self, input):
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                output_dimensionality=self.dimensions,
                task_type="RETRIEVAL_QUERY"
            )
            embeddings.append(result["embedding"])
        return embeddings

class GoogleEmbeddingProvider(EmbeddingProvider):
    def get_embedding_function(self, config: RAGConfig):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return GoogleEmbeddingFunction(
            api_key=api_key,
            model_name="models/embedding-001",
            dimensions=config.embedding["dimensions"]
        )

class EmbeddingFactory:
    _providers = {
        "all-MiniLM-L6-v2": ChromaDBDefaultEmbeddingProvider,
        "text-embedding-3-small": OpenAIEmbeddingProvider,
        "gemini-embedding-001": GoogleEmbeddingProvider,
    }
    @classmethod
    def get_provider(cls, model_name):
        provider_cls = cls._providers.get(model_name)
        if not provider_cls:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        return provider_cls()

# --- LLM Providers ---
class LLMProvider:
    def generate_response(self, context, query, system_prompt):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def generate_response(self, context, query, system_prompt):
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content

class GoogleProvider(LLMProvider):
    def __init__(self, model_name):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    def generate_response(self, context, query, system_prompt):
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.model.generate_content(prompt)
        return response.text

class ClaudeProvider(LLMProvider):
    def __init__(self, model_name):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
    def generate_response(self, context, query, system_prompt):
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class OllamaProvider(LLMProvider):
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    def generate_response(self, context, query, system_prompt):
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        return response.json()["response"]

class LLMFactory:
    _providers = {
        "gpt-4o-mini": OpenAIProvider,
        "gemini-1.5-flash": GoogleProvider,
        "llama3.2": OllamaProvider,
        "claude-sonnet-4-20250514": ClaudeProvider,
    }
    @classmethod
    def get_provider(cls, model_name):
        provider_cls = cls._providers.get(model_name)
        if not provider_cls:
            raise ValueError(f"Unsupported LLM model: {model_name}")
        return provider_cls(model_name)

# --- Data Loader ---
class DataLoader:
    @staticmethod
    def load_hotpot_qa_data(name: str, subset: str, split: str, ingest_limit: int =None):
        print(f"Loading dataset: {name}, subset: {subset}, split: {split}")
        dataset = load_dataset(name, subset, split=split)
        if ingest_limit and ingest_limit > 0:
            print(f"Ingest limit set to {ingest_limit}. Limiting dataset size.")
            dataset = dataset.select(range(ingest_limit))
        return dataset
    @staticmethod
    def process_context_data(data):
        ids, documents, metadatas = [], [], []
        for idx, example in enumerate(data):
            context = example["context"]
            titles = context["title"]
            sentences = context["sentences"]
            for ctx_idx, (title, sentences_part) in enumerate(zip(titles, sentences)):
                document = f"{title} : {'; '.join(sentences_part)}"
                ids.append(f"{example['id']}_{ctx_idx}")
                documents.append(document)
                meta = {
                    "type": example.get("type", "unknown"),
                    "level": example.get("level", "unknown")
                }
                metadatas.append(meta)
        return ids, documents, metadatas

# --- Vector Store ---
class VectorStore:
    def __init__(self, config: RAGConfig, embedding_function):
        self.config = config
        self.client = chromadb.PersistentClient(
            path=config.vector_store["path"],
            settings=Settings(anonymized_telemetry=False)
        )

        col_name = self.config.vector_store["collection_name"]
        # If collection exists, delete it for a fresh start (for demo purposes)
        try:
            self.client.delete_collection(name=col_name)
            logging.info(f"Deleted existing collection: {col_name}")
        except Exception:
            pass  # Collection doesn't exist. Not an error

        self.collection = self.client.get_or_create_collection(
            name=col_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )


    def add_documents(self, ids, documents, metadatas):
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    def query(self, query_text, n_results):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

# --- RAG Pipeline ---
class RAGPipeline:
    def __init__(self, rag_config: RAGConfig):
        self.config = rag_config
        embedding_provider = EmbeddingFactory.get_provider(self.config.embedding_model)
        embedding_function = embedding_provider.get_embedding_function(self.config)
        self.vector_store = VectorStore(self.config, embedding_function)
        self.llm_provider = None
    def load_and_index_data(self):
        ingest_limit = self.config.dataset.get("ingest_limit", None)
        data = DataLoader.load_hotpot_qa_data(self.config.dataset["name"], self.config.dataset['subset'], self.config.dataset["split"],
                                              ingest_limit)
        print(f"Loaded {len(data)} examples from dataset.")
        ids, documents, metadatas = DataLoader.process_context_data(data)
        self.vector_store.add_documents(ids, documents, metadatas)
    def setup_llm(self, model_name):
        self.llm_provider = LLMFactory.get_provider(model_name)
    def query(self, user_query):
        results = self.vector_store.query(user_query, self.config.retrieval["top_k_results"])
        documents = results["documents"][0]
        distances = results["distances"][0]
        print(f"Retrieved {len(documents)} documents with distances: {distances}")
        context_parts = []
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            context_parts.append(f"Document {i+1} (similarity: {1-distance:.3f}):\n{doc}\n")
        context = "\n".join(context_parts)
        if len(context) > self.config.retrieval["max_context_length"]:
            print(f"Context length ({len(context)}) exceeds max limit ({self.config.retrieval['max_context_length']}). Truncating.")
            context = context[:self.config.retrieval["max_context_length"]] + "..."
        return self.llm_provider.generate_response(
            context, user_query, self.config.llm_generation["system_prompt"]
        )

# --- Main ---
def main():
    load_dotenv()   # Load environment variables from .env file

    # Set path to the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'rag_example_config_with_chromadb.yaml')
    config_data = ConfigUtils.load_config(config_path)

    # Set embedding model based on user selection
    selected_emb = ConfigUtils.select_embedding_model(config_data)
    config_data = ConfigUtils.update_config_with_embedding_model(config_data, selected_emb)
    print(f"Using embedding model: {selected_emb}")

    rag_config = RAGConfig(config_data)
    pipeline = RAGPipeline(rag_config)
    print("Loading and indexing data (this may take a while)...")
    pipeline.load_and_index_data()

    # Select and set-up LLM model based on user input
    llm_options = pipeline.config.available_llm_models
    print("\nSelect LLM model:")
    for i, (k, v) in enumerate(llm_options.items(), 1):
        print(f"{i}: {k} ({v['description']})")
    llm_choice = input("Enter your choice number: ").strip()
    llm_keys = list(llm_options.keys())

    retry = True
    num_tries = 0
    while retry:
        try:
            selected_llm = llm_keys[int(llm_choice)-1]
            break
        except (ValueError, IndexError):
            num_tries += 1
            print("Invalid choice.")
            if num_tries < 3:
                llm_choice = input("Please enter a valid choice number: ").strip()
                continue
            else:
                err_msg = "Too many invalid attempts for LLM selection."
                print(err_msg)
                raise ValueError(err_msg)
    pipeline.setup_llm(selected_llm)

    print("\nReady for queries. Type 'quit' to exit.")
    while True:
        user_query = input("\nEnter your question: ").strip()
        if user_query.lower() == "quit":
            break
        if not user_query:
            print("Please enter a valid question.")
            continue
        print("Processing your query...")
        try:
            response = pipeline.query(user_query)
            print(f"\nAnswer:\n{response}")
        except Exception as e:
            err_msg = f"Error processing query: {e}"
            logging.error(err_msg, exc_info=True)
            print(err_msg)


if __name__ == "__main__":
    main()