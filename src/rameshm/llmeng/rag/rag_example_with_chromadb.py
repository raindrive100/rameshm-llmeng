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

from dotenv import load_dotenv  # RRM Code change: Added to load environment variables from .env file
import logging  # RRM Code change: Added for logging errors
from typing import List, Dict   # RRM Code change: Added for type hints

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
            selected_emb = emb_keys[int(emb_choice)-1]
        except Exception:
            print("Invalid choice.")
            exit(1)
        return selected_emb

    @staticmethod
    def update_config_with_embedding_model(config_data, selected_emb):
        config_data["embedding_model"] = selected_emb
        return config_data

class RAGConfig:
    def __init__(self, config_dict):
        self._config = config_dict
        self.embedding_model = self._config["embedding_model"]
        self.llm_model = self._config["llm_model"]
        self.vector_store = self._config["vector_store"]
        self.embedding = self._config["embedding"]
        self.retrieval = self._config["retrieval"]
        self.llm_generation = self._config["llm_generation"]
        self.available_embedding_models = self._config["available_embedding_models"]
        self.available_llm_models = self._config["available_llm_models"]


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
                task_type="retrieval_document"
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
    def load_hotpot_qa_data():
        dataset = load_dataset("hotpot_qa", "distractor")
        return dataset["train"]
    @staticmethod
    def process_context_data(data, max_examples=None):
        ids, documents, metadatas = [], [], []
        for idx, example in enumerate(data):
            if max_examples and idx >= max_examples:
                break
            context = example["context"]
            titles = context["title"]
            sentences = context["sentences"]
            doc_parts = []
            for t, s in zip(titles, sentences):
                doc_parts.append(f"{t} : {', '.join(s)}")
            document = ", ".join(doc_parts)
            ids.append(example["id"])
            documents.append(document)
            meta = {
                "type": example.get("type", "unknown"), # RRM Code change: "type" is from "example" and not from context.
                "level": example.get("level", "unknown")    # RRM Code change: "level" is from "example" and not from context.
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
        try:  # RRM Code change - Deleting existing collection for demo purposes
            # If collection exists, delete it for a fresh start (for demo purposes)
            self.client.delete_collection(name=col_name)
            logging.info(f"Deleted existing collection: {col_name}")
        except Exception:
            pass  # Collection doesn't exist

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
        data = DataLoader.load_hotpot_qa_data()
        max_examples = self.config.retrieval.get("max_examples")
        ids, documents, metadatas = DataLoader.process_context_data(data, max_examples)
        self.vector_store.add_documents(ids, documents, metadatas)
    def setup_llm(self, model_name):
        self.llm_provider = LLMFactory.get_provider(model_name)
    def query(self, user_query):
        results = self.vector_store.query(user_query, self.config.retrieval["top_k_results"])
        documents = results["documents"][0]
        distances = results["distances"][0]
        context_parts = []
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            context_parts.append(f"Document {i+1} (similarity: {1-distance:.3f}):\n{doc}\n")
        context = "\n".join(context_parts)
        if len(context) > self.config.retrieval["max_context_length"]:
            context = context[:self.config.retrieval["max_context_length"]] + "..."
        return self.llm_provider.generate_response(
            context, user_query, self.config.llm_generation["system_prompt"]
        )

# --- Main ---
def main():
    load_dotenv()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'rag_config_copilot_gpt.yaml')

    config_data = ConfigUtils.load_config(config_path)
    selected_emb = ConfigUtils.select_embedding_model(config_data)
    config_data = ConfigUtils.update_config_with_embedding_model(config_data, selected_emb)
    print(f"Using embedding model: {selected_emb}")

    rag_config = RAGConfig(config_data)
    pipeline = RAGPipeline(rag_config)
    print("Loading and indexing data (this may take a while)...")
    pipeline.load_and_index_data()
    llm_options = pipeline.config.available_llm_models
    print("\nSelect LLM model:")
    for i, (k, v) in enumerate(llm_options.items(), 1):
        print(f"{i}: {k} ({v['description']})")
    llm_choice = input("Enter your choice number: ").strip()
    llm_keys = list(llm_options.keys())
    try:
        selected_llm = llm_keys[int(llm_choice)-1]
    except Exception:
        print("Invalid choice.")
        return
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
            logging.error(f"Error: {e}", exc_info=True) # RRM Code change: Added logging for errors.

if __name__ == "__main__":
    main()