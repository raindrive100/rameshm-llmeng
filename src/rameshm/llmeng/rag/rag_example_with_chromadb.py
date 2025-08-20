import os
import yaml
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datasets import load_dataset
import openai
import google.generativeai as genai
import anthropic
from dotenv import load_dotenv
from logging_config import MyLogger

logger = None   # Set logger as a global variable.


# --- Config Loader ---
class ConfigUtils:
    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


    @staticmethod
    def select_model(config_data, model_type: str) -> str:
        """Selects LLM model from available options in the config data."""
        if model_type.lower() == "llm":
            model_options = config_data["available_llm_models"]
        elif model_type.lower() == "embedding":
            model_options = config_data["available_embedding_models"]
        else:
            raise ValueError("model_choice must be either 'llm' or 'embedding'")

        print(f"\nSelect {model_type} model:")
        for i, (k, v) in enumerate(model_options.items(), 1):
            print(f"{i}: {k} ({v['description']})")
        model_keys = list(model_options.keys())

        # Select LLM with retries
        retry = True
        num_tries = 0
        selected_model = None
        while retry:
            try:
                model_choice = input("Enter your choice number: ").strip()
                print("\n") # For better readability in console
                if int(model_choice) > 0:
                    selected_model = model_keys[int(model_choice) - 1]
                else:
                    raise ValueError("Choice [{model_choice}] must be a positive number.")
                break
            except (ValueError, IndexError):
                num_tries += 1
                print("Invalid model choice.")
                if num_tries < 3:
                    continue
                else:
                    err_msg = f"Too many invalid attempts for {model_type} model selection."
                    print(err_msg)
                    raise ValueError(err_msg)
        return selected_model


    @staticmethod
    def update_config_with_embedding_model(config_data, selected_emb):
        config_data["embedding_model"] = selected_emb
        return config_data


    @staticmethod
    def update_config_with_llm_model(config_data, selected_llm):
        config_data["llm_model"] = selected_llm
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
        logger.debug(f"Creating Embedder Model Class: {provider_cls.__name__}")
        if not provider_cls:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        return provider_cls()

# --- LLM Providers ---
class LLMProvider:
    def generate_response(self, context, query, system_prompt):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    # Both OpenAI and Ollama use the same OpenAI API client
    def __init__(self, model_name):
        self.model_name = model_name
        if ("llama" in model_name):
            self.client = openai.OpenAI(base_url = "http://localhost:11434/v1", api_key=os.getenv("OLLAMA_API_KEY", "ollama"))
        elif ("gpt" in model_name):
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported OpenAI model: {model_name}")
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


class LLMFactory:
    _providers = {
        "gpt-4o-mini": OpenAIProvider,
        "gemini-1.5-flash": GoogleProvider,
        "llama3.2": OpenAIProvider, # OpenAIProvider handles Ollama models too
        "claude-sonnet-4-20250514": ClaudeProvider,
    }
    @classmethod
    def get_provider(cls, model_name):
        provider_cls = cls._providers.get(model_name)
        logger.debug(f"Creating LLM Model Class: {provider_cls.__name__}")
        if not provider_cls:
            raise ValueError(f"Unsupported LLM model: {model_name}")
        return provider_cls(model_name)

# --- Data Loader ---
class DataLoader:
    @staticmethod
    def load_hotpot_qa_data(name: str, subset: str, split: str, ingest_limit: int =None):
        logger.debug(f"Loading dataset: {name}, subset: {subset}, split: {split}")
        dataset = load_dataset(name, subset, split=split)
        if ingest_limit and ingest_limit > 0:
            logger.debug(f"Limiting Dataset size to: {ingest_limit} per the set ingesting limit.")
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
            logger.info(f"Deleted existing collection: {col_name}")
        except Exception:
            pass  # Collection doesn't exist. Not an error

        self.collection = self.client.get_or_create_collection(
            name=col_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": self.config.vector_store['distance_metric']}
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
        logger.debug(f"Loaded {len(data)} examples from dataset.")
        ids, documents, metadatas = DataLoader.process_context_data(data)
        self.vector_store.add_documents(ids, documents, metadatas)
    def setup_llm(self, model_name):
        self.llm_provider = LLMFactory.get_provider(model_name)
    def query(self, user_query):
        results = self.vector_store.query(user_query, self.config.retrieval["top_k_results"])
        documents = results["documents"][0]
        distances = results["distances"][0]
        logger.debug(f"Retrieved {len(documents)} documents with distances: {distances}")
        context_parts = []
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            context_parts.append(f"Document {i+1} (similarity: {1-distance:.3f}):\n{doc}\n")
        context = "\n".join(context_parts)
        if len(context) > self.config.retrieval["max_context_length"]:
            logger.debug(f"Context length ({len(context)}) exceeds max limit ({self.config.retrieval['max_context_length']}). Truncating.")
            context = context[:self.config.retrieval["max_context_length"]] + "..."
        return self.llm_provider.generate_response(
            context, user_query, self.config.llm_generation["system_prompt"]
        )

# --- Main ---
def main(log_level: str = "INFO"):
    load_dotenv()   # Load environment variables from .env file

    # Set up logging
    global logger
    log_file_nm = "log/rag_example_with_chromadb.log"
    logger= MyLogger(log_file_nm=log_file_nm, log_level=log_level).get_logger("llm_engineering")

    # Set path to the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'rag_example_config_with_chromadb.yaml')
    config_data = ConfigUtils.load_config(config_path)

    # Set embedding model based on user selection
    selected_emb = ConfigUtils.select_model(config_data, "embedding")
    config_data = ConfigUtils.update_config_with_embedding_model(config_data, selected_emb)
    logger.info(f"Using embedding model: {selected_emb}")

    # Set LLM model based on user selection
    selected_llm = ConfigUtils.select_model(config_data, "LLM")
    config_data = ConfigUtils.update_config_with_llm_model(config_data, selected_llm)
    logger.info(f"Using LLM model: {selected_llm}")

    rag_config = RAGConfig(config_data)
    pipeline = RAGPipeline(rag_config)

    logger.debug("Loading and indexing data (this may take a while)...")
    pipeline.load_and_index_data()

    pipeline.setup_llm(rag_config.llm_model)    #selected_llm)

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
            logger.info(f"\nAnswer:\n{response} ** For query: {user_query} **\n")
        except Exception as e:
            err_msg = f"Error processing query: {e}"
            logger.error(err_msg, exc_info=True)


if __name__ == "__main__":
    log_level = input(f"Input the preferred logging level to be one of: DEBUG | Info | Error: ")
    if log_level.lower() not in ["debug", "info", "warning", "error", "critical"]:
        print(f"Invalid log level '{log_level}'. Defaulting to INFO.")
        log_level = "INFO"
    main(log_level=log_level.upper())