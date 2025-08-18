import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from datasets import load_dataset
import openai
import google.generativeai as genai
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv  # RRM Code change to load environment variables from .env file

# Load configuration
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "data": {
            "dataset_name": "hotpot_qa",
            "config_name": "distractor",
            "split": "train"
        },
        "chromadb": {
            "persist_directory": "./chroma_db",
            "collection_name": "hotpot_qa_collection",
            "distance_metric": "cosine"
        },
        "embedding": {
            "dimension_limit": 1024
        },
        "llm": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }


# Abstract base classes for extensibility
class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        pass


class LLMModel(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response from the LLM."""
        pass


# Embedding Model Implementations
class OpenAIEmbedding(EmbeddingModel):
    """OpenAI text-embedding-3-small implementation."""

    def __init__(self, api_key: str, dimension_limit: int = 1024):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"
        self.dimension_limit = dimension_limit

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimension_limit
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimension_limit
        )
        return [data.embedding for data in response.data]


class GeminiEmbedding(EmbeddingModel):
    """Google Gemini embedding implementation."""

    def __init__(self, api_key: str, dimension_limit: int = 1024):
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"  # Updated model name
        self.dimension_limit = dimension_limit

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            output_dimensionality=self.dimension_limit
        )
        return result['embedding']

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                output_dimensionality=self.dimension_limit
            )
            embeddings.append(result['embedding'])
        return embeddings


class ChromaDBDefaultEmbedding(EmbeddingModel):
    """ChromaDB default embedding (all-MiniLM-L6-v2)."""

    def __init__(self):
        # ChromaDB's default embedding function
        from chromadb.utils import embedding_functions
        self.ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.ef([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        return self.ef(texts)


# LLM Model Implementations
class OpenAILLM(LLMModel):
    """OpenAI GPT implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1000):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        """Generate response from OpenAI GPT."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class GeminiLLM(LLMModel):
    """Google Gemini implementation."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7, max_tokens: int = 1000):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        """Generate response from Gemini."""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return response.text


class ClaudeLLM(LLMModel):
    """Anthropic Claude implementation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.7,
                 max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # TODO: Human input required here - Install anthropic client library
        # pip install anthropic

    def generate_response(self, prompt: str) -> str:
        """Generate response from Claude."""
        # TODO: Human input required here - Implement Claude API call
        # This requires the anthropic library and proper API implementation
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text


class LlamaLLM(LLMModel):
    """Local Llama implementation via Ollama."""

    def __init__(self, api_key: str = "ollama", model: str = "llama3.2", temperature: float = 0.7,
                 max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "http://localhost:11434"

    def generate_response(self, prompt: str) -> str:
        """Generate response from local Llama via Ollama."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"


# Factory functions for extensibility
def create_embedding_model(model_choice: str, config: Dict[str, Any]) -> EmbeddingModel:
    """Factory function to create embedding models."""
    dimension_limit = config["embedding"]["dimension_limit"]

    if model_choice == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIEmbedding(api_key, dimension_limit)

    elif model_choice == "gemini":
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return GeminiEmbedding(api_key, dimension_limit)

    elif model_choice == "chromadb_default":
        return ChromaDBDefaultEmbedding()

    else:
        raise ValueError(f"Unsupported embedding model: {model_choice}")


def create_llm_model(model_choice: str, config: Dict[str, Any]) -> LLMModel:
    """Factory function to create LLM models."""
    temperature = config["llm"]["temperature"]
    max_tokens = config["llm"]["max_tokens"]

    if model_choice == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAILLM(api_key, "gpt-4o-mini", temperature, max_tokens)

    elif model_choice == "gemini":
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return GeminiLLM(api_key, "gemini-1.5-flash", temperature, max_tokens)

    elif model_choice == "claude":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ClaudeLLM(api_key, "claude-sonnet-4-20250514", temperature, max_tokens)

    elif model_choice == "llama":
        api_key = os.getenv('OLLAMA_API_KEY', "ollama")
        return LlamaLLM(api_key, "llama3.2", temperature, max_tokens)

    else:
        raise ValueError(f"Unsupported LLM model: {model_choice}")


# Main RAG Pipeline Class
class RAGPipeline:
    """Complete RAG pipeline implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self.llm_model = None
        self.chroma_client = None
        self.collection = None

    def setup_chromadb(self):
        """Initialize ChromaDB with persistence."""
        persist_dir = self.config["chromadb"]["persist_directory"]

        # Create ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        print(f"ChromaDB initialized with persistence directory: {persist_dir}")

    def load_and_process_data(self):
        """Load and process the HotpotQA dataset."""
        print("Loading HotpotQA dataset...")

        # Load dataset
        dataset = load_dataset(
            self.config["data"]["dataset_name"],
            self.config["data"]["config_name"]
        )
        train_data = dataset[self.config["data"]["split"]]
        print(f"Loaded {len(train_data)} examples from HotpotQA")
        ingest_limit = self.config["data"].get("ingest_limit", None)  # RRM Code change: For demo purposes Limit the number of examples to process
        if ingest_limit and ingest_limit > 0:   # RRM Code change: For demo purposes Limit the number of examples to process
            train_data = train_data.select(range(ingest_limit))
        print(f"Limiting to first {ingest_limit} examples for ingestion. Processing {len(train_data)} examples.")   # RRM Code Change: print for info

        # Process data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for idx, example in enumerate(train_data):
            # Use document id for collection.ids
            doc_id = example.get('id', f'doc_{idx}')
            ids.append(doc_id)

            # Concatenate title and sentences from context
            context_parts = []
            for i, (title, sentences) in enumerate(zip(example['context']['title'], example['context']['sentences'])):
                # Join sentences list into a single string
                sentences_str = ' '.join(sentences) if isinstance(sentences, list) else sentences
                context_parts.append(f"{title} : {sentences_str}")

            # Join all context parts
            document_text = ', '.join(context_parts)
            documents.append(document_text)

            # Optional: Store metadata with type and level from supporting_facts
            # Note: HotpotQA structure may vary, adapting to available fields
            metadata = {
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
                'level': example.get('level', 'unknown'),
                'type': example.get('type', 'unknown')
            }
            metadatas.append(metadata)

            # Limit processing for demonstration (remove this in production)
            if idx >= 1000:  # Process first 1000 examples
                break

        return ids, documents, metadatas

    def create_collection(self, embedding_model: EmbeddingModel):
        """Create ChromaDB collection with the selected embedding model."""
        collection_name = self.config["chromadb"]["collection_name"]

        # Create embedding function wrapper for ChromaDB
        class EmbeddingFunction:
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model

            def __call__(self, input):
                if isinstance(input, str):
                    return [self.embedding_model.embed_text(input)]
                elif isinstance(input, list):
                    return self.embedding_model.embed_texts(input)

        embedding_function = EmbeddingFunction(embedding_model)

        try:    # RRM Code change: Given that we use multiple embedding models delete collection if it exists
            self.chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
            # RRM Code change: Commented out the get_collection as we delete the collection if it exists

            # self.collection = self.chroma_client.get_collection(
            #     name=collection_name,
            #     embedding_function=embedding_function
            # )
            # print(f"Using existing collection: {collection_name}")

        except Exception:   # RRM Code Change:Possible that the collection doesn't exist in which case create it.
            pass    # RRM Code change: Ignore error if collection doesn't exist
        # Create new collection with cosine similarity
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Created new collection: {collection_name}")

        # Load and add data to collection
        ids, documents, metadatas = self.load_and_process_data()

        print(f"Adding {len(documents)} documents to collection...")
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"Added batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

        print("Data loading completed!")

    def retrieve_context(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant context for the user query."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")

        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Extract and format the retrieved documents
        contexts = results['documents'][0] if results['documents'] else []

        if not contexts:
            return "No relevant context found."

        # Format the retrieved context
        formatted_context = "\n\n".join([f"Context {i + 1}: {doc}" for i, doc in enumerate(contexts)])
        return formatted_context

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using the selected LLM."""
        if not self.llm_model:
            raise ValueError("LLM model not initialized.")

        # Create RAG prompt
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please state that clearly.

Context:
{context}

Question: {query}

Answer:"""

        return self.llm_model.generate_response(prompt)

    def query(self, user_query: str) -> str:
        """Complete RAG pipeline query."""
        print(f"\nProcessing query: {user_query}")

        # Retrieve relevant context
        print("Retrieving relevant context...")
        context = self.retrieve_context(user_query)

        # Generate response
        print("Generating response...")
        response = self.generate_response(user_query, context)

        return response


def get_user_choice(prompt: str, options: List[str]) -> str:
    """Get user choice from a list of options."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input("Enter your choice (number): ")) - 1
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def main():
    """Main function to run the RAG pipeline."""
    print("=== RAG Pipeline Demo ===")

    # RRM Code change: Load environment variables from .env file
    load_dotenv()  # Load environment variables from .env file

    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))  # RRM Code change to set config path relative to script directory
    config_path = os.path.join(script_dir, 'rag_example_config_sonnet4_with_chromadb.yaml')  # RRM Code change to set config path relative to script directory
    config = load_config(config_path)   # RRM Code change to set config path relative to script directory
    #config = load_config()

    # Initialize RAG pipeline
    rag = RAGPipeline(config)

    # Setup ChromaDB
    rag.setup_chromadb()

    # Get user choice for embedding model
    embedding_options = [
        "text-embedding-3-small (OpenAI)",
        "gemini-embedding-001 (Google)",
        "all-MiniLM-L6-v2 (ChromaDB Default)"
    ]
    embedding_choice = get_user_choice("Choose an embedding model:", embedding_options)

    # Map choice to model key
    embedding_map = {
        embedding_options[0]: "openai",
        embedding_options[1]: "gemini",
        embedding_options[2]: "chromadb_default"
    }

    # Create embedding model
    try:
        rag.embedding_model = create_embedding_model(embedding_map[embedding_choice], config)
        print(f"✓ Embedding model initialized: {embedding_choice}")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # Create collection
    rag.create_collection(rag.embedding_model)

    # Get user choice for LLM model
    llm_options = [
        "gpt-4o-mini (OpenAI)",
        "gemini-1.5-flash (Google)",
        "llama3.2 (Local via Ollama)",
        "claude-sonnet-4-20250514 (Anthropic)"
    ]
    llm_choice = get_user_choice("Choose an LLM model:", llm_options)

    # Map choice to model key
    llm_map = {
        llm_options[0]: "openai",
        llm_options[1]: "gemini",
        llm_options[2]: "llama",
        llm_options[3]: "claude"
    }

    # Create LLM model
    try:
        rag.llm_model = create_llm_model(llm_map[llm_choice], config)
        print(f"✓ LLM model initialized: {llm_choice}")
    except Exception as e:
        print(f"Error initializing LLM model: {e}")
        return

    # Interactive query loop
    print("\n=== Ready for queries! ===")
    print("Enter 'quit' to exit.")

    while True:
        user_query = input("\nEnter your query: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_query:
            print("Please enter a valid query.")
            continue

        try:
            response = rag.query(user_query)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    # Create default config file if it doesn't exist
    # RRM Code change: The config path is set relative to the script directory in main
    # config_path = "config.yaml"    # RRM Code change to set config path relative to script directory
    # if not os.path.exists(config_path):
    #     default_config = get_default_config()
    #     with open(config_path, 'w') as f:
    #         yaml.dump(default_config, f, default_flow_style=False)
    #     print(f"Created default config file: {config_path}")

    main()