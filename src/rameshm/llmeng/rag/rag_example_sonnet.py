import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datasets import load_dataset
import openai
import google.generativeai as genai
import anthropic
import requests

from rameshm.llmeng.utils import init_utils # RRM Code change to import init_utils from rameshm.llmeng.utils

# Initialize the logger and sets environment variables
logger = init_utils.set_environment_logger() # RRM Code change to set logger and environment variables

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline - stores full config with easy access"""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with full configuration dictionary"""
        self._config = config_dict

        # Cache commonly accessed values for convenience
        self.embedding_model = self.get('embedding_model', 'all-MiniLM-L6-v2')
        self.llm_model = self.get('llm_model', 'gpt-4o-mini')

        # Vector store config
        vector_store = self.get('vector_store', {})
        self.vector_store_path = vector_store.get('path', './chroma_db')
        self.collection_name = vector_store.get('collection_name', 'hotpot_qa_collection')

        # Embedding config
        embedding_config = self.get('embedding', {})
        self.embedding_dimensions = embedding_config.get('dimensions', 1024)

        # Retrieval config
        retrieval = self.get('retrieval', {})
        self.top_k_results = retrieval.get('top_k_results', 5)
        self.max_context_length = retrieval.get('max_context_length', 4000)

        # LLM generation config
        llm_gen = self.get('llm_generation', {})
        self.temperature = llm_gen.get('temperature', 0.7)
        self.system_prompt = llm_gen.get('system_prompt', "You are a helpful assistant.")

        # Store model info for easy access
        self.available_embedding_models = self.get('available_embedding_models', {})
        self.available_llm_models = self.get('available_llm_models', {})

    def get(self, key: str, default=None):
        """Get config value with default"""
        return self._config.get(key, default)

    def get_nested(self, *keys, default=None):
        """Get nested config value (e.g., get_nested('vector_store', 'path'))"""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def get_model_info(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        models_key = f"available_{model_type}_models"
        models = self.get(models_key, {})
        return models.get(model_name, {})

    def get_embedding_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get info for current or specified embedding model"""
        model_name = model_name or self.embedding_model
        return self.get_model_info('embedding', model_name)

    def get_llm_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get info for current or specified LLM model"""
        model_name = model_name or self.llm_model
        return self.get_model_info('llm', model_name)

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._config.update(updates)
        # Refresh cached values
        self.__init__(self._config)

    @property
    def full_config(self) -> Dict[str, Any]:
        """Access to full configuration dictionary"""
        return self._config.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create RAGConfig from dictionary"""
        return cls(config_dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def get_embedding_function(self, config: RAGConfig):
        """Returns the embedding function for ChromaDB"""
        pass


class DefaultEmbeddingProvider(EmbeddingProvider):
    """Default embedding provider using ChromaDB's built-in models"""

    def get_embedding_function(self, config: RAGConfig):
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""

    def get_embedding_function(self, config: RAGConfig):
        logger.debug(f"Using OpenAI embedding model: {config.embedding_model}")  # RRM Code change to log model name
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )

class CustomGoogleEmbeddingFunction:
    """Custom Google embedding function for ChromaDB"""

    def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)

    #def __call__(self, input_texts: List[str]) -> List[List[float]]: # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
    def __call__(self, input: List[str]) -> List[List[float]]: # RRM Code change to match ChromaDB expected signature (changed input_texts to input)
        """Generate embeddings for input texts"""
        logger.debug(f"In CustomGoogleEmbeddingFunction with model: {self.model_name} and input text Length: {len(input)}") # RRM Code change to log model name and input text length
        try:
            embeddings = []
            for text in input:  # RRM Code change to iterate over input texts instead of input_texts
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    output_dimensionality=1024,  # Limit to 1024 as requested
                    task_type="retrieval_document"
                )
                #print(f"result type: {type(result)} Result: {result} Embedding Type: {result['embedding']}") # RRM Code change to print result details
                embeddings.append(result['embedding'])  # RRM Code change to append embedding to list
            log_msg = f"Total embeddings generated: {len(embeddings)}. Embedding Dimension: {len(embeddings[0])}" # RRM Code change to print total embeddings generated
            print(f"{log_msg}")
            logger.debug(log_msg)
            return embeddings
        except Exception as e:
            logger.error(f"Google embedding error: {e}")
            raise


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google embedding provider"""

    # Mapping from config names to Google API model names
    MODEL_NAME_MAPPING = {
        "gemini-embedding-001": "models/embedding-001",
        "text-embedding-004": "models/text-embedding-004",  # Future model support
    }

    def get_embedding_function(self, config: RAGConfig):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        # Map from config model name to Google API model name
        config_model_name = config.embedding_model
        google_model_name = self.MODEL_NAME_MAPPING.get(
            config_model_name,
            "models/embedding-001"  # Default fallback
        )

        logger.info(f"Using Google model: {google_model_name} (config: {config_model_name})")

        # Use custom Google embedding function since ChromaDB might not have built-in support
        return CustomGoogleEmbeddingFunction(
            api_key=api_key,
            model_name=google_model_name
        )


class EmbeddingFactory:
    """Factory class for creating embedding providers"""

    _providers = {
        "all-MiniLM-L6-v2": DefaultEmbeddingProvider,
        "text-embedding-3-small": OpenAIEmbeddingProvider,
        "gemini-embedding-001": GoogleEmbeddingProvider,
    }

    @classmethod
    def get_provider(cls, model_name: str) -> EmbeddingProvider:
        """Get embedding provider by model name"""
        provider_class = cls._providers.get(model_name)
        if not provider_class:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        return provider_class()


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, context: str, query: str, system_prompt: Optional[str]) -> str: # RRM Code change to add system prompt as optional parameter
        """Generate response using the LLM"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = " ") -> str:   # RRM Code change to add system prompt as optional parameter
        logger.debug(f"Generating response Using OpenAI Model: {self.model_name}")  # RRM Code change to log model name
        prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

        try:
            system_prompt = system_prompt if system_prompt else "You are a helpful assistant that answers questions based on the provided context."
            system_prompt = system_prompt.replace("\n", ". ")    # RRM Code change to replace newlines with spaces in system prompt
            logger.debug(f"System Prompt: {system_prompt}")  # RRM Code change to log system prompt
            logger.debug(f"User Prompt: {prompt}")  # RRM Code change to log user prompt
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
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            #return f"Error generating response: {e}"


class GoogleProvider(LLMProvider):
    """Google Gemini LLM provider"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = " ") -> str:   # RRM Code change to add system prompt as optional parameter
        logger.debug(f"Generating response Using Google AI Model: {self.model_name}")  # RRM Code change to log model name
        system_prompt = system_prompt if system_prompt else "You are a helpful assistant that answers questions based on the provided context."  # RRM Code change to set default system prompt if not provided
        system_prompt = system_prompt.replace("\n",". ")  # RRM Code change to replace newlines with spaces in system prompt
        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""

        system_prompt = system_prompt if system_prompt else "You are a helpful assistant that answers questions based on the provided context." # RRM Code change to set default system prompt if not provided
        system_prompt = system_prompt.replace("\n",". ")  # RRM Code change to replace newlines with spaces in system prompt
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise   # RRM Code change to raise exception instead of returning error message
            #return f"Error generating response: {e}"


class ClaudeProvider(LLMProvider):
    """Claude LLM provider"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv('ANTHROPIC_API_KEY')    # RRM Code change to use ANTHROPIC_API_KEY environment variable
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = " ") -> str:   # RRM Code change to add system prompt as optional parameter
        logger.debug(f"Generating response Using Antropic AI Model: {self.model_name}")  # RRM Code change to log model name
        prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

        system_prompt = system_prompt if system_prompt else "You are a helpful assistant that answers questions based on the provided context." # RRM Code change to set default system prompt if not provided
        system_prompt = system_prompt.replace("\n",". ")  # RRM Code change to replace newlines with spaces in system prompt
        logger.debug(f"System Prompt: {system_prompt}")  # RRM Code change to log system prompt
        logger.debug(f"User Prompt: {prompt}")  # RRM Code change to log user prompt
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                system=system_prompt,   # RRM Code change to pass system prompt
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise   # RRM Code change to raise exception instead of returning error message
            #return f"Error generating response: {e}"


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.api_key = os.getenv('OLLAMA_API_KEY', 'ollama')

    def generate_response(self, context: str, query: str, system_prompt: Optional[str] = " ") -> str:   # RRM Code change to add system prompt as optional parameter
        logger.debug(f"Generating response Using Ollama AI Model: {self.model_name}")  # RRM Code change to log model name
        system_prompt = system_prompt if system_prompt else "You are a helpful assistant that answers questions based on the provided context."  # RRM Code change to set default system prompt if not provided
        system_prompt = system_prompt.replace("\n",". ")  # RRM Code change to replace newlines with spaces in system prompt

        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""

        logger.debug(f"System Prompt: {system_prompt}")  # RRM Code change to log system prompt
        logger.debug(f"User Prompt: {prompt}")  # RRM Code change to log user prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise   # RRM Code change to raise exception instead of returning error message
            # return f"Error generating response: {e}"


class LLMFactory:
    """Factory class for creating LLM providers"""

    _providers = {
        "gpt-4o-mini": OpenAIProvider,
        "gemini-1.5-flash": GoogleProvider,
        "llama3.2": OllamaProvider,
        "claude-sonnet-4-20250514": ClaudeProvider,
    }

    @classmethod
    def get_provider(cls, model_name: str) -> LLMProvider:
        """Get LLM provider by model name"""
        provider_class = cls._providers.get(model_name)
        if not provider_class:
            raise ValueError(f"Unsupported LLM model: {model_name}")
        return provider_class(model_name)


class DataLoader:
    """Class for loading and processing HotpotQA dataset"""

    @staticmethod
    def load_hotpot_qa_data():
        """Load HotpotQA dataset"""
        logger.info("Loading HotpotQA dataset...")
        try:
            dataset = load_dataset("hotpot_qa", "distractor")
            train_data = dataset["train"]
            logger.info(f"Loaded {len(train_data)} training examples")
            return train_data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @staticmethod
    def process_context_data(data, max_examples: Optional[int] = None):
        """Process context data for ChromaDB"""
        ids = []
        documents = []
        metadatas = []

        logger.info("Processing context data...")
        max_data_size = max_examples if max_examples and max_examples < len(data) else len(data)  # RRM Code change to limit data size

        try:  # RRM Code to handle errors in processing
            for idx, example in enumerate(data):
                if idx >= max_data_size:  # RRM Code change
                    logger.info(f"Reached max examples limit: {max_data_size}. Stopping data processing.")  # RRM Code change to log when max examples limit is reached
                    break
                # logger.debug(f"Processing example {idx + 1}/{len(data)}: {example['id']} Document Length: {len(documents)}") # RRM Code change to print progress
                context = example["context"]

                ctx_idx = 0
                titles = context["title"]
                sentences = context["sentences"]
                for ctx_idx, title in enumerate(titles):  # RRM Code change
                    doc_id = f"{example['id']}_{ctx_idx + 1}"
                    ids.append(doc_id)

                    # # Concatenate title and sentences
                    # title = ctx["title"]
                    # sentences = ctx["sentences"]

                    # Format: "title : sentence1, sentence2, ..."
                    sentences_str = ", ".join(sentences[ctx_idx])
                    document = f"{title} : {sentences_str}"
                    documents.append(document)

                    # Optional metadata
                    metadata = {
                        "type": example.get("type", "unknown"),  # RRM Code change changed from ctx to example
                        "level": example.get("level", "unknown"),
                        "original_id": example["id"]
                    }
                    metadatas.append(metadata)
        except Exception as e:  # RRM Code Added to handle errors in processing
            logger.error(f"Error processing context data: {e}")
            raise   # RRM Code change to raise exception instead of returning error message

        logger.info(f"Processed {len(documents)} documents")
        return ids, documents, metadatas


class VectorStore:
    """ChromaDB vector store wrapper"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = chromadb.PersistentClient(
            path=config.vector_store_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

    def create_collection(self, embedding_function):
        """Create or get ChromaDB collection"""
        try:
            # Delete existing collection if it exists (for fresh start)
            try:
                self.client.delete_collection(name=self.config.collection_name)
                logger.info(f"Deleted existing collection: {self.config.collection_name}")
            except Exception:
                pass  # Collection doesn't exist

            # Create new collection with cosine similarity
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created collection: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        """Add documents to the collection"""
        if not self.collection:
            raise ValueError("Collection not initialized")

        try:
            # Add documents in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                logger.info(f"Added batch {i // batch_size + 1}: {len(batch_docs)} documents")

            logger.info(f"Successfully added {len(documents)} documents to collection")

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(self, query_text: str, n_results: int = None) -> Dict[str, Any]:
        """Query the collection"""
        if not self.collection:
            raise ValueError("Collection not initialized")

        n_results = n_results or self.config.top_k_results

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise


class RAGPipeline:
    """Main RAG pipeline class"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.vector_store = VectorStore(self.config)
        self.llm_provider = None
        self.data_loaded = False

    def _load_config(self, config_path: str) -> RAGConfig:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            return RAGConfig.from_dict(config_data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def setup_embedding_model(self):
        """Setup embedding model"""
        logger.info(f"Setting up embedding model: {self.config.embedding_model}")
        embedding_provider = EmbeddingFactory.get_provider(self.config.embedding_model)
        embedding_function = embedding_provider.get_embedding_function(self.config)

        self.vector_store.create_collection(embedding_function)

    def load_and_index_data(self):
        """Load and index the HotpotQA data"""
        if self.data_loaded:
            logger.info("Data already loaded")
            return

        # Load data
        data = DataLoader.load_hotpot_qa_data()

        # Process data
        dataset_config = self.config.get("dataset", None)
        #print(f"Dataset Config Type: {type(dataset_config)} Dataset Config Contents: {dataset_config}")  # RRM Code change to print dataset config
        max_examples = dataset_config.get("max_examples", None)
        ids, documents, metadatas = DataLoader.process_context_data(data, max_examples)

        # Add to vector store
        self.vector_store.add_documents(ids, documents, metadatas)
        self.data_loaded = True
        logger.info("Data indexing completed")

    def setup_llm(self, model_name: str):
        """Setup LLM provider"""
        logger.info(f"Setting up LLM: {model_name}")
        self.llm_provider = LLMFactory.get_provider(model_name)

    def query(self, user_query: str) -> str:
        """Process user query and generate response"""
        if not self.llm_provider:
            raise ValueError("LLM not initialized")

        # Retrieve relevant context
        logger.info(f"Retrieving context for query: {user_query}")
        results = self.vector_store.query(user_query)

        # Extract context
        documents = results["documents"][0]  # First query results
        distances = results["distances"][0]

        # Combine retrieved documents as context
        context_parts = []
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            context_parts.append(f"Document {i + 1} (similarity: {1 - distance:.3f}):\n{doc}\n")

        context = "\n".join(context_parts)

        # Limit context length if needed
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length] + "..."

        logger.info(f"Retrieved {len(documents)} relevant documents")

        # Generate response
        response = self.llm_provider.generate_response(context, user_query, self.config.system_prompt) # RRM Code change to pass system prompt from config
        return response


def get_user_choice(options: Dict[str, str], prompt: str) -> str:
    """Get user choice from a list of options"""
    print(f"\n{prompt}")
    for key, value in options.items():
        print(f"{key}: {value}")

    while True:
        choice = input("\nEnter your choice: ").strip()
        if choice in options:
            return choice
        print("Invalid choice. Please try again.")


def display_model_info(config: RAGConfig):
    """Display available models with their information"""
    print("\n=== Available Embedding Models ===")
    if config.available_embedding_models:
        for i, (model_key, model_info) in enumerate(config.available_embedding_models.items(), 1):
            desc = model_info.get('description', 'No description')
            provider = model_info.get('provider', 'Unknown')
            api_required = "API Key Required" if model_info.get('api_key_required', False) else "No API Key"
            print(f"{i}. {model_key}")
            print(f"   Provider: {provider}")
            print(f"   Description: {desc}")
            print(f"   {api_required}")
            print()

    print("\n=== Available LLM Models ===")
    if config.available_llm_models:
        for i, (model_key, model_info) in enumerate(config.available_llm_models.items(), 1):
            desc = model_info.get('description', 'No description')
            provider = model_info.get('provider', 'Unknown')
            api_required = "API Key Required" if model_info.get('api_key_required', False) else "No API Key"
            cost_tier = model_info.get('cost_tier', 'Unknown')
            print(f"{i}. {model_key}")
            print(f"   Provider: {provider}")
            print(f"   Description: {desc}")
            print(f"   Cost Tier: {cost_tier}")
            print(f"   {api_required}")
            print()


def get_model_choices_from_config(config: RAGConfig):
    """Generate model choice dictionaries from config"""
    embedding_options = {}
    llm_options = {}

    if config.available_embedding_models:
        for i, model_key in enumerate(config.available_embedding_models.keys(), 1):
            embedding_options[str(i)] = model_key
    else:
        # Fallback to hardcoded options
        embedding_options = {
            "1": "all-MiniLM-L6-v2",
            "2": "text-embedding-3-small",
            "3": "gemini-embedding-001"
        }

    if config.available_llm_models:
        for i, model_key in enumerate(config.available_llm_models.keys(), 1):
            llm_options[str(i)] = model_key
    else:
        # Fallback to hardcoded options
        llm_options = {
            "1": "gpt-4o-mini",
            "2": "gemini-1.5-flash",
            "3": "llama3.2",
            "4": "claude-sonnet-4-20250514"
        }

    return embedding_options, llm_options


def main():
    """Main function"""
    print("=== RAG Pipeline Demo ===")

    try:
        # Load configuration
        # Set file path to be same as the script directory.
        script_dir = os.path.dirname(os.path.abspath(__file__)) # RRM Code change to set config path relative to script directory
        config_path = os.path.join(script_dir, 'rag_config_sonnet_Aug10.yaml') # RRM Code change to set config path relative to script directory
        #config_path = "rag_config_sonnet_Aug10.yaml"

        # Check if config file exists, create default if not
        if not os.path.exists(config_path):
            create_default_config(config_path)

        # Initialize pipeline
        pipeline = RAGPipeline(config_path)

        # Display available models
        display_model_info(pipeline.config)

        # Get user choices for models
        embedding_options, llm_options = get_model_choices_from_config(pipeline.config)

        # Get embedding model choice
        embedding_choice = get_user_choice(
            embedding_options,
            "Select embedding model:"
        )
        pipeline.config.embedding_model = embedding_options[embedding_choice]

        # Get LLM choice
        llm_choice = get_user_choice(
            llm_options,
            "Select LLM model:"
        )
        selected_llm = llm_options[llm_choice]

        # Setup models
        print("\nInitializing models...")
        pipeline.setup_embedding_model()
        pipeline.setup_llm(selected_llm)

        # Load and index data
        print("\nLoading and indexing data...")
        pipeline.load_and_index_data()

        # Interactive query loop
        print("\n=== Ready for queries ===")
        print("Type 'quit' to exit")

        while True:
            user_query = input("\nEnter your question: ").strip()

            if user_query.lower() == 'quit':
                break

            if not user_query:
                print("Please enter a valid question.")
                continue

            try:
                print("\nProcessing your query...")
                response = pipeline.query(user_query)
                print(f"\nAnswer:\n{response}")
                logger.info(f"\nAnswer:\n{response}")   # RRM Code change to log the response
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise   # RRM Code change to raise exception instead of returning error message

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")


def create_default_config(config_path: str):
    """Create default configuration file"""
    default_config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gpt-4o-mini",
        "vector_store": {
            "path": "./chroma_db",
            "collection_name": "hotpot_qa_collection",
            "distance_metric": "cosine"
        },
        "embedding": {
            "dimensions": 1024,
            "batch_size": 1000
        },
        "retrieval": {
            "top_k_results": 5,
            "max_context_length": 4000,
            "similarity_threshold": 0.0
        },
        "llm_generation": {
            "temperature": 0.7,
            "max_response_tokens": 1000,
            "system_prompt": "You are a helpful assistant that answers questions based on the provided context."
        },
        "available_embedding_models": {
            "all-MiniLM-L6-v2": {
                "provider": "chromadb_default",
                "description": "Local sentence transformer model, good performance, no API costs",
                "api_key_required": False,
                "dimensions": 384
            },
            "text-embedding-3-small": {
                "provider": "openai",
                "description": "OpenAI's efficient embedding model with good performance",
                "api_key_required": True,
                "api_key_env": "OPENAI_API_KEY",
                "dimensions": 1536
            },
            "gemini-embedding-001": {
                "provider": "google",
                "description": "Google's embedding model optimized for semantic similarity",
                "api_key_required": True,
                "api_key_env": "GOOGLE_API_KEY",
                "dimensions": 768
            }
        },
        "available_llm_models": {
            "gpt-4o-mini": {
                "provider": "openai",
                "description": "OpenAI's efficient GPT-4 variant, good for most tasks",
                "api_key_required": True,
                "api_key_env": "OPENAI_API_KEY",
                "cost_tier": "low"
            },
            "gemini-1.5-flash": {
                "provider": "google",
                "description": "Google's fast and efficient Gemini model",
                "api_key_required": True,
                "api_key_env": "GOOGLE_API_KEY",
                "cost_tier": "low"
            },
            "claude-sonnet-4-20250514": {
                "provider": "anthropic",
                "description": "Claude Sonnet 4, excellent reasoning and analysis",
                "api_key_required": True,
                "api_key_env": "ANTHROPIC_API_KEY",
                "cost_tier": "medium"
            },
            "llama3.2": {
                "provider": "ollama",
                "description": "Meta's LLaMA 3.2, runs locally via Ollama",
                "api_key_required": False,
                "endpoint": "http://localhost:11434",
                "cost_tier": "free"
            }
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Created default configuration file: {config_path}")


if __name__ == "__main__":
    main()