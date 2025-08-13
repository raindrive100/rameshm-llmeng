import os
import json
import logging
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from dotenv import load_dotenv

from google.generativeai import GenerativeModel, configure
from openai import OpenAI
from anthropic import Anthropic
from ollama import Client as OllamaClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class EmbeddingModelFactory:
    """
    A factory for creating embedding functions based on configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_embedding_function(self, model_name: str):
        """
        Returns the appropriate embedding function based on the model name.
        """
        logging.info(f"Selected embedding model: {model_name}")    # RRM Code change: Added print statement to show selected embedding model
        if model_name == self.config["EMBEDDING_MODELS"]["GOOGLE"]:
            return self._create_google_embedding_function()
        elif model_name == self.config["EMBEDDING_MODELS"]["OPENAI"]:
            return self._create_openai_embedding_function()
        elif model_name == self.config["EMBEDDING_MODELS"]["CHROMA"]:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config["EMBEDDING_MODELS"]["CHROMA"]
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")

    def _create_google_embedding_function(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
        configure(api_key=google_api_key)
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=google_api_key,
            model_name=self.config["EMBEDDING_MODELS"]["GOOGLE"],
            #output_dimensionality=self.config["EMBEDDING_MODELS"]["DIMENSIONS"]    # RRM Code change: Removed output_dimensionality as it is not a parameter for GoogleGenerativeAiEmbeddingFunction
        )

    def _create_openai_embedding_function(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=self.config["EMBEDDING_MODELS"]["OPENAI"],
            #embedding_dim=self.config["EMBEDDING_MODELS"]["DIMENSIONS"]    # RRM Code change: Removed embedding_dim as it is not a parameter for OpenAIEmbeddingFunction
        )


class LLMFactory:
    """
    A factory for creating LLM clients based on configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_llm_client(self, llm_model: str):  # RRM Code change: changed paramter from llm_choice to llm_model
        """
        Returns the appropriate LLM client based on the user's choice.
        """
        # llm_model = self.config["LLM_PROVIDERS"].get(llm_choice.upper())  # RRM Code change: llm_model is now passed directly instead of llm_choice
        # if not llm_model:
        #     raise ValueError(f"Unsupported LLM choice: {llm_choice}")

        if "gpt" in llm_model:
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "gemini" in llm_model:
            configure(api_key=os.getenv("GOOGLE_API_KEY"))  # RRM Code change: Added configure call to set Google API key which was missing.
            return GenerativeModel(llm_model)
        elif "llama" in llm_model:
            return OllamaClient(host=os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
        elif "claude" in llm_model:
            return Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        exit()


def get_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Loads and preprocesses the data from Hugging Face.
    """
    logging.info("Loading and processing dataset...")
    dataset = load_dataset(config["DATASET"]["NAME"], config["DATASET"]["INSTANCE"], split=config["DATASET"]["SPLIT"])

    processed_data = []
    docs_to_process = config["DATASET"].get("DOCS_TO_PROCESS", None)    # RRM Code change: Adding code to process subset for testing
    docs_processed = 0   # RRM Code change: Adding code to process subset for testing
    for entry in dataset:
        context = entry['context']

        # Concatenate title and sentences for collection.documents
        documents_list = []
        for title, sentences in zip(context['title'], context['sentences']):
            # Ensure sentences is a list of strings and join them
            if isinstance(sentences, list):
                sentences = ' '.join(sentences)
            documents_list.append(f"{title} : {sentences}")

        # Join all title:sentence pairs into a single string
        document = ", ".join(documents_list)

        # Extract metadata
        metadata = {
            "type": entry.get('type', []),    # RRM Code change: "type" is at "entry" level, not "context"
            "level": entry.get('level', [])   # RRM Code change: "type" is at "entry" level, not "context"
        }

        # Use the ID from the first context item
        doc_id = entry['id'] if entry['id'] else None    # RRM Code change: "id" is at "entry" level, not "context"

        if doc_id:
            processed_data.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata
            })

        docs_processed += 1  # RRM Code change: Adding code to process subset for testing
        if docs_to_process and docs_processed >= docs_to_process:    # RRM Code change: Adding code to process subset for testing
            logging.info(f"Processed {docs_processed} documents, stopping as per configuration.")
            break
        else:
            continue

    return processed_data


def setup_chroma_db(config: Dict[str, Any], embedding_function):
    """
    Sets up and persists the ChromaDB collection.
    """
    logging.info("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=config["VECTOR_STORE"]["PATH"])

    # RRM Code Change: For testing purposes. Delete collection if it exists.
    col_name = config["VECTOR_STORE"]["COLLECTION_NAME"]
    try:    # RRM Code change - Deleting existing collection for demo purposes
        # If collection exists, delete it for a fresh start (for demo purposes)
        client.delete_collection(name=col_name)
        logging.info(f"Deleted existing collection: {col_name}")
    except Exception:
        pass  # Collection doesn't exist


    # Define collection metadata for cosine similarity
    collection = client.get_or_create_collection(
        name=col_name,  # RRM Code change: Use collection name from config
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def populate_chroma_db(collection, processed_data: List[Dict[str, Any]]):
    """
    Adds documents to the ChromaDB collection if it's empty.
    """
    if collection.count() == 0: # RRM Code change: For demo purposes Collection is always deleted and recreated, so it will always be empty
        logging.info("Populating ChromaDB with documents...")
        ids = [item["id"] for item in processed_data]
        documents = [item["document"] for item in processed_data]
        metadatas = [item["metadata"] for item in processed_data]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logging.info(f"Added {len(documents)} documents to ChromaDB.")
    else:
        logging.info("ChromaDB already populated. Skipping data insertion.")


def retrieve_context(collection, user_query: str, n_results: int = 5) -> List[str]:
    """
    Retrieves the most relevant documents from the ChromaDB collection.
    """
    results = collection.query(
        query_texts=[user_query],
        n_results=n_results
    )
    return results['documents'][0] if results['documents'] else []


def generate_response(llm_client, llm_model: str, user_query: str, retrieved_context: List[str]) -> str:
    """
    Generates a response using the selected LLM and retrieved context.
    """
    context_str = "\n".join(retrieved_context)
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the user's question. "
        f"If you don't know the answer, just say you don't have enough information.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    logging.info("Generating response with LLM...")

    if isinstance(llm_client, OpenAI):
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # TODO: Human input required here for Gemini's API call, as there are various ways to call it (e.g., chat vs. generate_content).
    # For now, we'll use a placeholder.
    elif isinstance(llm_client, GenerativeModel):
        response = llm_client.generate_content(prompt)
        return response.text
        #return f"# TODO: Gemini API call needs to be implemented. The model {llm_model} was selected. Placeholder prompt: {prompt}"

    elif isinstance(llm_client, OllamaClient):
        response = llm_client.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    # TODO: Human input required here for Claude's API call, as there are various ways to call it (e.g., with different message formats and system prompts).
    # For now, we'll use a placeholder.
    elif isinstance(llm_client, Anthropic):
        response = llm_client.messages.create(
            model=llm_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        #return f"# TODO: Claude API call needs to be implemented. The model {llm_model} was selected. Placeholder prompt: {prompt}"

    return "Error: Unsupported LLM client."


def main():
    """
    Main function to run the RAG pipeline.
    """
    # 1-a Disable anonymous usage tracking by ChromaDB
    os.environ["CHROMA_DISABLE_ANONYMOUS_USAGE_TRACKING"] = "1"  # RRM Code change to disable anonymous usage tracking

    # 1-b Set the path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # RRM Code change to set config path relative to script directory
    config_path = os.path.join(script_dir, 'rag_config_gemini25.json')  # RRM Code change to set config path relative to script directory

    # 1. Load configuration
    config = load_config(config_path) # RRM Code change: changed config file name from config.jsom to config_path

    # 2. Select embedding model
    print("Choose an embedding model:")
    print(f"1. Google ({config['EMBEDDING_MODELS']['GOOGLE']})")
    print(f"2. OpenAI ({config['EMBEDDING_MODELS']['OPENAI']})")
    print(f"3. ChromaDB Default ({config['EMBEDDING_MODELS']['CHROMA']})")
    embed_choice = input("Enter your choice (1, 2, or 3): ")

    if embed_choice == '1':
        embed_model_name = config["EMBEDDING_MODELS"]["GOOGLE"]
    elif embed_choice == '2':
        embed_model_name = config["EMBEDDING_MODELS"]["OPENAI"]
    elif embed_choice == '3':
        embed_model_name = config["EMBEDDING_MODELS"]["CHROMA"]
    else:
        logging.error("Invalid embedding model choice. Exiting.")
        return

    # 3. Initialize embedding model and vector store
    try:
        embedding_factory = EmbeddingModelFactory(config)
        embedding_function = embedding_factory.get_embedding_function(embed_model_name)
        collection = setup_chroma_db(config, embedding_function)
    except ValueError as e:
        logging.error(f"Error initializing embedding model or vector store: {e}") # RRM Code change: Added exc_info=True to log the stack trace
        raise   #return RRM Code change: Changed from return to raise to propagate the exception

    # 4. Load and populate data
    data = get_data(config)
    populate_chroma_db(collection, data)

    # 5. Select LLM
    print("\nChoose an LLM for response generation:")
    print(f"1. GPT-4o-mini ({config['LLM_PROVIDERS']['GPT']})")
    print(f"2. Gemini-1.5-flash ({config['LLM_PROVIDERS']['GEMINI']})")
    print(f"3. LLaMA 3.2 (local via Ollama, {config['LLM_PROVIDERS']['LLAMA']})")
    print(f"4. Claude Sonnet ({config['LLM_PROVIDERS']['CLAUDE']})")
    llm_choice = input("Enter your choice (1, 2, 3, or 4): ")

    if llm_choice == '1':
        llm_model_name = config['LLM_PROVIDERS']['GPT']
    elif llm_choice == '2':
        llm_model_name = config['LLM_PROVIDERS']['GEMINI']
    elif llm_choice == '3':
        llm_model_name = config['LLM_PROVIDERS']['LLAMA']
    elif llm_choice == '4':
        llm_model_name = config['LLM_PROVIDERS']['CLAUDE']
    else:
        logging.error("Invalid LLM choice. Exiting.")
        return
    logging.info(f"Using LLM model: {llm_model_name}")  # RRM Code change: Added print statement to show selected LLM model

    try:
        llm_factory = LLMFactory(config)
        llm_client = llm_factory.get_llm_client(llm_model_name) # RRM Code change: changed parameter from llm_choice to llm_model_name
    except ValueError as e:
        logging.error(f"Error initializing LLM client: {e}")
        raise #return RRM Code change: Changed from return to raise to propagate the exception

    # 6. Main RAG loop
    while True:
        user_query = input("\nEnter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break

        # 7. Retrieve context
        retrieved_context = retrieve_context(collection, user_query)
        if not retrieved_context:
            print("No relevant context found. Please try a different query.")
            continue

        # 8. Generate and print response
        try:
            response = generate_response(llm_client, llm_model_name, user_query, retrieved_context)
            print("\n" + "=" * 50)
            print("Generated Response:")
            print(response)
            print("=" * 50)
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)  # RRM Code change: Added exc_info=True to log the stack trace


if __name__ == "__main__":
    main()