# rag_pipeline.py

import os
import yaml
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from typing import Dict, List, Any
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from ollama import Client as OllamaClient
import time
from tqdm import tqdm

from dotenv import load_dotenv  # RRM Code change to load environment variables from .env file


# --- Configuration and Environment Setup ---
load_dotenv()   # RRM Code change to load environment variables from .env file
# Load the configuration from the YAML file
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

script_dir = os.path.dirname(os.path.abspath(__file__)) # RRM Code change to set config path relative to script directory
config_path = os.path.join(script_dir, 'rag_example_config_gemini25_with_chromadb.yaml') # RRM Code change to set config path relative to script directory
CONFIG = load_config(config_path)   # RRM Code change to load config from YAML file with relative path
#CONFIG = load_config("config.yaml")

# Check and set API keys from environment variables
def get_api_key(provider_name: str, env_var: str) -> str:
    """Retrieves an API key from an environment variable."""
    key = os.getenv(env_var)
    if not key:
        print(f"Warning: {provider_name} API key not found. Please set the '{env_var}' environment variable.")
        return None
    return key


# --- Embedding Model Management ---
def get_embedding_function(model_name: str) -> embedding_functions.EmbeddingFunction:
    """
    Factory function to create the appropriate embedding function based on the model name.
    This architecture supports adding new embedding models by extending the factory.
    """
    if model_name == 'gemini-embedding-001':
        api_key = get_api_key('Google', 'GOOGLE_API_KEY')
        if not api_key:
            # TODO: Human input required here if key is not available
            raise ValueError("Google API key is required for gemini-embedding-001.")
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            output_dimensionality=CONFIG['embedding_output_dimensionality']
        )
    elif model_name == 'text-embedding-3-small':
        api_key = get_api_key('OpenAI', 'OPENAI_API_KEY')
        if not api_key:
            # TODO: Human input required here if key is not available
            raise ValueError("OpenAI API key is required for text-embedding-3-small.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            #embedding_dim=CONFIG['embedding_output_dimensionality']    # RRM Code Change: Use embedding_dim is not a parameter in OpenAIEmbeddingFunction
        )
    elif model_name == 'all-MiniLM-L6-v2':
        # ChromaDB's default embedding function
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")


# --- Data Preparation and Ingestion ---
def load_and_prepare_data(max_points: int) -> tuple[List[str], List[str], List[dict]]:
    """
    Loads the hotpot_qa dataset, prepares documents, ids, and metadata.
    Data concatenation:
    - collection.documents: 'title[0] : sentences[0], title[1] : sentences[1], ...'
    - collection.ids: 'id' from each document entry
    - collection.metadatas: optional dictionary with 'type' and 'level'
    """
    print("Loading hotpot_qa dataset...")
    dataset = load_dataset("hotpot_qa", CONFIG['data_instance'], split=CONFIG['data_split'])

    if max_points and max_points > 0:
        dataset = dataset.select(range(max_points))

    documents = []
    ids = []
    metadatas = []

    docs_processed = 0  # RRM Code Change: Track number of documents processed
    max_docs_to_process = CONFIG['max_hotpot_qa_docs_to_process'] # RRM Code Change: Limit number of documents to process
    for item in tqdm(dataset, desc="Preparing documents"):
        if docs_processed >= max_docs_to_process: # RRM Code Change: Stop processing if limit reached
            break
        docs_processed += 1
        id_str = item['id']
        context_data = item['context']

        # Concatenate title and sentences as a single string for the document
        document_parts = [f"{title} : {sentences}" for title, sentences in
                          zip(context_data['title'], context_data['sentences'])]
        document = " ".join(document_parts)

        # Store the id
        ids.append(id_str)

        # Store the prepared document
        documents.append(document)

        # Optional: Store metadata
        # TODO: The 'type' and 'level' fields are not directly present in the `distractor` split of the hotpot_qa dataset.
        # Placeholder for metadata as per the prompt's optional requirement.
        metadatas.append({'source': 'hotpot_qa', 'split': CONFIG['data_split']})

    print(f"Loaded {len(documents)} documents.")
    return documents, ids, metadatas


# --- ChromaDB Vector Store Management ---
def setup_vector_store() -> chromadb.Collection:
    """
    Sets up a persistent ChromaDB client and collection.
    - Uses a user-selected embedding model.
    - Sets cosine similarity for the HNSW index.
    - Creates a new collection or gets an existing one.
    """
    client = chromadb.PersistentClient(path=CONFIG['persist_directory'])
    embedding_func = get_embedding_function(CONFIG['embedding_model'])

    # The HNSW space must be set at collection creation
    metadata = {"hnsw:space": "cosine"}

    # RRM Code Change: Delete collection if it exists so that we can create new ones with same name but different embeddings
    # RRM Code Change: Below try-except block is for demo purposes to ensure a fresh start
    try:    # RRM Code change - Deleting existing collection for demo purposes
        # If collection exists, delete it for a fresh start (for demo purposes)
        client.delete_collection(name=CONFIG['collection_name'])    # RRM Code change - Deleting existing collection for demo purposes
        print(f"Deleted existing collection: {CONFIG['collection_name']}")    # RRM Code change - Print statement for deleted collection
    except Exception:
        pass  # Collection doesn't exist

    try:
        collection = client.get_or_create_collection(
            name=CONFIG['collection_name'],
            embedding_function=embedding_func,
            metadata=metadata
        )
    except Exception as e:
        print(f"Error creating/getting collection: {e}")
        # TODO: Human input required here to handle collection creation errors
        raise

    return collection


# --- LLM Integration and Response Generation ---
def get_llm_client(model_name: str):
    """
    Factory function to get the appropriate LLM client.
    This supports future extensibility via config updates.
    """
    if model_name == 'gpt-4o-mini':
        api_key = get_api_key('OpenAI', 'OPENAI_API_KEY')
        if not api_key:
            # TODO: Human input required here if key is not available
            raise ValueError("OpenAI API key required for gpt-4o-mini.")
        return OpenAI(api_key=api_key)
    elif model_name == 'gemini-1.5-flash':
        api_key = get_api_key('Google', 'GOOGLE_API_KEY')
        if not api_key:
            # TODO: Human input required here if key is not available
            raise ValueError("Google API key required for gemini-1.5-flash.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    elif model_name == 'claude-sonnet-4-20250514':
        api_key = get_api_key('Claude', 'ANTHROPIC_API_KEY')
        if not api_key:
            # TODO: Human input required here if key is not available
            raise ValueError("Claude API key required for claude-sonnet-4-20250514.")
        return Anthropic(api_key=api_key)
    elif model_name == 'llama3.2':
        # Ollama client for local llama3.2 model
        return OllamaClient(host=CONFIG['llama_api_base'])
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")


def generate_response(llm_client, model_name: str, query: str, context: List[str]) -> str:
    """Generates a response from the LLM using the retrieved context."""
    full_context = "\n\n".join(context)
    prompt = f"Based on the following context, answer the user's query:\n\nContext:\n{full_context}\n\nQuery: {query}\n\nAnswer:"
    print(f"DELETE THIS PRINT: Sending prompt to {model_name}...: {prompt}")

    try:
        if model_name in ['gpt-4o-mini', 'llama3.2']:
            messages = [{"role": "user", "content": prompt}]
            if model_name == 'gpt-4o-mini':
                response = llm_client.chat.completions.create(model=model_name, messages=messages)
                return response.choices[0].message.content
            elif model_name == 'llama3.2':
                # Use a different client method for Ollama
                response = llm_client.chat(model='llama3.2', messages=messages)
                return response['message']['content']
        elif model_name == 'gemini-1.5-flash':
            response = llm_client.generate_content(prompt)
            return response.text
        elif model_name == 'claude-sonnet-4-20250514':
            response = llm_client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            return "Unsupported LLM model for response generation."
    except Exception as e:
        return f"An error occurred during LLM generation: {e}"


# --- Main RAG Pipeline Execution ---
def main():
    """Main function to run the RAG pipeline."""
    # 1. User prompts for model selection
    print("Welcome to the RAG Pipeline!")

    # Select Embedding Model
    supported_embeddings = ['gemini-embedding-001', 'text-embedding-3-small', 'all-MiniLM-L6-v2']
    print("\nSelect an embedding model:")
    for i, model in enumerate(supported_embeddings):
        print(f"  {i + 1}. {model}")

    embedding_choice = int(input("Enter the number of your choice: ")) - 1
    CONFIG['embedding_model'] = supported_embeddings[embedding_choice]
    print(f"Selected embedding model: {CONFIG['embedding_model']}")

    # Select LLM
    supported_llms = ['gpt-4o-mini', 'gemini-1.5-flash', 'llama3.2', 'claude-sonnet-4-20250514']
    print("\nSelect an LLM:")
    for i, model in enumerate(supported_llms):
        print(f"  {i + 1}. {model}")

    llm_choice = int(input("Enter the number of your choice: ")) - 1
    CONFIG['llm_model'] = supported_llms[llm_choice]
    print(f"Selected LLM: {CONFIG['llm_model']}")

    # 2. Data Ingestion
    print("\n--- Starting Data Ingestion ---")
    documents, ids, metadatas = load_and_prepare_data(CONFIG['max_data_points'])

    # 3. Vector Store Setup and Indexing
    print("\n--- Setting up Vector Store ---")
    collection = setup_vector_store()

    # Check if the collection is empty and needs to be populated
    if collection.count() == 0:
        print("Collection is empty. Adding documents...")
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Successfully added {len(documents)} documents to the collection.")
    else:
        print(f"Collection already contains {collection.count()} documents. Skipping data ingestion.")

    # 4. Main RAG Loop
    print("\n--- RAG Pipeline Ready ---")
    llm_client = get_llm_client(CONFIG['llm_model'])

    while True:
        user_query = input("\nEnter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break

        print("\nRetrieving relevant context...")
        start_time = time.time()

        # ChromaDB handles the embedding of the query internally using the collection's embedding function
        retrieved_results = collection.query(
            query_texts=[user_query],
            n_results=CONFIG['n_results']
        )
        retrieval_time = time.time() - start_time

        retrieved_docs = retrieved_results['documents'][0]

        print(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds.")
        print("\n--- Generated Response ---")

        # 5. LLM Generation
        response = generate_response(llm_client, CONFIG['llm_model'], user_query, retrieved_docs)
        print(response)

        print("\n--- End of Response ---")


if __name__ == "__main__":
    main()