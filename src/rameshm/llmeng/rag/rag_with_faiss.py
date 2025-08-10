import os
from openai import OpenAI
#from google import generativeai as genai
from google import genai
from google.genai import types
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings as ChromaSettings
from datasets import load_dataset, Dataset
from rameshm.llmeng.utils import init_utils
import rag_constants
from typing import List, Tuple

my_logger = init_utils.set_environment_logger()

def load_hotspot_data(primary_split: str = "distractor", secondary_split: str = "train",
                      dataset_size: int = 0) -> Dataset:
    # Load Hugging Face's hotpot_qa dataset
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Setting this environment variable disables the warning about symlinks in the Hugging Face Hub
    dataset = load_dataset("hotpot_qa", primary_split, trust_remote_code=True) # Load distractor split. Has two datasets: fullwiki and distractor
    if dataset_size:
        return dataset[secondary_split].select(range(dataset_size))  # Use a smaller subset for testing; Has two splits: 'train' and 'validation'
    else:
        return dataset[secondary_split] # return the full dataset


def get_client_openai() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_client_gemini() -> genai.Client:
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

#embedding_model = "text-embedding-3-small"  # Example embedding model, adjust as needed

def get_embeddings_openai(embed_texts: List[str]) -> List[List[float]]:
    client = get_client_openai()
    embedding_model = rag_constants.RAG_EMBEDDING_MODEL_OPENAI
    response = client.embeddings.create(
        model=embedding_model,
        input=embed_texts
    )
    # Extract and return all embeddings in the same order
    return [item.embedding for item in response.data]


def get_embeddings_gemini(embed_texts: List[str]) -> List[List[float]]:
    client = get_client_gemini()
    response = client.models.embed_content(
        model=rag_constants.RAG_EMBEDDING_MODEL_GEMINI,
        config =types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=1053
        ),
        contents=embed_texts,
    )
    print(f"Metadata: {response.metadata} Length of embeddings: {len(response.embeddings)}")
    #gemini_embeddings = []
    return [embedding_lcl.values for embedding_lcl in response.embeddings]
    # for embedding_lcl in result.embeddings:
    #     print(f"Embedding values: {embedding_lcl.values} Length of embedding values: {len(embedding_lcl.values)}")
    #     # Append the embedding values to the list
    #     gemini_embeddings.append(embedding_lcl.values)
    # return gemini_embeddings


def create_faiss_vector_store(embeddings: List[List[float]]) -> faiss.Index:
    """
    Create a FAISS index from the embeddings and text lookup.
    """
    embedding_dim = len(embeddings[0])
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(vectors)
    return index


def search_faiss(user_query: str, index: faiss.Index, top_k=3, embedding_model_from: str = "openai", ):
    if embedding_model_from == "openai":
        query_embedding = get_embeddings_openai([user_query])[0]  # Get the embedding for the query
    elif embedding_model_from == "gemini":
        query_embedding = get_embeddings_gemini([user_query])[0]
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    #print(f"Length of indices: {len(indices[0])}")
    #print(f"Length of distances: {len(distances[0])}")
    return distances, indices
    #return [text_lookup[i] for i in indices[0]]


def chat_with_faiss_context_gemini(user_query: str, context_texts: List[str] = None):
    """
    Takes a user query, and RAG context to send to LLM
    """
    # Construct the prompt for the Gemini model
    # Combine the retrieved context with the user's query
    context_str = "\n\n".join(context_texts)
    system_prompt = f"You are a helpful assistant. Give brief and accurate answers. Use the following context to answer the user's question. If the information is not in the context, say that you don't have enough information to answer. Do not make anything up if you haven't been provided with relevant context."
    prompt = f"""
        Context:
        {context_str}

        User's question: {user_query}
        """
    print(f"Sending prompt to Gemini...: {prompt}")

    # Call Gemini API and Get Response ---
    client = get_client_gemini()
    chat_completion = client.models.generate_content(
        model=rag_constants.RAG_LLM_MODEL_GEMINI,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=1000
        ),
        contents=prompt
    )

    # Return the final answer
    model_answer = chat_completion.text
    print("\n--- Gemini Model's Answer ---")
    print(model_answer)
    print("--------------------")
    return model_answer


def chat_with_faiss_context_openai(user_query: str, context_texts: List[str] = None):
    """
    Takes a user query, finds relevant context from a FAISS index, and
    sends both to an OpenAI model to generate a response.
    """
    # Construct the prompt for the OpenAI model
    # We combine the retrieved context with the user's query
    context_str = "\n\n".join(context_texts)
    #system_prompt = "You are a helpful assistant. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."
    system_prompt = f"You are a helpful assistant. Give brief and accurate answers. Use the following context to answer the user's question. If the information is not in the context, say that you don't have enough information to answer. Do not make anything up if you haven't been provided with relevant context."
    prompt = f"""
    Context:
    {context_str}

    User's question: {user_query}
    """
    print(f"Sending prompt to OpenAI... {prompt}")
    client = get_client_openai()
    # Call OpenAI API and Get Response ---
    chat_completion = client.chat.completions.create(
        model=rag_constants.RAG_LLM_MODEL_OPENAI,  # Or gpt-4, or other chat model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )

    # 4. Return the final answer
    model_answer = chat_completion.choices[0].message.content
    print("\n--- OpenAI Model's Answer ---")
    print(model_answer)
    print("--------------------")
    return model_answer

def get_test_embeddings(embedding_model_from: str = "openai") -> Tuple[List[float], List[str]]:
    test_strings = ["Capital city of France is Paris.", "Capital city of Germany is Berlin.",
                    "capital city of Italy is Rome.", "Capital city of Spain is Madrid.",
                    "Capital city of Japan is Tokyo.", "Capital city of India is New Delhi.",
                    "Capital city of USA is Washington DC.", "Capital city of UK is London.",
                    "There is no capital city in Antarctica.", # Distractors
                    "City's themselves are not capitals, but they are important.",
                    "NY City is not the capital of USA, but it is a major city."]

    # Create embeddings for the test strings
    text_lookup = [f"{i}: {text}" for i, text in enumerate(test_strings)]
    embeddings = []
    if embedding_model_from == "openai":
        embeddings.extend(get_embeddings_openai(text_lookup))
    elif embedding_model_from == "gemini":
        embeddings.extend(get_embeddings_gemini(text_lookup))
    return embeddings, text_lookup
# for text in text_lookup:
#     print(f"Creating embedding for: {text}")
#     embeddings.append(get_embeddings_openai([text])[0])  # Get the embedding for each text
# faiss_index = create_faiss_vector_store(embeddings, text_lookup)

if __name__ == "__main__":
    user_query = "France capital?"
    for embedding_model_from in ["openai", "gemini"]:
        print(f"\n\nUsing embedding model from: {embedding_model_from}")
        embeddings, text_lookup = get_test_embeddings(embedding_model_from)  # Change to "gemini" for Gemini embeddings)
        print(f"Number of documents: {len(text_lookup)} ; Number of embeddings: {len(embeddings)}")
        faiss_index = create_faiss_vector_store(embeddings)
        # 2.1. Embed the user's query
        print(f"Processing query: '{user_query}'...")
        context_texts = []
        distances, indices = search_faiss(embedding_model_from=embedding_model_from,
                                          user_query=user_query, index=faiss_index, top_k=3)
        if len(indices):
            context_texts = [text_lookup[i] for i in indices[0]]  # Get the context texts based on indices
            # give a case statement to handle different embedding models
            if embedding_model_from == "openai":
                chat_with_faiss_context_openai(user_query, context_texts)
            elif embedding_model_from == "gemini":
                chat_with_faiss_context_gemini(user_query, context_texts)
