"""
Can you please write Python code that satisfies the following criteria:
1. Generate code that uses RAG.
2. The code should use ChromaDB for storing and retrieving vectors.
3. ChromaDB should be persisted to disk. The embedding model should be configurable to use OpenAI, Google's Gemini, or HuggingFace's open-source models.
4. ChromaDB should use "cosine similarity" as the distance metric for vector similarity search.
5. The code should provide options to chose an embedding model that is from OpenAI or Google's Gemini family or some of the opensource encoding models from HuggingFace.
6. Use the hotspot_qa dataset as the source of data. Use the `distractor` split of the dataset. Use the "training" split from the "distractor" split.
7. The code should be able to retrieve the relevant context from the dataset based on a query.
8. The user should be able to select the LLM Model to use for generating responses based on the retrieved context. For now limit the LLM Model selection to gpt-4o-mini, gemini-1.5-flash, and llama3.2, claude-sonnet-4-20250514.
9. The code should be able to generate a response based on the retrieved context and the user's query.
10. The code should be as config driven as possible.
11. The code should be well-commented, easy to understand, modular, and maintainable.
"""