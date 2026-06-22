import chromadb

# Create local vector database
client = chromadb.Client()

# Create a collection to store documents
collection = client.create_collection(name="my_first_collection")

# Add documents
collection.add(
    documents=[
        "Python is a popular programming language for AI.",
        "LangChain helps connect LLMs with tools and memory.",
        "ChromaDB stores text as vector embeddings for search.",
        "Groq provides free and fast LLM API access.",
        "RAG stands for Retrieval Augmented Generation.",
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print(f"Total documents stored: {collection.count()}")

# Search by meaning
results = collection.query(
    query_texts=["what is a framework for building AI apps?"],
    n_results=2
)

print("\nQuery: 'what is a framework for building AI apps?'")
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {doc}")

# Second search
results2 = collection.query(
    query_texts=["free API for language models"],
    n_results=2
)

print("\nQuery: 'free API for language models'")
for i, doc in enumerate(results2['documents'][0]):
    print(f"Result {i+1}: {doc}")