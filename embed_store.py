from sentence_transformers import SentenceTransformer
import chromadb

# Load free local embedding model (downloads once ~80MB)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample chunks (simulating what PDF loader gives you)
chunks = [
    "Python is a popular programming language used in AI and data science.",
    "LangChain is a framework for building applications with large language models.",
    "ChromaDB is a vector database that stores embeddings for semantic search.",
    "RAG stands for Retrieval Augmented Generation used in AI chatbots.",
    "Groq provides free and fast API access to open source LLM models.",

]

# Convert chunks to embeddings
embeddings = embed_model.encode(chunks).tolist()
print(f"Embeddings created: {len(embeddings)}")
print(f"Each embedding size: {len(embeddings[0])} numbers")

# Store in ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="my_chunks")

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"\nTotal chunks stored: {collection.count()}")

# Run test query — retrieve top 3 chunks
query = "how does semantic search work?"
query_embedding = embed_model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print(f"\nQuery: '{query}'")
print("\nTop 3 most relevant chunks:")
for i, doc in enumerate(results['documents'][0]):
    print(f"\nRank {i+1}: {doc}")