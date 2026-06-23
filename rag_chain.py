from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

# Setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="knowledge_base")

# Knowledge base chunks with source info
documents = [
    {"text": "Python is a popular programming language used in AI and data science.", "source": "python_basics.txt"},
    {"text": "LangChain is a framework for building applications with large language models.", "source": "langchain_docs.txt"},
    {"text": "ChromaDB is a vector database that stores embeddings for semantic search.", "source": "chromadb_docs.txt"},
    {"text": "RAG stands for Retrieval Augmented Generation used in AI chatbots.", "source": "rag_guide.txt"},
    {"text": "Groq provides free and fast API access to open source LLM models.", "source": "groq_docs.txt"},
    {"text": "Embeddings convert text into numbers that represent meaning.", "source": "embeddings_guide.txt"},
    {"text": "Machine learning models learn patterns from large amounts of data.", "source": "ml_basics.txt"},
    {"text": "Vector search finds similar documents based on meaning not keywords.", "source": "vector_search.txt"},
    {"text": "LangGraph is a framework for building stateful multi-agent applications.", "source": "langgraph_docs.txt"},
    {"text": "CrewAI allows building multi-agent systems where agents collaborate on tasks.", "source": "crewai_docs.txt"},
    {"text": "wheqather in hydrabad is 35 degree clear .","source": "https://www.accuweather.com/en/in/hyderabad/202190/hourly-weather-forecast/202190"},
]

# Store in ChromaDB
texts = [d["text"] for d in documents]
sources = [d["source"] for d in documents]
embeddings = embed_model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"source": s} for s in sources],
    ids=[f"doc_{i}" for i in range(len(texts))]
)

print(f"Knowledge base ready: {collection.count()} chunks stored\n")
print("=" * 55)


# RAG function — full chain
def ask(question):
    # Step 1 — embed the question
    question_embedding = embed_model.encode([question]).tolist()

    # Step 2 — retrieve top 3 relevant chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results['documents'][0]
    retrieved_sources = [m['source'] for m in results['metadatas'][0]]

    # Step 3 — build prompt with context
    context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

    prompt = f"""Answer the question using ONLY the information provided below.
If the answer is not in the information, say "I don't have enough information."

Information:
{context}

Question: {question}

Answer:"""

    # Step 4 — send to Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Step 5 — print with source citations
    print(f"Question : {question}")
    print(f"Answer   : {answer}")
    print(f"Sources  : {', '.join(retrieved_sources)}")
    print("=" * 55)


# Test with 5 different questions
ask("What is RAG and how is it used?")
ask("Which tools can I use to build AI agents?")
ask("How does vector search find relevant documents?")
ask("What is the best free LLM API available?")
ask("What is the weather in Hyderabad today?")