from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")
print(f"First page preview: {pages[0].page_content[:200]}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # each chunk = max 500 characters
    chunk_overlap=50   # 50 characters overlap between chunks (keeps context)
)

chunks = splitter.split_documents(pages)

print(f"\nTotal chunks created: {len(chunks)}")

# Inspect first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Length : {len(chunk.page_content)} characters")
    print(f"Content: {chunk.page_content}")