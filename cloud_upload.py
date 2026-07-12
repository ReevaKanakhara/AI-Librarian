import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1 READ
file_path = "2390-Article Text-9173-1-10-20251003.pdf"
print(f"Reading {file_path}...")
loader = PyPDFLoader(file_path)
data = loader.load()
print(f"   Loaded {len(data)} pages.")

# 2 CHUNK
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)
print(f"Spliting into {len(chunks)} chunks.")

# 3 FREE LOCAL EMBEDDINGS
print("Loading embedding model (first time downloads ~80MB, be patient)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")

# 4 UPLOAD TO PINECONE
index_name = "ai-librarian"
print(f"Uploading {len(chunks)} chunks to Pinecone...")
vector_store = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name=index_name
)

print("SUCCESS! Your research is now in the cloud database.")