import os
import time
from dotenv import load_dotenv
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. FREE LOCAL EMBEDDINGS (no API key needed) ---
print("Loading embedding model (first time takes ~1 min to download)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 2. CONNECT TO PINECONE ---
print("Connecting to your Dwarka Research Cloud...")
vector_store = PineconeVectorStore(index_name="ai-librarian", embedding=embeddings)

# --- 3. GROQ LLM ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- 4. QA CHAIN ---
prompt = PromptTemplate.from_template("""
You are a helpful research assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}
Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. CHAT LOOP ---
print("\nLIBRARIAN READY! Ask about your Dwarka paper.")
print("(Type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    print("Searching your research paper...")
    try:
        response = qa_chain.invoke(query)
        print(f"\nAI: {response}\n" + "-" * 50)
    except Exception as e:
        print(f"Error: {e}")