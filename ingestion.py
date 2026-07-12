"""
ingestion.py — replaces the old cloud_upload.py.
Instead of one hardcoded PDF, this ingests ANY uploaded PDF, tags every chunk
with which paper it came from (paper_id, title, authors, year, color, page),
and upserts it into the shared Pinecone index.
"""

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

import db

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vectorstore(pinecone_api_key: str, index_name: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
    )


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    title: str,
    authors: str,
    year: str,
    pinecone_api_key: str,
    index_name: str,
) -> dict:
    """Registers the paper in SQLite, chunks + embeds the PDF, tags every
    chunk with paper metadata, and upserts to Pinecone."""
    paper_id, color = db.add_paper(title, authors, year, filename)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()  # each page already has metadata["page"]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata.update({
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "color": color,
            })

        vectorstore = get_vectorstore(pinecone_api_key, index_name)
        vectorstore.add_documents(chunks)

    finally:
        os.unlink(tmp_path)

    return {"paper_id": paper_id, "title": title, "color": color, "chunks": len(chunks)}
