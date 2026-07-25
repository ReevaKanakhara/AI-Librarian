"""
ingestion.py — replaces the old cloud_upload.py.
Instead of one hardcoded PDF, this ingests ANY uploaded PDF, tags every chunk
with which paper it came from (paper_id, title, authors, year, color, page),
and upserts it into the shared Pinecone index.
"""

import os
import json
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


def delete_paper_vectors(paper_id: str, pinecone_api_key: str, index_name: str) -> int:
    """Actually purges every vector belonging to this paper from Pinecone.

    Serverless Pinecone indexes (the free/default tier) do NOT support
    delete-by-metadata-filter directly — that only works on pod-based
    indexes. So instead: query for every vector matching this paper_id
    (using a dummy zero vector + the metadata filter, which pods and
    serverless both support), collect their ids, then delete by id.
    Returns the number of vectors deleted, so the caller/UI can confirm
    it wasn't a silent no-op."""
    from pinecone import Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    dim = 384  # all-MiniLM-L6-v2 output size
    dummy_vector = [0.0] * dim
    ids_to_delete = set()

    # Paginate: Pinecone query top_k caps at 10000 per call, which is far
    # more than any single paper will realistically produce in chunks.
    result = index.query(
        vector=dummy_vector,
        top_k=10000,
        filter={"paper_id": {"$eq": paper_id}},
        include_values=False,
        include_metadata=False,
    )
    for match in result.get("matches", []):
        ids_to_delete.add(match["id"])

    if ids_to_delete:
        index.delete(ids=list(ids_to_delete))

    return len(ids_to_delete)


def extract_metadata_from_pdf(file_bytes: bytes, groq_client) -> dict:
    """Reads the first page of a PDF and asks the LLM to guess title/authors/year.
    Used to auto-fill the upload form so the user doesn't have to type it in."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        first_page_text = pages[0].page_content[:3000] if pages else ""
    finally:
        os.unlink(tmp_path)

    prompt = f"""Extract the title, authors, and publication year from this excerpt of an
academic paper's first page. Respond ONLY with valid JSON, nothing else:
{{"title": "...", "authors": "...", "year": "..."}}
If a field can't be determined, use an empty string for it.

Excerpt:
{first_page_text}
"""
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        return {
            "title": parsed.get("title", ""),
            "authors": parsed.get("authors", ""),
            "year": parsed.get("year", ""),
        }
    except Exception:
        return {"title": "", "authors": "", "year": ""}


def summarize_text(text: str, groq_client) -> str:
    """One-time, deterministic 3-bullet digest — this is what populates the
    Notes drawer's 'Digest' section automatically, rather than leaving it
    blank until the user manually saves something."""
    prompt = f"""Summarize this academic paper excerpt in exactly 3 short bullet points covering
its main contribution, method, and key result. Respond with ONLY the 3 bullets, each starting
with "- ", nothing else.

Paper text:
{text[:6000]}
"""
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()


def get_paper_chunks_text(paper_id: str, pinecone_api_key: str, index_name: str, max_chunks: int = 12) -> str:
    """Pulls back a paper's own stored chunks from Pinecone (ordered by page)
    so an existing paper's digest can be backfilled without needing the
    original PDF file again — we never keep the raw upload after ingest,
    only its embedded chunks."""
    vectorstore = get_vectorstore(pinecone_api_key, index_name)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": max_chunks, "filter": {"paper_id": paper_id}}
    )
    docs = retriever.invoke("summary abstract introduction conclusion results")
    docs.sort(key=lambda d: d.metadata.get("page", 0))
    return "\n".join(d.page_content for d in docs)


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    title: str,
    authors: str,
    year: str,
    pinecone_api_key: str,
    index_name: str,
    groq_client=None,
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

        if groq_client is not None:
            try:
                full_text = "\n".join(p.page_content for p in pages)
                summary = summarize_text(full_text, groq_client)
                db.set_paper_summary(paper_id, summary)
            except Exception:
                pass  # digest is a nice-to-have; never fail the upload over it

    finally:
        os.unlink(tmp_path)

    return {"paper_id": paper_id, "title": title, "color": color, "chunks": len(chunks)}