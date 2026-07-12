"""
main.py — AI Librarian API.
Endpoints:
  GET    /api/health
  POST   /api/papers            upload + ingest a new PDF
  GET    /api/papers            list papers in the stack
  DELETE /api/papers/{id}       remove a paper (registry only — see note below)
  POST   /api/chat              cross-paper synthesis, streamed
  GET    /api/notebook          current notebook entries
  GET    /api/notebook/export   notebook as markdown
"""

import os
import json
import time
from typing import List, Optional

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

import db
import ingestion
import rag
import notebook

load_dotenv()
db.init_db()

app = FastAPI(title="AI Librarian API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ai-librarian")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

groq_client = Groq(api_key=GROQ_API_KEY)
vectorstore = ingestion.get_vectorstore(PINECONE_API_KEY, PINECONE_INDEX)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    active_paper_ids: Optional[List[str]] = None  # None = search all papers


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/papers")
async def upload_paper(
    file: UploadFile = File(...),
    title: str = Form(...),
    authors: str = Form(""),
    year: str = Form(""),
):
    file_bytes = await file.read()
    result = ingestion.ingest_pdf(
        file_bytes, file.filename, title, authors, year,
        PINECONE_API_KEY, PINECONE_INDEX,
    )
    return result


@app.get("/api/papers")
async def get_papers():
    return db.list_papers()


@app.delete("/api/papers/{paper_id}")
async def remove_paper(paper_id: str):
    # NOTE: this removes the paper from the registry (so it disappears from
    # "The Stacks" list and stops being searchable via new uploads), but does
    # NOT delete its vectors from Pinecone. Deleting specific vectors by
    # metadata filter is a slightly bigger step — flag this to me once you're
    # ready to wire it up and I'll add a proper Pinecone delete-by-filter call.
    db.delete_paper(paper_id)
    return {"deleted": paper_id}


@app.get("/api/notebook")
async def get_notebook():
    return db.list_notebook_entries()


@app.get("/api/notebook/export")
async def export_notebook():
    md = notebook.export_markdown()
    return PlainTextResponse(md, media_type="text/markdown")


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    history = [m.dict() for m in req.history]

    standalone_q = rag.contextualize(groq_client, req.question, history)
    docs = rag.retrieve_attributed(vectorstore, standalone_q, req.active_paper_ids)
    context = rag.build_context_block(docs)

    qa_messages = (
        [{"role": "system", "content": rag.SYNTHESIS_SYSTEM_PROMPT.format(context=context)}]
        + history
        + [{"role": "user", "content": req.question}]
    )

    source_paper_ids = list({d.metadata.get("paper_id") for d in docs if d.metadata.get("paper_id")})

    def generate():
        yield sse({"type": "retrieved", "count": len(docs), "paper_ids": source_paper_ids})

        t0 = time.time()
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=qa_messages,
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                yield sse({"type": "token", "value": token})

        latency = time.time() - t0
        sources = [{
            "paper_id": d.metadata.get("paper_id"),
            "title": d.metadata.get("title"),
            "color": d.metadata.get("color"),
            "page": d.metadata.get("page"),
            "text": d.page_content,
        } for d in docs]

        yield sse({"type": "done", "sources": sources, "latency": latency})

        # Best-effort notebook logging — never let this break the chat response.
        try:
            notebook.maybe_log(groq_client, req.question, full_answer, source_paper_ids)
        except Exception:
            pass

    return StreamingResponse(generate(), media_type="text/event-stream")
