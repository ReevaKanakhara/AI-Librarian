"""
main.py — AI Librarian API.
Endpoints:
  GET    /api/health
  POST   /api/papers                upload + ingest a new PDF
  GET    /api/papers                list papers in the stack
  PATCH  /api/papers/{id}           correct title/authors/year without re-ingesting
  DELETE /api/papers/{id}           remove a paper — registry AND Pinecone vectors
  POST   /api/sessions              create a new chat session
  GET    /api/sessions              list chat sessions (history, most recent first)
  PATCH  /api/sessions/{id}         rename a chat session
  GET    /api/sessions/{id}/messages   full message log for a session
  DELETE /api/sessions/{id}         delete a chat session
  POST   /api/chat                  cross-paper synthesis, streamed, persisted to a session
  POST   /api/messages/{id}/feedback   thumbs up/down on an assistant message
  GET    /api/notebook              saved notes (user-curated, not auto-logged)
  POST   /api/notebook              save a Q&A exchange as a note
  GET    /api/notebook/export       notes as markdown
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
import pdf_export

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
    session_id: str


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/extract-metadata")
async def extract_metadata(file: UploadFile = File(...)):
    file_bytes = await file.read()
    return ingestion.extract_metadata_from_pdf(file_bytes, groq_client)


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
        PINECONE_API_KEY, PINECONE_INDEX, groq_client=groq_client,
    )
    return result


@app.get("/api/papers")
async def get_papers():
    return db.list_papers()


@app.delete("/api/papers/{paper_id}")
async def remove_paper(paper_id: str):
    # Real deletion: purge the paper's vectors from Pinecone first, then
    # drop it from the SQLite registry. Order matters — if the vector
    # delete fails we don't want the registry to have already forgotten
    # the paper existed (that would orphan vectors with no way to find
    # them again by paper_id from the UI).
    deleted_count = ingestion.delete_paper_vectors(paper_id, PINECONE_API_KEY, PINECONE_INDEX)
    db.delete_paper(paper_id)
    return {"deleted": paper_id, "vectors_removed": deleted_count}


class PaperUpdateRequest(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None


@app.patch("/api/papers/{paper_id}")
async def edit_paper(paper_id: str, req: PaperUpdateRequest):
    db.update_paper(paper_id, title=req.title, authors=req.authors, year=req.year)
    return {"updated": paper_id}


@app.post("/api/papers/{paper_id}/summarize")
async def summarize_existing_paper(paper_id: str):
    # For papers uploaded before the digest feature existed — pulls the
    # paper's own already-embedded chunks back from Pinecone (we don't keep
    # the raw PDF after ingest) and generates the same 3-bullet digest.
    text = ingestion.get_paper_chunks_text(paper_id, PINECONE_API_KEY, PINECONE_INDEX)
    if not text.strip():
        return {"error": "No indexed text found for this paper."}
    summary = ingestion.summarize_text(text, groq_client)
    db.set_paper_summary(paper_id, summary)
    return {"paper_id": paper_id, "summary": summary}


@app.post("/api/sessions")
async def new_session():
    session_id = db.create_session()
    return db.get_session(session_id)


@app.get("/api/sessions")
async def get_sessions():
    return db.list_sessions()


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return db.list_messages(session_id)


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    db.delete_session(session_id)
    return {"deleted": session_id}


class SessionRenameRequest(BaseModel):
    title: str


@app.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, req: SessionRenameRequest):
    db.rename_session(session_id, req.title.strip()[:80])
    return db.get_session(session_id)


class FeedbackRequest(BaseModel):
    rating: str  # "up" or "down"


@app.post("/api/messages/{message_id}/feedback")
async def feedback(message_id: str, req: FeedbackRequest):
    if req.rating not in ("up", "down", "none"):
        return {"error": "rating must be 'up', 'down', or 'none'"}
    if req.rating == "none":
        db.clear_message_feedback(message_id)
    else:
        db.set_message_feedback(message_id, req.rating)
    return {"message_id": message_id, "rating": req.rating}


@app.get("/api/notebook")
async def get_notebook():
    return db.list_notebook_entries()


@app.get("/api/feedback/flagged")
async def get_flagged():
    # Surfaces what you've thumbs-downed so the feedback isn't invisible —
    # there's no fine-tuning loop here (this app doesn't retrain anything),
    # but at minimum you should be able to see what you flagged and revisit
    # it, rather than it silently vanishing into a database.
    return db.list_down_voted()


class NoteRequest(BaseModel):
    question: str
    answer: str
    source_paper_ids: List[str] = []


@app.post("/api/notebook")
async def save_note(req: NoteRequest):
    # User-curated notes only now — no automatic LLM classification, which
    # is what was producing self-contradictory auto-logged entries. The
    # user decides what's worth keeping.
    summary = f"Q: {req.question}\nA: {req.answer}"
    entry_id = db.add_notebook_entry("saved", summary, req.source_paper_ids)
    return {"id": entry_id}


class PdfExportRequest(BaseModel):
    title: str = "AI Librarian Export"
    markdown: str


@app.post("/api/export/pdf")
async def export_pdf(req: PdfExportRequest):
    pdf_bytes = pdf_export.markdown_to_pdf_bytes(req.markdown, req.title)
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{req.title[:50]}.pdf"'},
    )


@app.get("/api/notebook/export")
async def export_notebook():
    md = notebook.export_markdown()
    return PlainTextResponse(md, media_type="text/markdown")


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    history = [m.dict() for m in req.history]

    is_substantive = rag.needs_retrieval(req.question)

    if is_substantive:
        standalone_q = rag.contextualize(groq_client, req.question, history)
        docs = rag.retrieve_attributed(vectorstore, standalone_q, req.active_paper_ids)
        context = rag.build_context_block(docs)
        library_index = rag.build_library_index(db.list_papers())
        system_prompt = rag.SYNTHESIS_SYSTEM_PROMPT.format(
            library_index=library_index, context=context)
    else:
        # Small talk / greetings: skip retrieval entirely so the answer
        # doesn't drag in unrelated citation chips (this is what was
        # happening with a plain "Hey").
        docs = []
        system_prompt = rag.CASUAL_SYSTEM_PROMPT

    qa_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": req.question}]
    )

    source_paper_ids = list({d.metadata.get("paper_id") for d in docs if d.metadata.get("paper_id")})

    # Auto-title a fresh session from its first question, same pattern as
    # Claude/Gemini — only rename if it's still using the default title.
    session = db.get_session(req.session_id)
    if session is None:
        # The frontend sent a session_id that doesn't exist server-side —
        # e.g. a browser tab left open across a database migration/redeploy,
        # still holding a session id from a backing store that no longer
        # has it. Rather than 500ing on the foreign key violation this
        # causes, just create it: self-healing instead of crashing.
        db.ensure_session(req.session_id)
    elif session["title"] == "New chat":
        title = req.question.strip()[:60]
        if len(req.question.strip()) > 60:
            title += "…"
        db.rename_session(req.session_id, title)

    db.add_message(req.session_id, "user", req.question)

    def generate():
        yield sse({"type": "retrieved", "count": len(docs), "paper_ids": source_paper_ids})

        t0 = time.time()
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=qa_messages,
            # temperature=0 for real paper questions (accuracy over
            # variety); casual/identity chat gets some warmth so a retry
            # doesn't come back byte-for-byte identical every time.
            temperature=0 if is_substantive else 0.8,
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
        all_sources = [{
            "paper_id": d.metadata.get("paper_id"),
            "title": d.metadata.get("title"),
            "color": d.metadata.get("color"),
            "page": d.metadata.get("page"),
            "text": d.page_content,
        } for d in docs] if is_substantive else []

        # The model tags which paper ids it actually referenced (see the
        # CITED_PAPER_IDS marker in SYNTHESIS_SYSTEM_PROMPT) — this is what
        # stops "compare X and Y" from citing every other active paper too,
        # since retrieval always pulls from every active paper by design
        # (that's what fixed the earlier "starved paper" bug), but not
        # every retrieved paper is necessarily relevant to a given question.
        clean_answer, cited_ids = rag.split_cited_ids(full_answer)
        if cited_ids is None:
            # Marker missing/unparseable — fail open rather than silently
            # hiding every citation.
            sources = all_sources
        else:
            sources = [s for s in all_sources if s["paper_id"] in cited_ids]

        # Persist the assistant's answer and grab its id so the frontend can
        # attach thumbs up/down feedback to this exact message.
        message_id = None
        try:
            message_id = db.add_message(req.session_id, "assistant", clean_answer, sources)
            db.touch_session(req.session_id)
        except Exception:
            pass

        yield sse({"type": "done", "answer": clean_answer, "sources": sources, "latency": latency, "message_id": message_id})

    return StreamingResponse(generate(), media_type="text/event-stream")