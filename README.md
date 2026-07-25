# AI Librarian

A personal research assistant for your own paper library. Upload PDFs, ask questions across
them, and get answers with page-accurate citations — synthesis and comparison across multiple
papers at once, not just single-document Q&A.

## Features

- **Upload & auto-extract** — drop in a PDF, title/authors/year auto-fill from the first page
- **Cross-paper synthesis** — ask questions spanning multiple papers, with citations scoped to
  what's actually relevant (not every paper you've uploaded)
- **Chat history** — real multi-session history, renameable, with retry-and-compare response
  variants
- **Notes** — an auto-generated 3-point digest per paper (generated once at upload), plus
  manually saved answers and flagged (thumbs-down) responses for review
- **PDF export** — download any answer or your full notes as a formatted PDF
- **Paper management** — toggle which papers are active for search, edit metadata inline, delete
  with a real Pinecone vector purge (not just hidden from the list)

## Stack

- **Backend**: FastAPI, Python
- **Vector store**: Pinecone
- **LLM**: Llama 3.3 70B for synthesis, Llama 3.1 8B for metadata extraction and digests
  (both via Groq)
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace, runs locally)
- **Database**: SQLite (paper registry, chat history, notes)
- **Frontend**: single-file HTML/CSS/JS (`frontend/index.html`), no build step

## Local setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
PINECONE_INDEX=ai-librarian
```

Run the backend:
```powershell
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in a browser. It talks to `http://localhost:8000` by default.

## Deployment

See `DEPLOY.md` for the full walkthrough — backend on Render, frontend on Netlify.

## Project structure

```
main.py           FastAPI app — all API endpoints
db.py             SQLite storage (papers, chat sessions/messages, notes, feedback)
ingestion.py      PDF parsing, chunking, embedding, Pinecone upsert/delete
rag.py            Retrieval + prompt construction for synthesis
notebook.py       Notes export (markdown)
pdf_export.py     Markdown → PDF conversion
frontend/index.html   The entire UI
```
