# AI Librarian

A personal research assistant for your own paper library. Upload PDFs, ask questions across
them, and get answers with page-accurate citations — synthesis and comparison across multiple
papers at once, not single-document Q&A.

**Live**: https://librarian-app-2026.netlify.app
**API**: https://ai-librarian-5iu3.onrender.com

---

## Features

- **Upload & auto-extract** — drop in a PDF, title/authors/year auto-fill from the first page
- **Cross-paper synthesis** — ask questions spanning multiple papers; citations are scoped to
  what's actually relevant to the question, not every paper in the library
- **Numbered, page-accurate citations** — every claim traces back to a specific paper and page
- **Chat history** — real multi-session history, renameable, lazily created (no clutter from
  chats you never used)
- **Retry with variants** — regenerate a response and flip between attempts (`‹1/3›`-style nav)
- **Real feedback** — thumbs up/down persist across reloads; flagged (👎) answers are reviewable,
  not just silently logged
- **Notes** — an auto-generated 3-point digest per paper (deterministic, generated once at
  upload) plus manually saved answers you choose to keep
- **PDF export** — download any single answer or your full notes as a formatted PDF
- **Paper management** — toggle which papers are active for search, edit metadata inline, delete
  with a real vector purge (not just hidden from the list)
- **Small-talk aware** — greetings and questions about the assistant itself don't trigger
  unnecessary retrieval or citations

## Tech stack

- **Backend**: FastAPI (Python)
- **LLM**: Llama 3.3 70B for synthesis, Llama 3.1 8B for metadata extraction and digests —
  both via [Groq](https://groq.com)
- **Embeddings**: Pinecone's hosted `multilingual-e5-large` model (1024-dim), called over the
  API — deliberately **not** a local model. An earlier version ran embeddings locally via
  `sentence-transformers`/`torch`, which needed 500MB+ RAM and OOM-killed uploads on a free-tier
  512MB server. Hosted inference removes that entirely: this process never loads a model into
  memory, regardless of PDF size.
- **Vector store**: Pinecone (serverless, cosine similarity, 1024 dimensions)
- **Database**: SQLite — paper registry, chat sessions/messages, notes, feedback
- **Frontend**: single-file HTML/CSS/JS (`frontend/index.html`), no build step, no framework

## Project structure

```
main.py           FastAPI app — all API endpoints, request/response handling
db.py             SQLite storage — papers, chat sessions/messages, notes, feedback
ingestion.py      PDF parsing, chunking, hosted-embedding calls, Pinecone upsert/delete
rag.py            Retrieval + prompt construction for synthesis, small-talk gating
notebook.py       Notes export (markdown)
pdf_export.py     Markdown → PDF conversion
frontend/
  index.html      The entire UI
requirements.txt
runtime.txt       Pins Python 3.11 (langchain/pinecone ecosystem compatibility)
render.yaml       Render deploy config
Procfile          Alternate deploy config (Railway/Heroku-style platforms)
```

## API endpoints

```
GET    /api/health
POST   /api/extract-metadata          auto-fill title/authors/year from a PDF's first page
POST   /api/papers                    upload + ingest a new PDF
GET    /api/papers                    list papers in the library
PATCH  /api/papers/{id}               correct title/authors/year without re-ingesting
POST   /api/papers/{id}/summarize     backfill a digest for a paper uploaded before that feature
DELETE /api/papers/{id}               remove a paper — registry AND Pinecone vectors
POST   /api/sessions                  create a chat session
GET    /api/sessions                  list chat sessions
PATCH  /api/sessions/{id}             rename a session
GET    /api/sessions/{id}/messages    full message log for a session
DELETE /api/sessions/{id}             delete a session
POST   /api/chat                      ask a question — streamed response
POST   /api/messages/{id}/feedback    thumbs up/down on an assistant message
GET    /api/notebook                  saved notes
POST   /api/notebook                  save a Q&A exchange as a note
GET    /api/notebook/export           notes as markdown
GET    /api/feedback/flagged          answers you've thumbs-downed
POST   /api/export/pdf                render markdown to a downloadable PDF
```

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

Your Pinecone index must be created with **dimension 1024** and **metric cosine** (to match the
hosted embedding model) — see app.pinecone.io → Indexes → Create Index.

Run the backend:
```powershell
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in a browser. It talks to `http://localhost:8000` by default (see
`API_BASE` near the top of the `<script>` block if you need to point it elsewhere).

## Deployment

**Backend → Render** (free tier works; see `render.yaml`)
1. Connect your GitHub repo at render.com
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set environment variables: `PINECONE_API_KEY`, `GROQ_API_KEY`, `PINECONE_INDEX`,
   `PYTHON_VERSION=3.11.9`

**Frontend → Netlify**
1. Edit `API_BASE` in `frontend/index.html` to point at your Render URL
2. Drag `index.html` onto app.netlify.com/drop, or connect the repo for auto-deploys
3. Lock down CORS in `main.py` (`allow_origins`) to your actual Netlify URL once you have it

## Known limitations

- **Ephemeral disk on Render's free tier** — `librarian.db` resets on redeploy/restart. Fine for
  a personal demo; a paid disk or migrating off SQLite would fix this for anything long-lived.
- **Free tier sleeps after 15 min idle** — first request after that takes 30-50s to wake up.
- **Retry variants aren't grouped after reload** — regenerating a response stores each attempt as
  a separate row in the chat log; the `‹1/n›` navigation only groups them within the live session.
- **Editing a paper's metadata** only updates the SQLite registry (used for "list all
  papers/authors" questions), not the per-chunk metadata already embedded in Pinecone — so a
  corrected author name won't retroactively appear in older passage citations from that paper.
