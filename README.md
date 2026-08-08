# AI Librarian

A research assistant for cross-referencing your own paper library — upload PDFs, ask questions
that span multiple papers at once, get answers with page-accurate citations.

**Live**: https://librarian-app-2026.netlify.app

## What is this?

AI Librarian is a full-stack RAG (retrieval-augmented generation) application built for
synthesis across a personal document library, not single-document Q&A.

```
Upload PDF → Auto-extract metadata → Chunk & embed → Ask across papers → Cited, streamed answer
```

Most "chat with your PDF" tools answer one document at a time or silently blend everything into
one anonymous context. This is built to keep every claim attributed to a specific paper and
page, and to only cite papers that were actually relevant to the question — not every paper in
the library by default.

## Architecture

```
┌──────────────────┐     HTTP      ┌───────────────────────────────┐
│   Static HTML/JS │ ────────────► │       FastAPI Backend         │
│   (Netlify)      │               │       (Render)                │
│                  │               │                               │
│  Chat UI         │               │  ┌─────────────────────────┐  │
│  Paper management│               │  │   Retrieval + Synthesis │  │
│  Notes drawer    │               │  │   (Groq / Llama 3.3)    │  │
│  PDF export      │               │  └────────────┬────────────┘  │
└──────────────────┘               │               │               │
                                   │  ┌────────────▼────────────┐  │
                                   │  │  Pinecone (hosted       │  │
                                   │  │  embeddings + vectors)  │  │
                                   │  └─────────────────────────┘  │
                                   │                               │
                                   │  Postgres (Neon)              │
                                   │  — papers, chats, notes       │
                                   └───────────────────────────────┘
```

## Features

**Ingestion**
- Auto-extracts title/authors/year from a PDF's first page (Llama 3.1 8B)
- Chunked and embedded via Pinecone's hosted `multilingual-e5-large` model
- Auto-generated 3-point digest per paper, cached once at upload (not regenerated per chat)

**Retrieval & synthesis**
- Cross-paper questions answered by Llama 3.3 70B, streamed token-by-token
- Numbered citations tied to exact source paper + page number
- Retrieval scoped to what's semantically relevant — a 2-paper comparison cites 2 papers, not
  every paper in the active set
- Small-talk and identity questions ("hi", "who are you") skip retrieval entirely — no forced
  citations on a greeting

**Chat**
- Real multi-session history, renameable, lazily created (no clutter from unused drafts)
- Retry regenerates in place with `‹1/n›` navigation between attempts
- Thumbs up/down persist across reloads; downvoted answers are reviewable, not just logged

**Export**
- Any single answer, or the full notes, downloadable as a formatted PDF

## Engineering highlights

| Challenge | Solution |
|---|---|
| Forced per-paper retrieval to avoid starving out papers in "list everything" questions | Over-corrected into citing irrelevant papers on narrow questions — reverted to scoped top-k search; full-library questions answered from a separate cached paper index instead |
| Local embedding model (`sentence-transformers`) OOM-killed uploads on a 512MB free-tier server | Switched to Pinecone's hosted embedding inference — no local model in memory, regardless of PDF size |
| SQLite on Render's free tier resets on every restart/sleep cycle | Migrated to hosted Postgres (Neon) — chat history and paper registry now survive redeploys |
| Stale client session IDs after a database migration caused hard 500s | Session creation is now idempotent server-side — an unrecognized session_id self-heals instead of crashing |
| Auto-classifying every chat exchange into "findings" produced contradictory notes over time | Replaced with user-curated saves — nothing enters Notes unless explicitly kept |

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Single-file HTML/CSS/JS, no framework, no build step |
| Backend | FastAPI (Python) |
| LLM | Llama 3.3 70B (synthesis), Llama 3.1 8B (extraction/digests) — via Groq |
| Embeddings | Pinecone-hosted `multilingual-e5-large`, 1024-dim |
| Vector store | Pinecone (serverless) |
| Database | Postgres (Neon) |
| Hosting | Render (backend), Netlify (frontend) |

## Known limitations

- No authentication — single shared library, not multi-tenant (see repo discussion for scope if
  that changes)
- Free-tier hosting sleeps after 15 min idle; first request after that takes 30-50s
- Editing a paper's metadata updates the registry but not already-embedded chunk metadata in
  Pinecone

---

Built by Reeva Kanakhara
