"""
db.py — lightweight SQLite storage for:
  - papers: the registry of uploaded documents (title, authors, color tag, etc.)
  - notebook_entries: auto-logged findings / open questions / disagreements

Pinecone still stores the actual embedded chunks. This file only stores the
metadata needed to render "The Stacks" list and "The Notebook" panel.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "librarian.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year TEXT,
            color TEXT NOT NULL,
            filename TEXT,
            uploaded_at TEXT NOT NULL
        )
    """)
    # Migration: older registries won't have this column yet. A one-time,
    # deterministic per-paper summary generated at ingest time — shown in
    # the Notes drawer as the "Digest" so it's never blank for new uploads,
    # and (unlike the old auto-classifier) it's generated once and cached,
    # so it can't contradict itself across different chat sessions.
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN summary TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notebook_entries (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            source_paper_ids TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Real chat history — separate from notebook_entries (which is an
    # auto-logged findings/open-questions/disagreements digest, not a
    # transcript). Each session is one conversation thread, shown in the
    # left rail like Claude/Gemini's chat history.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )
    """)
    # Thumbs up/down on individual assistant messages.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_feedback (
            message_id TEXT PRIMARY KEY,
            rating TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# A fixed, deliberately chosen palette (not randomly generated) so colors
# stay visually consistent and legible against the app's parchment theme.
PALETTE = [
    "#9C7A2E",  # ochre
    "#2B6E5C",  # forest teal
    "#8A3B2E",  # brick
    "#3B5A8A",  # slate blue
    "#6E4A8A",  # plum
    "#8A6E2E",  # bronze
    "#2E6E8A",  # steel blue
    "#8A2E5A",  # wine
]


def assign_color():
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()["c"]
    conn.close()
    return PALETTE[count % len(PALETTE)]


def add_paper(title, authors, year, filename):
    paper_id = str(uuid.uuid4())[:8]
    color = assign_color()
    conn = get_conn()
    conn.execute(
        "INSERT INTO papers (id, title, authors, year, color, filename, uploaded_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (paper_id, title, authors, year, color, filename,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return paper_id, color


def list_papers():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM papers ORDER BY uploaded_at ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_paper(paper_id):
    conn = get_conn()
    conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
    conn.commit()
    conn.close()


def update_paper(paper_id, title=None, authors=None, year=None):
    """Corrects a paper's registry metadata in place — e.g. fixing a missed
    author name — without deleting and re-ingesting the whole PDF. Note:
    this only updates the SQLite registry (which is what LIBRARY INDEX-based
    questions like 'list all authors' read from). It does NOT rewrite the
    per-chunk metadata already stored in Pinecone, so passages retrieved for
    subject-matter questions may still carry the old authors string in their
    citation metadata until the paper is re-uploaded."""
    conn = get_conn()
    fields, values = [], []
    if title is not None:
        fields.append("title=?"); values.append(title)
    if authors is not None:
        fields.append("authors=?"); values.append(authors)
    if year is not None:
        fields.append("year=?"); values.append(year)
    if fields:
        values.append(paper_id)
        conn.execute(f"UPDATE papers SET {', '.join(fields)} WHERE id=?", values)
        conn.commit()
    conn.close()


def create_session(title="New chat"):
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
        (session_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return session_id


def list_sessions():
    conn = get_conn()
    # Only sessions that actually have at least one message — a session
    # created but never used (e.g. the current blank "New chat" you're
    # sitting on) shouldn't clutter the history list.
    rows = conn.execute("""
        SELECT s.* FROM chat_sessions s
        WHERE EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id)
        ORDER BY s.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def rename_session(session_id, title):
    conn = get_conn()
    conn.execute(
        "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?",
        (title, datetime.now(timezone.utc).isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def touch_session(session_id):
    conn = get_conn()
    conn.execute(
        "UPDATE chat_sessions SET updated_at=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def delete_session(session_id):
    conn = get_conn()
    conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


def add_message(session_id, role, content, sources=None):
    msg_id = str(uuid.uuid4())[:8]
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_messages (id, session_id, role, content, sources, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (msg_id, session_id, role, content,
         json.dumps(sources) if sources is not None else None,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return msg_id


def list_messages(session_id):
    conn = get_conn()
    rows = conn.execute(
        """SELECT m.*, f.rating AS feedback_rating
           FROM chat_messages m
           LEFT JOIN message_feedback f ON f.message_id = m.id
           WHERE m.session_id=? ORDER BY m.created_at ASC""",
        (session_id,),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else None
        out.append(d)
    return out


def add_notebook_entry(entry_type, text, source_paper_ids):
    entry_id = str(uuid.uuid4())[:8]
    conn = get_conn()
    conn.execute(
        "INSERT INTO notebook_entries (id, type, text, source_paper_ids, created_at) "
        "VALUES (?,?,?,?,?)",
        (entry_id, entry_type, text, json.dumps(source_paper_ids),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return entry_id


def list_notebook_entries():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM notebook_entries ORDER BY created_at ASC").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["source_paper_ids"] = json.loads(d["source_paper_ids"])
        out.append(d)
    return out


def set_paper_summary(paper_id, summary):
    conn = get_conn()
    conn.execute("UPDATE papers SET summary=? WHERE id=?", (summary, paper_id))
    conn.commit()
    conn.close()


def list_down_voted():
    conn = get_conn()
    rows = conn.execute("""
        SELECT m.content, m.created_at, s.title AS session_title, s.id AS session_id
        FROM message_feedback f
        JOIN chat_messages m ON m.id = f.message_id
        JOIN chat_sessions s ON s.id = m.session_id
        WHERE f.rating = 'down'
        ORDER BY f.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_message_feedback(message_id, rating):
    conn = get_conn()
    conn.execute(
        "INSERT INTO message_feedback (message_id, rating, created_at) VALUES (?,?,?) "
        "ON CONFLICT(message_id) DO UPDATE SET rating=excluded.rating, created_at=excluded.created_at",
        (message_id, rating, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def clear_message_feedback(message_id):
    conn = get_conn()
    conn.execute("DELETE FROM message_feedback WHERE message_id=?", (message_id,))
    conn.commit()
    conn.close()