"""
db.py — Postgres storage for:
  - papers: the registry of uploaded documents (title, authors, color tag, etc.)
  - chat_sessions / chat_messages: real chat history
  - notebook_entries: user-saved notes
  - message_feedback: thumbs up/down

This used to be SQLite, writing to a local file. Moved to Postgres because
Render's free tier has no persistent disk — every time the free instance
sleeps and wakes back up, it's a fresh container, and a local SQLite file
would be wiped along with it. A hosted Postgres database (e.g. Neon/Supabase
free tier) lives independently of the app server, so it survives restarts.

Every function here has the exact same name and signature as before — only
the internals changed. Nothing outside this file needed to change.

Pinecone still stores the actual embedded chunks. This file only stores the
metadata needed to render the UI.
"""

import os
import json
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

DATABASE_URL = os.getenv("DATABASE_URL", "")


def get_conn():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year TEXT,
            color TEXT NOT NULL,
            filename TEXT,
            uploaded_at TEXT NOT NULL,
            summary TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notebook_entries (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            source_paper_ids TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cur.execute("""
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS message_feedback (
            message_id TEXT PRIMARY KEY,
            rating TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


PALETTE = [
    "#9C7A2E", "#2B6E5C", "#8A3B2E", "#3B5A8A",
    "#6E4A8A", "#8A6E2E", "#2E6E8A", "#8A2E5A",
]


def assign_color():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM papers")
    count = cur.fetchone()["c"]
    cur.close()
    conn.close()
    return PALETTE[count % len(PALETTE)]


def add_paper(title, authors, year, filename):
    paper_id = str(uuid.uuid4())[:8]
    color = assign_color()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO papers (id, title, authors, year, color, filename, uploaded_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (paper_id, title, authors, year, color, filename,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return paper_id, color


def list_papers():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers ORDER BY uploaded_at ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def delete_paper(paper_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM papers WHERE id=%s", (paper_id,))
    conn.commit()
    cur.close()
    conn.close()


def update_paper(paper_id, title=None, authors=None, year=None):
    conn = get_conn()
    cur = conn.cursor()
    fields, values = [], []
    if title is not None:
        fields.append("title=%s"); values.append(title)
    if authors is not None:
        fields.append("authors=%s"); values.append(authors)
    if year is not None:
        fields.append("year=%s"); values.append(year)
    if fields:
        values.append(paper_id)
        cur.execute(f"UPDATE papers SET {', '.join(fields)} WHERE id=%s", values)
        conn.commit()
    cur.close()
    conn.close()


def create_session(title="New chat"):
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (%s,%s,%s,%s)",
        (session_id, title, now, now),
    )
    conn.commit()
    cur.close()
    conn.close()
    return session_id


def ensure_session(session_id, title="New chat"):
    """Creates a session with this EXACT id if it doesn't already exist.
    Used to heal a client-provided session_id that the frontend still holds
    but which the database has no record of (e.g. a browser tab open across
    a database migration) — avoids a foreign key violation on the first
    message sent through it, instead of hard-failing with a 500."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (%s,%s,%s,%s) "
        "ON CONFLICT (id) DO NOTHING",
        (session_id, title, now, now),
    )
    conn.commit()
    cur.close()
    conn.close()


def list_sessions():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.* FROM chat_sessions s
        WHERE EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id)
        ORDER BY s.updated_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chat_sessions WHERE id=%s", (session_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def rename_session(session_id, title):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE chat_sessions SET title=%s, updated_at=%s WHERE id=%s",
        (title, datetime.now(timezone.utc).isoformat(), session_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def touch_session(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE chat_sessions SET updated_at=%s WHERE id=%s",
        (datetime.now(timezone.utc).isoformat(), session_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def delete_session(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
    cur.execute("DELETE FROM chat_sessions WHERE id=%s", (session_id,))
    conn.commit()
    cur.close()
    conn.close()


def add_message(session_id, role, content, sources=None):
    msg_id = str(uuid.uuid4())[:8]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_messages (id, session_id, role, content, sources, created_at) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (msg_id, session_id, role, content,
         json.dumps(sources) if sources is not None else None,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return msg_id


def list_messages(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT m.*, f.rating AS feedback_rating
           FROM chat_messages m
           LEFT JOIN message_feedback f ON f.message_id = m.id
           WHERE m.session_id=%s ORDER BY m.created_at ASC""",
        (session_id,),
    )
    rows = cur.fetchall()
    cur.close()
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
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notebook_entries (id, type, text, source_paper_ids, created_at) "
        "VALUES (%s,%s,%s,%s,%s)",
        (entry_id, entry_type, text, json.dumps(source_paper_ids),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return entry_id


def delete_notebook_entry(entry_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM notebook_entries WHERE id=%s", (entry_id,))
    conn.commit()
    cur.close()
    conn.close()


def list_notebook_entries():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM notebook_entries ORDER BY created_at ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["source_paper_ids"] = json.loads(d["source_paper_ids"])
        out.append(d)
    return out


def set_paper_summary(paper_id, summary):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE papers SET summary=%s WHERE id=%s", (summary, paper_id))
    conn.commit()
    cur.close()
    conn.close()


def list_down_voted():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.content, m.created_at, s.title AS session_title, s.id AS session_id
        FROM message_feedback f
        JOIN chat_messages m ON m.id = f.message_id
        JOIN chat_sessions s ON s.id = m.session_id
        WHERE f.rating = 'down'
        ORDER BY f.created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def set_message_feedback(message_id, rating):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO message_feedback (message_id, rating, created_at) VALUES (%s,%s,%s) "
        "ON CONFLICT (message_id) DO UPDATE SET rating=excluded.rating, created_at=excluded.created_at",
        (message_id, rating, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()


def clear_message_feedback(message_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM message_feedback WHERE message_id=%s", (message_id,))
    conn.commit()
    cur.close()
    conn.close()