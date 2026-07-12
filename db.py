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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notebook_entries (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            source_paper_ids TEXT NOT NULL,
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
