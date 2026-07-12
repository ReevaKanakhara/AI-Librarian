"""
notebook.py — after each answer, a small classification call decides whether
the exchange is worth saving as a "finding," "open_question," or
"disagreement." Not every exchange gets logged — only substantive ones.
"""

import json
import db

CLASSIFY_PROMPT = """Given this Q&A exchange from a research assistant, decide if it contains
something worth saving to a running research notebook.

Categories:
- "finding": a concrete result or fact established by the source(s)
- "open_question": something unresolved, or worth investigating further
- "disagreement": two sources conflicting or one contradicting another
- "skip": small talk, a repeat of something already established, or not substantive

Respond ONLY with valid JSON, nothing else:
{{"category": "finding|open_question|disagreement|skip", "summary": "one clear sentence, or empty string if skip"}}

Question: {question}
Answer: {answer}
"""


def maybe_log(groq_client, question: str, answer: str, source_paper_ids: list):
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(question=question, answer=answer)}],
        temperature=0,
        max_tokens=150,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return None

    if parsed.get("category") in ("finding", "open_question", "disagreement") and parsed.get("summary"):
        db.add_notebook_entry(parsed["category"], parsed["summary"], source_paper_ids)
        return parsed
    return None


def export_markdown() -> str:
    entries = db.list_notebook_entries()
    papers = {p["id"]: p for p in db.list_papers()}

    lines = ["# Research Notebook", ""]
    sections = [
        ("finding", "Findings"),
        ("disagreement", "Disagreements"),
        ("open_question", "Open Questions"),
    ]
    for key, label in sections:
        section_entries = [e for e in entries if e["type"] == key]
        if not section_entries:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for e in section_entries:
            sources = ", ".join(papers.get(pid, {}).get("title", pid) for pid in e["source_paper_ids"])
            lines.append(f"- {e['text']} _(Source: {sources})_")
        lines.append("")

    lines.append("## Bibliography")
    lines.append("")
    for p in papers.values():
        lines.append(f"- {p['title']} — {p['authors']} ({p['year']})")

    return "\n".join(lines)
