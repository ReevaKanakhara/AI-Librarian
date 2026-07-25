"""
notebook.py — "Saved Notes" export. Notes are now user-curated: the person
clicks "Save to notes" on an answer they want to keep, rather than an LLM
auto-classifying every exchange as a finding/disagreement/open-question.
The auto-classifier was producing self-contradictory entries over time
(the same author question logged different, conflicting "findings" on
different calls) — a manual save-list has no such contradiction because
it's just what the user chose to keep, verbatim.
"""

import db


def export_markdown() -> str:
    entries = db.list_notebook_entries()
    papers = {p["id"]: p for p in db.list_papers()}

    lines = ["# Saved Notes", ""]

    saved = [e for e in entries if e["type"] == "saved"]
    if saved:
        for e in saved:
            sources = ", ".join(papers.get(pid, {}).get("title", pid) for pid in e["source_paper_ids"])
            lines.append(f"## {e['text'].splitlines()[0].replace('Q: ', '')}")
            lines.append("")
            body = "\n".join(e["text"].splitlines()[1:]).replace("A: ", "", 1)
            lines.append(body)
            if sources:
                lines.append(f"\n_Source: {sources}_")
            lines.append("")

    # Backward-compat: older entries logged before this change used these
    # categories via an auto-classifier that's no longer called.
    legacy_sections = [
        ("finding", "Findings (legacy)"),
        ("disagreement", "Disagreements (legacy)"),
        ("open_question", "Open Questions (legacy)"),
    ]
    for key, label in legacy_sections:
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