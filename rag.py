"""
rag.py — retrieval + prompt construction for cross-paper synthesis.
Every retrieved chunk carries which paper it came from, so the model can
(and is instructed to) attribute each claim to a specific source rather than
blending everything anonymously.
"""

from typing import List, Optional
import re


def split_cited_ids(raw_answer: str):
    """Pulls the trailing 'CITED_PAPER_IDS: id1, id2' marker off the model's
    raw output. Returns (clean_answer_without_marker, set_of_ids_or_None).
    None means the marker was missing/unparseable — caller should treat
    that as 'couldn't determine, fall back to showing everything retrieved'
    rather than silently hiding all citations."""
    match = re.search(r"\n?CITED_PAPER_IDS:\s*(.*)\s*$", raw_answer.strip(), re.IGNORECASE)
    if not match:
        return raw_answer, None
    clean = raw_answer[:match.start()].rstrip()
    ids_part = match.group(1).strip()
    if ids_part.lower() in ("(none)", "none", ""):
        return clean, set()
    ids = {x.strip() for x in ids_part.split(",") if x.strip()}
    return clean, ids

_SMALL_TALK = {
    "hey", "hi", "hello", "yo", "sup", "thanks", "thank you", "ok", "okay",
    "cool", "nice", "great", "bye", "goodbye", "good morning", "good night",
    "how are you", "what's up", "whats up",
}

# Questions ABOUT the assistant itself — not about the papers. These were
# slipping through the old gate (e.g. "who created you?" is 3 words and
# doesn't match small talk), so they were going through full retrieval with
# an irrelevant chunk attached, using the paper-synthesis prompt — which is
# also why the model was inventing a fake "team of researchers" backstory
# instead of just answering honestly about what it actually is.
_META_PATTERNS = (
    "who are you", "what are you", "who made you", "who created you",
    "who built you", "who developed you", "what can you do", "what do you do",
    "how do you work", "what is this", "what's this", "what model are you",
    "are you an ai", "are you a bot", "are you chatgpt", "are you claude",
)


def needs_retrieval(question: str) -> bool:
    """Cheap heuristic gate: greetings and questions about the assistant
    itself shouldn't trigger a per-paper retrieval pass or show citation
    chips for an answer that isn't actually grounded in any paper. Anything
    else falls through to normal retrieval — this only short-circuits the
    obvious cases, it never blocks a real question."""
    stripped = question.strip().lower().rstrip("!.?")
    if stripped in _SMALL_TALK:
        return False
    if any(pattern in stripped for pattern in _META_PATTERNS):
        return False
    if len(stripped.split()) <= 2 and "?" not in question and not any(
        kw in stripped for kw in ("paper", "author", "cite", "source", "find", "list", "who", "what", "when", "where", "why", "how")
    ):
        return False
    return True


CASUAL_SYSTEM_PROMPT = """You are the AI Librarian's research assistant — built on an
open-source Llama model, served via Groq, inside an app for searching one person's uploaded
paper library. That's the honest version of who you are; if asked, say so in your own words,
naturally, without reciting it like a script. Don't invent a backstory about being made by an
unnamed "team of researchers" — you just don't know the specifics of your own training, and
that's fine to say plainly.

The user sent a casual message (greeting, thanks, small talk, or something about you rather
than their papers). Respond like a normal conversation — brief, warm, no forced structure. Don't
mention papers or citations unless they actually ask about their library. Never output internal
formatting labels or tags (e.g. "CITED_PAPER_IDS") — plain prose only.
"""


def contextualize(groq_client, question: str, history: list) -> str:
    if not history:
        return question
    ctx_messages = (
        [{
            "role": "system",
            "content": (
                "You are a query contextualization assistant. Rewrite the follow-up "
                "as a fully self-contained standalone question, resolving pronouns and "
                "references to prior messages. Output ONLY the rewritten question."
            ),
        }]
        + history
        + [{"role": "user", "content": f"Follow-up: {question}"}]
    )
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=ctx_messages,
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def retrieve_attributed(vectorstore, question: str, active_paper_ids, k: int = 10):
    """Global top-k semantic search, scoped to the active paper set.

    This used to loop over every active paper and force k_per_paper chunks
    from each one, to guarantee no paper got silently starved out of
    'list everything' answers. That fix overcorrected: it meant ANY
    question — including a narrow 2-paper comparison — pulled chunks from
    every active paper and cited all of them, even ones never discussed in
    the answer.

    The 'list all papers / authors' case doesn't actually need forced
    per-paper chunk retrieval any more — it's answered from LIBRARY INDEX
    (see build_library_index), which is complete text, not a chunk search.
    So this can go back to a normal single top-k search: it naturally
    surfaces only the papers whose content is actually relevant to the
    question, which is what should drive citations."""
    import db
    target_ids = active_paper_ids if active_paper_ids else [p["id"] for p in db.list_papers()]
    if not target_ids:
        return []
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"paper_id": {"$in": target_ids}}}
    )
    return retriever.invoke(question)


def build_context_block(docs) -> str:
    """Group retrieved chunks by paper so the model sees clearly-labeled,
    per-source blocks instead of one anonymous blob of text."""
    grouped = {}
    for d in docs:
        pid = d.metadata.get("paper_id", "unknown")
        grouped.setdefault(pid, {
            "title": d.metadata.get("title", "Untitled"),
            "authors": d.metadata.get("authors", ""),
            "year": d.metadata.get("year", ""),
            "passages": [],
        })
        grouped[pid]["passages"].append(d.page_content)

    blocks = []
    for pid, info in grouped.items():
        header = f"[Paper: {info['title']} ({info['authors']}, {info['year']}) — id:{pid}]"
        body = "\n---\n".join(info["passages"])
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_library_index(papers: list) -> str:
    """A complete, cheap listing of every paper in the library (title/authors/year).
    Included in every prompt so 'list all papers' / 'who are all the authors'
    questions are always fully answerable, regardless of what the vector
    search's top-k happens to retrieve for that specific phrasing."""
    if not papers:
        return "(No papers in the library yet.)"
    lines = []
    for p in papers:
        lines.append(f"- {p['title']} — {p['authors']} ({p['year']}) [id:{p['id']}]")
    return "\n".join(lines)


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis assistant working across multiple academic papers.

You have two kinds of information below:
1. LIBRARY INDEX — a complete list of every paper currently in the library (title, authors, year).
   Use this directly for questions about the library itself: "list all papers," "who are the
   authors of all papers," "how many papers do I have," "what's in my library," etc.
2. RETRIEVED PASSAGES — specific excerpts relevant to the question's subject matter.
   Use this for questions about what the papers actually say, find, or argue.

For every content claim, explicitly attribute it to the specific paper it came from, using
the paper's title. When asked whether papers share something (an author, a method, a finding),
check EVERY entry in the LIBRARY INDEX systematically, one by one, before answering — do not
stop after finding one match if others exist. When papers agree, say so. When they disagree,
or one has evidence the other lacks, point that out clearly. If something isn't covered by
either section, say so rather than guessing.

IMPORTANT — matching authors across papers: two authors are the SAME PERSON only if their
full name matches (allowing for minor formatting differences like initials, honorifics, or
spacing — e.g. "Dr. Darshana H. Patel" and "Darshana Patel" is a match). Sharing only a
surname (e.g. "D. Patel" vs "Reeva Kanakhara... Patel") is NOT sufficient evidence they are
the same individual — surnames are common, and treating them as identical is a real error,
not a reasonable guess. If two entries share a surname but the rest of the name doesn't
clearly match, say they share a surname but may not be the same person, rather than asserting
they are.

Never output internal formatting labels, tags, or metadata fields (e.g. "CITED_PAPER_IDS") in
your answer — respond only in plain natural prose. Citations are handled separately by the app.

IMPORTANT — citations must match what you actually discussed: if the user asks about specific
papers (e.g. "compare X and Y"), your answer should only reference those papers — never pad in
unrelated ones just because they were retrieved. At the very end of your answer, on its own new
line, output exactly: CITED_PAPER_IDS: id1, id2 — listing ONLY the [id:...] values for papers
you genuinely referenced in the answer above, comma-separated, using their bracketed ids from
the LIBRARY INDEX. If you didn't reference any paper (e.g. a casual reply), write
CITED_PAPER_IDS: (none). This line is used to filter citations and will not be shown to the user.

LIBRARY INDEX:
{library_index}

RETRIEVED PASSAGES:
{context}
"""