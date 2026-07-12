"""
rag.py — retrieval + prompt construction for cross-paper synthesis.
Every retrieved chunk carries which paper it came from, so the model can
(and is instructed to) attribute each claim to a specific source rather than
blending everything anonymously.
"""

from typing import List, Optional


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
        model="llama-3.3-70b-versatile",
        messages=ctx_messages,
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def retrieve_attributed(vectorstore, question: str, active_paper_ids: Optional[List[str]] = None, k: int = 6):
    search_kwargs = {"k": k}
    if active_paper_ids:
        search_kwargs["filter"] = {"paper_id": {"$in": active_paper_ids}}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
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


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis assistant working across multiple academic papers.
Answer using ONLY the context below. For every claim, explicitly attribute it to the
specific paper it came from, using the paper's title — never say just "the context" or "the source."
When papers agree, say so explicitly. When papers disagree, or one has evidence the
other lacks, point that out clearly and directly.
If the answer isn't in the context, say so rather than guessing.

Context:
{context}
"""
