import streamlit as st
import time
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI Librarian — Dwarka Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0B0E14; color: #c9cdd6; }
.stApp { background-color: #0B0E14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] { background-color: #0e1119; border-right: 1px solid #1a1f2e; }
[data-testid="stSidebar"] * { color: #8b92a5 !important; }

.sidebar-logo { font-family: 'DM Serif Display', serif; font-size: 1.35rem; color: #e8e2d5 !important; padding: 1.2rem 0 0.8rem 0; border-bottom: 1px solid #1e2330; margin-bottom: 1rem; letter-spacing: 0.01em; }
.sidebar-label { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase; color: #3a4055 !important; margin-bottom: 0.4rem; margin-top: 1.2rem; }
.sidebar-history-item { font-family: 'Inter', sans-serif; font-size: 0.78rem; color: #6a7490 !important; padding: 0.5rem 0.75rem; margin-bottom: 0.2rem; border: 1px solid transparent; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-radius: 0; cursor: pointer; }
.sidebar-history-item:hover { background: #141820; border-color: #1e2636; color: #9aaabb !important; }
.sidebar-empty { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #2a3040 !important; padding: 0.4rem 0.2rem; letter-spacing: 0.06em; }
.sidebar-value { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #5a6480 !important; padding: 0.35rem 0.6rem; background: #0B0E14; border: 1px solid #1a1f2e; border-radius: 0; margin-bottom: 0.3rem; }
.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3d8b6e; margin-right: 6px; box-shadow: 0 0 6px #3d8b6e88; }
.sidebar-footer { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: #2a3040 !important; letter-spacing: 0.08em; text-align: center; padding-top: 1rem; border-top: 1px solid #1a1f2e; margin-top: 1.5rem; }

.page-header { margin-bottom: 2rem; padding-bottom: 1.2rem; border-bottom: 1px solid #1a1f2e; }
.page-title { font-family: 'DM Serif Display', serif; font-size: 2.2rem; font-weight: 400; color: #e8e2d5; margin: 0 0 0.3rem 0; }
.page-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: #2e4a6a; }

.source-card { background: #0d1117; border: 1px solid #161c28; border-left: 2px solid #1e3a5a; padding: 0.65rem 0.9rem; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.71rem; color: #4a5a70; line-height: 1.65; border-radius: 0; }
.source-label { font-size: 0.56rem; letter-spacing: 0.12em; text-transform: uppercase; color: #2a3a50; margin-bottom: 0.3rem; }
.latency-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #2e6b50; background: #0b1510; border: 1px solid #1a3025; padding: 0.15rem 0.4rem; margin-top: 0.5rem; display: inline-block; border-radius: 0; letter-spacing: 0.06em; }

.stButton > button { background: #0e1520 !important; border: 1px solid #1e3048 !important; color: #6a9ccc !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important; border-radius: 0 !important; padding: 0.45rem 1.1rem !important; letter-spacing: 0.08em !important; }
.stButton > button:hover { background: #152030 !important; border-color: #2e5a8e !important; color: #aaccee !important; }

.welcome-block { text-align: center; padding: 4rem 2rem; }
.welcome-glyph { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #2e3a55; margin-bottom: 1rem; }
.welcome-text { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #4a5570; line-height: 2; }

hr { border-color: #1a1f2e !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0B0E14; }
::-webkit-scrollbar-thumb { background: #1e2636; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
# chat_history: list of {"role": "user"|"assistant", "content": str}
#               passed directly to Groq API for memory
# display_history: list of {"role", "content", "sources", "latency"}
#                  used only for rendering
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "groq_client" not in st.session_state:
    st.session_state.groq_client = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


# ─── Load Resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    groq_api_key     = get_secret("GROQ_API_KEY")
    pinecone_index   = get_secret("PINECONE_INDEX") or "ai-librarian"

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found.")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found.")

    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
    )
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    groq_client = Groq(api_key=groq_api_key)
    return retriever, groq_client


# ─── History-Aware RAG ────────────────────────────────────────────────────────
def answer_question(groq_client, retriever, chat_history: list, question: str):
    """
    Step 1 — Contextualize: if there is history, rewrite the follow-up
             into a fully standalone question using the LLM.
    Step 2 — Retrieve: query Pinecone with the standalone question.
    Step 3 — Answer: call Groq with full history + context, streaming.
    """

    # ── Step 1: Contextualize ──
    standalone_question = question
    if chat_history:
        ctx_messages = [
            {
                "role": "system",
                "content": (
                    "You are a query contextualization assistant. "
                    "Given the conversation history and a follow-up question, "
                    "rewrite the follow-up as a fully self-contained standalone question. "
                    "Resolve all pronouns and references to prior messages. "
                    "For example, if the user previously asked about authors and now asks "
                    "'What is their research about?', rewrite it as "
                    "'What is the research of [author names] about?'. "
                    "Output ONLY the rewritten question — no explanation."
                ),
            }
        ] + chat_history + [
            {"role": "user", "content": f"Follow-up: {question}"}
        ]
        ctx_resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=ctx_messages,
            temperature=0,
            max_tokens=200,
        )
        standalone_question = ctx_resp.choices[0].message.content.strip()

    # ── Step 2: Retrieve ──
    docs    = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # ── Step 3: Answer (streaming) ──
    qa_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise academic research assistant specialising in "
                "the study of Dwarka's underwater ruins as documented in the provided "
                "research paper. Answer ONLY from the context below. "
                "If the answer is not in the context, say so explicitly.\n\n"
                f"Context:\n{context}"
            ),
        }
    ] + chat_history + [
        {"role": "user", "content": question}
    ]

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=qa_messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    return stream, docs


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">◈ AI Librarian</div>', unsafe_allow_html=True)

    if st.button("+ New Chat", use_container_width=True):
        st.session_state.chat_history    = []
        st.session_state.display_history = []
        st.rerun()

    st.markdown('<div class="sidebar-label">Recent Conversations</div>', unsafe_allow_html=True)
    user_turns = [m for m in st.session_state.display_history if m["role"] == "user"]
    if not user_turns:
        st.markdown('<div class="sidebar-empty">No conversations yet.</div>', unsafe_allow_html=True)
    else:
        for msg in reversed(user_turns):
            t = msg["content"]
            truncated = t[:46] + "..." if len(t) > 46 else t
            st.markdown(f'<div class="sidebar-history-item">{truncated}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Llama 3.3 70B · Groq</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">all-MiniLM-L6-v2 · 384d</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value"><span class="status-dot"></span>Pinecone Connected</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="sidebar-footer">Developed by Reeva Kanakhara</div>',
        unsafe_allow_html=True
    )


# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">Dwarka Research Interface</div>
    <div class="page-subtitle">Retrieval-Augmented Generation &nbsp;·&nbsp; History-Aware Corpus Query</div>
</div>
""", unsafe_allow_html=True)

# Lazy init
if not st.session_state.initialized:
    with st.status("Connecting to knowledge base...", expanded=False):
        try:
            retriever, groq_client = load_resources()
            st.session_state.retriever   = retriever
            st.session_state.groq_client = groq_client
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            st.stop()


# ─── Chat History Display ─────────────────────────────────────────────────────
# Welcome shown only when no messages yet — disappears after first query
if not st.session_state.display_history:
    st.markdown("""
    <div class="welcome-block">
        <div class="welcome-glyph">◈</div>
        <div class="welcome-text">
            Query the academic corpus on Dwarka's submarine ruins.<br>
            Answers are grounded in retrieved source passages.<br>
            Full conversation memory enabled.
        </div>
    </div>
    """, unsafe_allow_html=True)

for entry in st.session_state.display_history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(entry["content"])
            if entry.get("sources"):
                with st.expander("Source References"):
                    for i, src in enumerate(entry["sources"], 1):
                        excerpt = src[:250]
                        suffix  = "..." if len(src) > 250 else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-label">Passage {i}</div>
                            {excerpt}{suffix}
                        </div>
                        """, unsafe_allow_html=True)
            if entry.get("latency"):
                st.markdown(
                    f'<div class="latency-tag">{entry["latency"]:.2f}s response time</div>',
                    unsafe_allow_html=True
                )


# ─── Chat Input ───────────────────────────────────────────────────────────────
query = st.chat_input("Ask about Dwarka's underwater structures, satellite findings, archaeological evidence...")

if query and query.strip():
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query.strip())
    st.session_state.display_history.append({"role": "user", "content": query.strip()})

    full_answer = ""
    source_docs = []
    latency     = None

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        t0 = time.time()
        try:
            stream, source_docs = answer_question(
                st.session_state.groq_client,
                st.session_state.retriever,
                st.session_state.chat_history,
                query.strip(),
            )

            def token_stream():
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    yield token

            full_answer = st.write_stream(token_stream())
            latency = time.time() - t0
            status.update(label="Done", state="complete", expanded=False)

        except Exception as e:
            full_answer = f"Error: {str(e)}"
            st.error(full_answer)
            status.update(label="Error", state="error")

        sources_text = [doc.page_content for doc in source_docs]
        if sources_text:
            with st.expander("Source References"):
                for i, src in enumerate(sources_text, 1):
                    excerpt = src[:250]
                    suffix  = "..." if len(src) > 250 else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-label">Passage {i}</div>
                        {excerpt}{suffix}
                    </div>
                    """, unsafe_allow_html=True)

        if latency:
            st.markdown(
                f'<div class="latency-tag">{latency:.2f}s response time</div>',
                unsafe_allow_html=True
            )

    # Update both histories
    st.session_state.chat_history.append({"role": "user",      "content": query.strip()})
    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.display_history.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources_text if source_docs else [],
        "latency": latency,
    })