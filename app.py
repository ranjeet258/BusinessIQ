import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Utility Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def new_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat() -> None:
    tid = new_thread_id()
    st.session_state["thread_id"] = tid
    _ensure_thread(tid)
    st.session_state["message_history"] = []


def _ensure_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            if content:
                result.append({"role": "assistant", "content": content})
    return result


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = new_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

_ensure_thread(st.session_state["thread_id"])

thread_key: str = st.session_state["thread_id"]
thread_docs: dict = st.session_state["ingested_docs"].setdefault(thread_key, {})
selected_thread = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🤖 LangGraph RAG Chatbot")
st.sidebar.markdown(f"**Thread:** `{thread_key[:8]}…`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# Document status
if thread_docs:
    latest = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 **{latest.get('filename')}**  \n"
        f"{latest.get('chunks')} chunks · {latest.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF indexed for this chat.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# Past conversations
st.sidebar.divider()
st.sidebar.subheader("Past conversations")
past_threads = st.session_state["chat_threads"][::-1]
if not past_threads:
    st.sidebar.caption("No past conversations yet.")
else:
    for tid in past_threads:
        label = f"{'▶ ' if tid == thread_key else ''}{str(tid)[:8]}…"
        if st.sidebar.button(label, key=f"thread-{tid}", use_container_width=True):
            selected_thread = tid

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("Multi-Utility Chatbot")
st.caption("Ask about your PDF, search the web, check stock prices, or do math.")

# Render history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask anything…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: dict = {"box": None}

        def _stream_response():
            for chunk, _ in chatbot.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="messages",
            ):
                # Show tool-use status
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                # Yield AI text tokens
                if isinstance(chunk, AIMessage):
                    if isinstance(chunk.content, list):
                        yield "".join(
                            item.get("text", "")
                            for item in chunk.content
                            if isinstance(item, dict)
                        )
                    elif isinstance(chunk.content, str) and chunk.content:
                        yield chunk.content

        ai_reply = st.write_stream(_stream_response())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Done", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_reply}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 {doc_meta.get('filename')} — "
            f"{doc_meta.get('chunks')} chunks · {doc_meta.get('documents')} pages"
        )

# ---------------------------------------------------------------------------
# Thread switching (sidebar click)
# ---------------------------------------------------------------------------

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["message_history"] = load_conversation(selected_thread)
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
