from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM + Embeddings
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# ---------------------------------------------------------------------------
# 2. In-memory PDF retriever store (per thread)
# ---------------------------------------------------------------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Return the FAISS retriever for a thread, or None if not yet indexed."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Index a PDF for a specific chat thread.

    Writes the bytes to a temp file, loads + splits the pages, builds a FAISS
    vector store and stores the retriever keyed by thread_id.

    Returns a summary dict: {filename, documents, chunks}.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(tmp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------

search_tool = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(region="us-en")
)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    ops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
    }
    if operation not in ops:
        return {"error": f"Unsupported operation '{operation}'. Use: add, sub, mul, div"}
    if operation == "div" and second_num == 0:
        return {"error": "Division by zero is not allowed"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": ops[operation](first_num, second_num),
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given ticker symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant passages from the PDF uploaded for this chat thread.
    Always pass the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Please upload a PDF first.",
            "query": query,
        }

    results = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in results],
        "metadata": [doc.metadata for doc in results],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# 4. Graph state
# ---------------------------------------------------------------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 5. Nodes
# ---------------------------------------------------------------------------


def chat_node(state: ChatState, config=None):
    """LLM node — may answer directly or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "For questions about an uploaded PDF, call `rag_tool` and pass "
            f"thread_id=`{thread_id}`. "
            "You may also use web search, stock price lookups, and the calculator "
            "whenever helpful. If no document is available, ask the user to upload a PDF."
        )
    )

    response = llm_with_tools.invoke([system, *state["messages"]], config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# 6. Persistence / checkpointer
# ---------------------------------------------------------------------------

_conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=_conn)

# ---------------------------------------------------------------------------
# 7. Build graph
# ---------------------------------------------------------------------------

_graph = StateGraph(ChatState)
_graph.add_node("chat_node", chat_node)
_graph.add_node("tools", tool_node)
_graph.add_edge(START, "chat_node")
_graph.add_conditional_edges("chat_node", tools_condition)
_graph.add_edge("tools", "chat_node")

chatbot = _graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 8. Helper utilities
# ---------------------------------------------------------------------------


def retrieve_all_threads() -> list:
    """Return all thread IDs stored in the checkpointer."""
    return list({
        cp.config["configurable"]["thread_id"]
        for cp in checkpointer.list(None)
    })


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
