import os
from typing import Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from core.state import AgentState
from agents.sql_agent import create_sql_node
from agents.rag_agent import create_rag_node
from agents.analysis_agent import create_analysis_node
from agents.whatsapp_agent import create_whatsapp_node
from data.db_manager import DuckDBManager
from data.vector_store import VectorStoreManager

load_dotenv()

def build_graph(
    db_manager: Optional[DuckDBManager] = None,
    vector_manager: Optional[VectorStoreManager] = None,
    google_api_key: Optional[str] = None,
    grok_api_key: Optional[str] = None,
    hf_key: Optional[str] = None
) -> StateGraph:
    """Builds and compiles the main LangGraph workflow."""
    
    if db_manager is None:
        db_manager = DuckDBManager()
        
    # Use provided key or fallback to env var
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY") or ""
        
    if not grok_api_key:
        grok_api_key = os.getenv("GROK_API_KEY") or ""
        
    if not hf_key:
        hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""
        
    if vector_manager is None:
        if hf_key:
            try:
                vector_manager = VectorStoreManager(hf_key)
            except Exception:
                vector_manager = None

    workflow_graph = StateGraph(AgentState)
    
    # Initialize Node Functions
    sql_node = create_sql_node(db_manager, google_api_key, grok_api_key, hf_key)
    rag_node = create_rag_node(vector_manager, google_api_key, grok_api_key, hf_key)
    analysis_node = create_analysis_node(google_api_key, grok_api_key, hf_key)
    whatsapp_node = create_whatsapp_node(google_api_key, grok_api_key, hf_key)
    
    def router(state: AgentState):
        query = state["messages"][-1].content.lower()
        if any(w in query for w in ["whatsapp", "campaign", "promo", "broadcast"]):
            return "whatsapp"
        if any(w in query for w in ["pdf", "document", "policy", "contract", "report"]):
            return "rag"
        if any(w in query for w in ["sql", "dataset", "excel", "csv", "table", "rows", "sales", "revenue", "kpi", "kpis", "churn", "count"]):
            return "sql"
        if db_manager and len(db_manager.tables) > 0:
            return "sql"
        return "rag"
        
    workflow_graph.add_node("sql", sql_node)
    workflow_graph.add_node("rag", rag_node)
    workflow_graph.add_node("analysis", analysis_node)
    workflow_graph.add_node("whatsapp", whatsapp_node)
    
    workflow_graph.add_conditional_edges(START, router, {"sql": "sql", "rag": "rag", "whatsapp": "whatsapp"})
    workflow_graph.add_edge("sql", "analysis")
    workflow_graph.add_edge("analysis", END)
    workflow_graph.add_edge("rag", END)
    workflow_graph.add_edge("whatsapp", END)
    
    return workflow_graph.compile()

# Default compiled workflow instance for direct import and evaluation
try:
    workflow = build_graph()
except Exception:
    workflow = None
