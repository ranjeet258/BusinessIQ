from langgraph.graph import StateGraph, START, END
from core.state import AgentState
from agents.sql_agent import create_sql_node
from agents.rag_agent import create_rag_node
from agents.analysis_agent import create_analysis_node
from agents.whatsapp_agent import create_whatsapp_node
from data.db_manager import DuckDBManager
from data.vector_store import VectorStoreManager

def build_graph(db_manager: DuckDBManager, vector_manager: VectorStoreManager, gemini_key: str):
    workflow = StateGraph(AgentState)
    
    sql_node = create_sql_node(db_manager, gemini_key)
    rag_node = create_rag_node(vector_manager, gemini_key)
    analysis_node = create_analysis_node(gemini_key)
    whatsapp_node = create_whatsapp_node(gemini_key)
    
    def router(state: AgentState):
        query = state["messages"][-1].content.lower()
        if "whatsapp" in query:
            return "whatsapp"
        if len(db_manager.tables) > 0:
            return "sql"
        return "rag"
        
    workflow.add_node("sql", sql_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("whatsapp", whatsapp_node)
    
    workflow.add_conditional_edges(START, router, {"sql": "sql", "rag": "rag", "whatsapp": "whatsapp"})
    workflow.add_edge("sql", "analysis")
    workflow.add_edge("analysis", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("whatsapp", END)
    
    return workflow.compile()
