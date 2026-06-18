from typing import Annotated, TypedDict, Any, List, Dict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of the LangGraph agent workflow.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    uploaded_files: List[Dict[str, Any]]
    current_sql_query: Optional[str]
    current_sql_results: Optional[List[Dict[str, Any]]]
    generated_charts: List[Any]
    insights: Optional[str]
