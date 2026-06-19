from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from data.vector_store import VectorStoreManager
from core.state import AgentState

def create_rag_node(vector_manager: VectorStoreManager, gemini_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=gemini_key)
    
    def rag_node(state: AgentState):
        query = state["messages"][-1].content
        retriever = vector_manager.get_retriever()
        
        if retriever:
            docs = retriever.invoke(query)
            context = "\\n".join([doc.page_content for doc in docs])
            
            system_msg = SystemMessage(
                content=f"You are an assistant. Answer based on context:\\n{context}"
            )
            response = llm.invoke([system_msg] + state["messages"])
            return {"messages": [response]}
        else:
            return {"messages": [SystemMessage(content="No documents loaded. Please upload a PDF.")]}
            
    return rag_node
