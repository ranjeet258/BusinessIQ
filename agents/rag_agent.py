from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage
from data.vector_store import VectorStoreManager
from core.state import AgentState

def create_rag_node(vector_manager: VectorStoreManager, google_api_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key)
    
    def rag_node(state: AgentState):
        query = state["messages"][-1].content
        retriever = vector_manager.get_retriever() if vector_manager else None
        
        if retriever:
            # 1. Retrieve the documents from FAISS
            docs = retriever.invoke(query)
            context_text = "\n".join([doc.page_content for doc in docs])
            
            # 2. Ask Gemini the question using the context
            system_msg = SystemMessage(
                content=f"You are an assistant. Answer based on context:\n{context_text}"
            )
            response = llm.invoke([system_msg] + state["messages"])
            
            # 3. Return BOTH the messages AND the context so the Evaluator can see it!
            return {
                "messages": [response],
                "context": context_text,
                "current_agent": "rag"
            }
        else:
            return {
                "messages": [AIMessage(content="No documents loaded. Please upload a PDF.")],
                "context": "No context available.",
                "current_agent": "rag"
            }
            
    return rag_node