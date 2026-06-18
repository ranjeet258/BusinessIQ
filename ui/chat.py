import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd

def render_chat_interface():
    """Renders the central chat interface and invokes LangGraph."""
    st.markdown("### BusinessIQ: Agentic AI Platform for Analytics, RAG, and 💬 WhatsApp Marketing Automation")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=350)
    
    with chat_container:
        if not st.session_state.messages:
            st.info("Hello! Upload some data, add API keys, and ask me to analyze it.")
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("table_data"):
                    st.dataframe(pd.DataFrame(message["table_data"]), use_container_width=True)

    if prompt := st.chat_input("E.g., What are the total sales?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
        with chat_container:
            with st.chat_message("assistant"):
                if "workflow" not in st.session_state:
                    msg = "Please configure your API keys and upload data to initialize the agent."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    return
                    
                message_placeholder = st.empty()
                message_placeholder.markdown("Agent is analyzing your request...")
                
                # Construct LangGraph State
                lc_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))
                
                initial_state = {
                    "messages": lc_messages,
                    "thread_id": "streamlit_session",
                    "uploaded_files": [],
                    "current_sql_query": st.session_state.get("current_sql_query"),
                    "current_sql_results": st.session_state.get("current_sql_results"),
                    "insights": st.session_state.get("insights")
                }
                
                try:
                    # Invoke Agent Workflow
                    final_state = st.session_state.workflow.invoke(initial_state)
                    
                    # Update global session state with outputs for dashboard
                    if final_state.get("current_sql_query"):
                        st.session_state.current_sql_query = final_state["current_sql_query"]
                    if final_state.get("current_sql_results"):
                        st.session_state.current_sql_results = final_state["current_sql_results"]
                    if final_state.get("insights"):
                        st.session_state.insights = final_state["insights"]
                        
                    # Extract final assistant response
                    agent_message = final_state["messages"][-1].content
                    message_placeholder.markdown(agent_message)
                    
                    table_data = None
                    if final_state.get("current_sql_results") and len(final_state["current_sql_results"]) <= 10:
                        table_data = final_state["current_sql_results"]
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                        
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": agent_message,
                        "table_data": table_data
                    })
                except Exception as e:
                    error_msg = f"Error during agent execution: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
