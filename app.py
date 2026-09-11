import streamlit as st
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface
from ui.dashboard import render_dashboard
from data.db_manager import DuckDBManager
from data.vector_store import VectorStoreManager
from data.document_parser import parse_tabular_data, parse_pdf
from agents.workflow import build_graph
from ui.style import setup_page
import re

setup_page()

def main():
    sidebar_config = render_sidebar()

    google_api_key = sidebar_config["google_api_key"]
    hf_key = sidebar_config["hf_key"]
    uploaded_files = sidebar_config["uploaded_files"]

    # Initialize Data Managers
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DuckDBManager()
        
    if "vector_manager" not in st.session_state and hf_key:
        try:
            st.session_state.vector_manager = VectorStoreManager(hf_key)
        except Exception as e:
            st.sidebar.error(f"Error initializing Vector Store: {e}")

    # Process Uploaded Files
    if uploaded_files and "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        file_bytes = file.read()
                        if file.name.endswith(('.csv', '.xlsx', '.xls')):
                            df = parse_tabular_data(file.name, file_bytes)
                            raw_name = file.name.split('.')[0].replace(' ', '_').replace('-', '_').lower()
                            table_name = re.sub(r'[^a-z0-9_]', '', raw_name)
                            if not table_name or table_name[0].isdigit():
                                table_name = 't_' + table_name
                            st.session_state.db_manager.register_dataframe(table_name, df)
                            st.session_state.processed_files.add(file.name)
                            st.sidebar.success(f"{file.name} tabular data loaded!")
                        elif file.name.endswith('.pdf'):
                            if "vector_manager" in st.session_state:
                                docs = parse_pdf(file_bytes)
                                st.session_state.vector_manager.ingest_documents(docs)
                                st.session_state.processed_files.add(file.name)
                                st.sidebar.success(f"{file.name} PDF loaded!")
                            else:
                                st.sidebar.warning("Please enter HuggingFace Token to process PDFs.")
                    except Exception as e:
                        st.sidebar.error(f"Error processing {file.name}: {e}")

    # Build LangGraph Workflow
    if google_api_key and "vector_manager" in st.session_state:
        st.session_state.workflow = build_graph(
            st.session_state.db_manager,
            st.session_state.vector_manager,
            google_api_key
        )

    # Render Layout
    chat_col, dash_col = st.columns([1, 1], gap="large")
    
    with chat_col:
        render_chat_interface()
        
    with dash_col:
        render_dashboard()

if __name__ == "__main__":
    main()
