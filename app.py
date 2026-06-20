import streamlit as st
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface
from ui.dashboard import render_dashboard
from data.db_manager import DuckDBManager
from data.vector_store import VectorStoreManager
from data.document_parser import parse_tabular_data, parse_pdf
from agents.workflow import build_graph
import re

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BusinessIQ: Agentic AI Platform for Analytics, RAG, and 💬 WhatsApp Marketing Automation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Aesthetic CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #4B4B4B;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #00A67E;
        box-shadow: 0 0 0 2px rgba(0, 166, 126, 0.2);
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    hr { border-color: #333333; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    sidebar_config = render_sidebar()
    
    gemini_key = sidebar_config["gemini_key"]
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
    if gemini_key and "vector_manager" in st.session_state:
        st.session_state.workflow = build_graph(
            st.session_state.db_manager,
            st.session_state.vector_manager,
            gemini_key
        )

    # Render Layout
    chat_col, dash_col = st.columns([1, 1], gap="large")
    
    with chat_col:
        render_chat_interface()
        
    with dash_col:
        render_dashboard()

if __name__ == "__main__":
    main()
