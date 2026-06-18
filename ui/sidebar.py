import streamlit as st

def render_sidebar():
    """Renders the left sidebar for configuration and uploads."""
    with st.sidebar:
        st.title("BusinessIQ")
        st.markdown("Your agentic platform for data insights.")
        
        if st.button("🔄 New Chat", use_container_width=True, type="primary", help="Start a completely fresh chat with no memory of previous messages."):
            st.session_state.messages = []
            for key in ["current_sql_query", "current_sql_results", "insights"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
        st.header("1. API Configuration")
        gemini_key = st.text_input("Google Gemini API Key", type="password", help="Required for the LLM agent")
        hf_key = st.text_input("HuggingFace API Token", type="password", help="Required for embeddings")
        
        with st.expander("WhatsApp Configuration (Optional)", expanded=False):
            wa_token = st.text_input("WhatsApp Access Token", type="password", help="Required to send WhatsApp messages")
            wa_phone_id = st.text_input("WhatsApp Phone Number ID", help="Required to send WhatsApp messages")
        
        st.header("2. Data Upload")
        uploaded_files = st.file_uploader(
            "Upload CSV, Excel, or PDF",
            type=["csv", "xlsx", "xls", "pdf"],
            accept_multiple_files=True
        )
        
        st.header("3. Active Filters")
        st.info("Filters will dynamically appear here when tabular data is processed.")
        
        # Save to session state
        if gemini_key:
            st.session_state.gemini_key = gemini_key
        if hf_key:
            st.session_state.hf_key = hf_key
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
        # Save optional keys
        if wa_token:
            st.session_state.wa_token = wa_token
        if wa_phone_id:
            st.session_state.wa_phone_id = wa_phone_id
            
        return {
            "gemini_key": gemini_key,
            "hf_key": hf_key,
            "uploaded_files": uploaded_files,
            "wa_token": wa_token,
            "wa_phone_id": wa_phone_id
        }
