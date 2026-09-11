import streamlit as st

def setup_page():
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
