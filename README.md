# Multi-Utility RAG Chatbot

A Streamlit chatbot powered by **LangGraph** + **Gemini 2.5 Flash** that can:

- 📄 Answer questions about an uploaded PDF (RAG via FAISS + HuggingFace embeddings)
- 🌐 Search the web (DuckDuckGo)
- 📈 Look up live stock prices (Alpha Vantage)
- 🧮 Perform arithmetic (calculator tool)
- 💾 Persist conversation history across sessions (SQLite checkpointer)

---

## Project structure

```
.
├── app.py           # Streamlit frontend
├── backend.py       # LangGraph agent, tools, PDF ingestion
├── requirements.txt # Python dependencies
├── .env.example     # Environment variable template
└── .streamlit/
    └── secrets.toml.example   # Streamlit Cloud secrets template
```

---

## Local setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Run the app
streamlit run app.py
```

---

## Streamlit Cloud deployment

1. Push the repo to GitHub (`.env` is in `.gitignore` — do not push it).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo.
3. Set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and paste:

```toml
GOOGLE_API_KEY = "..."
HUGGINGFACEHUB_API_TOKEN = "..."
ALPHA_VANTAGE_API_KEY = "..."
```

5. Click **Deploy**.

> **Note:** Streamlit Cloud has an ephemeral filesystem, so `chatbot.db` (conversation history)
> resets on each redeployment. PDF retrievers are also in-memory and reset on restarts.

---

## API keys needed

| Key | Where to get it |
|-----|-----------------|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `HUGGINGFACEHUB_API_TOKEN` | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `ALPHA_VANTAGE_API_KEY` | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) (free tier available) |
