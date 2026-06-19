# BusinessIQ 🤖📊💬

**Agentic AI Platform for Analytics, RAG, and WhatsApp Marketing Automation**

BusinessIQ is a cutting-edge, multi-modal AI platform built to revolutionize how businesses interact with their data, documents, and customers. By leveraging state-of-the-art agentic workflows, it unifies tabular data analytics, natural language document querying (RAG), and WhatsApp marketing automation into a single, cohesive, premium Streamlit interface.

---

## 🎥 Demo / How it works

Watch this quick video demonstrating how BusinessIQ works: 

[![BusinessIQ Demo](https://img.youtube.com/vi/GhUTTLJSlvU/maxresdefault.jpg)](https://youtu.be/GhUTTLJSlvU)

---

## 🌟 Key Features

*   **Conversational Data Analytics:** Upload your CSV or Excel files and instantly "talk" to your data. Behind the scenes, an intelligent SQL Agent converts natural language into optimized SQL queries, executing them at lightning speed.
*   **Intelligent Document Retrieval (RAG):** Upload PDF reports or documents and extract answers seamlessly. Using advanced Retrieval-Augmented Generation, the system understands the context of your files and provides accurate, cited responses.
*   **Auto-Generated Executive Insights:** Automatically analyzes uploaded datasets to generate comprehensive executive summaries and extract key performance indicators (KPIs) using Google's Gemini models.
*   **Dynamic Live Dashboarding:** View your tabular data, filtered query results, and deep-dive analytics in a rich, interactive dashboard interface powered by Plotly and Streamlit.
*   **WhatsApp Marketing Automation:** Features a specialized agent routing mechanism to handle WhatsApp marketing campaigns, automation logic, and customer outreach seamlessly.
*   **Multi-Agent Orchestration:** Intelligently routes user queries to the most appropriate specialized agent (SQL, RAG, Analysis, or WhatsApp) for maximum accuracy and efficiency.

---

## 🛠️ Techniques Used & Technical Architecture

The core of BusinessIQ is designed around an **Agentic AI Workflow** using an intelligent router to delegate tasks. 

### 1. Agentic Orchestration (LangGraph & LangChain)
Instead of a single monolithic prompt, the application uses **LangGraph** to build a `StateGraph`. A sophisticated routing function analyzes the user's intent and delegates the query to one of four specialized nodes:
*   `SQL Agent`: For tabular data querying.
*   `RAG Agent`: For document-based question answering.
*   `Analysis Agent`: For generating analytical insights.
*   `WhatsApp Agent`: For marketing automation.

### 2. In-Memory Tabular Processing (DuckDB)
For blazing-fast data analytics, the platform utilizes **DuckDB**. When CSV/Excel files are uploaded, they are parsed via `pandas` and registered as virtual tables in DuckDB. The LLM generates SQL queries against these tables, providing real-time data filtering without the overhead of a traditional database.

### 3. Retrieval-Augmented Generation (FAISS & HuggingFace)
Document intelligence is powered by a robust RAG pipeline:
*   **Parsing:** Extracts text from PDFs using `pdfplumber` and `PyPDF2`.
*   **Embeddings:** Generates dense vector representations using locally run open-source models via `sentence-transformers` and `HuggingFace`.
*   **Vector Store:** Stores and searches embeddings efficiently using **FAISS** (Facebook AI Similarity Search).

### 4. Generative AI Models (Google Gemini)
Leverages Google's **Gemini models** (e.g., `gemini-2.5-flash-lite`) via `langchain-google-genai` for reasoning, SQL generation, executive summarization, and conversational responses.

### 5. Premium Interactive Frontend
The user interface is built on **Streamlit** with custom CSS styling to deliver a premium, dark-mode aesthetic (glassmorphism effects, modern typography, responsive layouts).

---

## 📂 Project Structure

```text
📦 Multi-mode-chatbot
 ┣ 📂 agents               # LangGraph nodes and agent definitions
 ┃ ┣ 📜 analysis_agent.py  # Agent for deep data analysis
 ┃ ┣ 📜 rag_agent.py       # RAG logic for documents
 ┃ ┣ 📜 sql_agent.py       # Text-to-SQL logic
 ┃ ┣ 📜 whatsapp_agent.py  # WhatsApp automation agent
 ┃ ┗ 📜 workflow.py        # LangGraph state machine orchestrator
 ┣ 📂 core                 # Core configurations and state types
 ┃ ┗ 📜 state.py           # LangGraph AgentState definition
 ┣ 📂 data                 # Data ingestion and storage managers
 ┃ ┣ 📜 db_manager.py      # DuckDB connection and query execution
 ┃ ┣ 📜 document_parser.py # PDF and Tabular file parsing logic
 ┃ ┗ 📜 vector_store.py    # FAISS vector database management
 ┣ 📂 ui                   # Streamlit frontend components
 ┃ ┣ 📜 chat.py            # Central chat interface
 ┃ ┣ 📜 dashboard.py       # Analytics and KPI dashboard
 ┃ ┗ 📜 sidebar.py         # Configuration and file upload sidebar
 ┣ 📜 app.py               # Main Streamlit application entry point
 ┣ 📜 requirements.txt     # Python dependencies
 ┗ 📜 .env                 # Environment variables
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Multi-mode-chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (or configure keys directly in the app sidebar) and add your API keys:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   HUGGINGFACE_API_KEY=your_huggingface_token
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 💡 Usage Guide

1. **Configure Keys:** Upon launching the app, enter your Gemini API Key and HuggingFace API Token in the sidebar.
2. **Upload Data:** Upload CSV or Excel files for tabular analytics, or PDF files for document Q&A.
3. **Chat:** Ask questions in the chat interface. For example:
   *   *"What were our top 5 products by revenue last month?"* (Triggers SQL Agent)
   *   *"Summarize the Q3 financial report PDF."* (Triggers RAG Agent)
   *   *"Draft a WhatsApp marketing campaign for the new product launch."* (Triggers WhatsApp Agent)
4. **Explore the Dashboard:** Switch to the dashboard panel to view your raw tabular data, live filtered query results, auto-generated executive summaries, and KPI metrics.
