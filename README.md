# 🤖 Multi-Source Intelligent Knowledge Agent (Generalized RAG)

An enterprise-grade, highly adaptable, and **model-agnostic** **Retrieval-Augmented Generation (RAG)** assistant built with Python and LangChain. This system decouples the core mathematical AI layers from client applications, allowing developers to dynamically inject behavioral instructions, switch LLM providers, and upload multi-source private datasets via an external YAML control panel without modifying any source code.

---

## 🎯 Project Overview & Core Philosophy

In production environments, AI assistants must adapt to diverse enterprise ecosystems (such as HR, Finance, or Healthcare) while maintaining modularity. This project solves that challenge by providing a **fully generalized infrastructure** that:
1. **Reads configuration externally** via a `config.yaml` file (App titles, Model definitions, Chunk sizes).
2. **Accepts dynamic behavior prompts** via an interactive UI that updates the system state permanently.
3. **Supports any LLM or Provider**: While configured by default for cost-effective local privacy on-premise (via Ollama), the decoupled architecture allows developers to seamlessly switch to cloud-based enterprise APIs (Google Gemini, OpenAI ChatGPT, Anthropic Claude) simply by swapping the model parameter.

---

## 🏗️ Architectural Hierarchy

```text
📁 multi_source_agent/
│
├── 📁 data/                        # Sample source documents directory
│   ├── 📁 unstructured/            # Text files, PDFs, or unstructured reports
│   └── 📁 structured/              # Excel sheets or CSV data
│
├── 📁 vector_store/                # Local database storing binary FAISS vectors
│
├── 📁 src/                         # Core algorithmic source code
│   ├── __init__.py                 # Package initializer
│   ├── document_manager.py         # Step 1: Document ingestion & NLP Recursive Chunking
│   ├── vector_manager.py           # Step 2: Hugging Face Vector Embeddings & FAISS indexing
│   ├── rag_engine.py               # Step 3: Semantic similarity search & RAG prompt pipeline
│   └── tools_agent.py              # Step 4: Autonomous LangChain tool binding (.bind_tools)
│
├── app.py                          # Streamlit Interactive Web GUI (State management)
├── config.yaml                     # Developer & Admin Control Panel (System Prompts)
└── requirements.txt                # Python library dependency manifests
```

---

## ⚙️ Core Pipeline Mechanics

### 1. Ingestion & Recursive Chunking (`src/document_manager.py`)
Parses binary documents (integrating `pypdf` for complex PDFs) into raw layout-agnostic strings. It leverages LangChain's `RecursiveCharacterTextSplitter` to intelligently divide materials into micro-contexts (default `chunk_size: 500`, `chunk_overlap: 50`) based on structural separators (`\n\n`, `\n`, `.`), preserving semantic continuity.

### 2. High-Dimensional Vectorization (`src/vector_manager.py`)
Translates textual chunks into mathematical coordinate vectors using the state-of-the-art `all-MiniLM-L6-v2` transformer model from Hugging Face. The generated semantic numbers are indexed into a local, lightweight Meta `FAISS` vector database.

### 3. Context-Augmented Generation & Model Agnosticism (`src/rag_engine.py`)
Executes an ultra-fast semantic cosine similarity search on the local FAISS database (retrieving `k=3` neighboring chunks). The extracted text is injected into a dynamically constructed **Augmented Prompt** alongside the developer's custom instructions. Thanks to LangChain's standardized chat model interface, the engine can invoke local offline models (`llama3.1`) or instantly pivot to commercial cloud APIs (`gemini-1.5-flash`), depending entirely on the developer's infrastructure requirements.

### 4. Stateful UI orchestration (`app.py`)
Built with `Streamlit`, featuring a streamlined developer administration panel in the sidebar. It uses `st.session_state` to permanently memorize the uploaded database and the continuous conversation history, preventing redundant model calculations on page reloads.

---

## 🛠️ Installation & Deployment Options

### Option A: 100% Local Run (Default & Cost-Effective)
Ensure you have Python 3.10+ installed and the **Ollama** desktop application running locally on your machine.
Download your preferred local LLM model via your terminal:
```bash
ollama run llama3.1
```

### Option B: Cloud API Integration (Gemini, OpenAI, etc.)
If your enterprise configuration requires larger commercial models, simply plug in your API key into your environment variables:
```bash
# Example for Windows CMD
set GOOGLE_API_KEY="your_api_key_here"
```

### Setup & Launch
```bash
# Clone the repository
git clone https://github.com
cd Local-Intelligent-AI-Agent

# Install the dependencies
pip install -r requirements.txt

# Launch the application
python -m streamlit run app.py
```
Open `http://localhost:8501` in your web browser to interact with the GUI.

---

## 📊 Sample Live Showcase (Enterprise Test Case)
During development testing, the framework was injected with the **Official Miathon '03 Regulatory Framework PDF** (written in French) and configured with a Developer Persona to act as an *Enthusiastic English Event Organizer*. 

* **Input query:** `"Les participants doivent-ils apporter un équipement de couchage spécifique ?"`
* **Local AI Output:** *"Welcome to the Miathon! We are thrilled to have you! Regarding your question, yes, participants are invited to bring their own sleeping bags to rest on-site in case of fatigue. We want to ensure you have a comfortable experience!"* 

This successfully demonstrates the pipeline's capability to execute cross-lingual semantic extraction, automatic contextual translations, and strict behavioral adherence in real-time.