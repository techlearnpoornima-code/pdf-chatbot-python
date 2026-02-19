# 🏠 Fully Local PDF Chatbot — Python / FastAPI

Chat with any PDF document entirely on your local machine. No OpenAI, no cloud APIs — everything stays private.

---

## ⚡ Stack

 Python 
**FastAPI** (backend) + plain HTML/JS (frontend) |
**LangChain Python** |
**sentence-transformers** (Nomic embed, same model family) |
**FAISS** (in-memory vector store) |
**langchain-ollama** |
**asyncio + thread pool** |

---

## 🚀 Quick Start

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 2. Install Ollama and pull a model

```bash
# Install from https://ollama.ai
ollama pull mistral       # default model
# or: ollama pull llama3, phi3, gemma2, etc.
```

### 3. Create a virtual environment and install dependencies

```bash
uv venv                        # creates .venv/
uv pip install -r requirements.txt
```

> First run will download the Nomic embedding model (~270 MB) from HuggingFace.

### 4. Start the server

```bash
uv run python run.py
```

Open **http://localhost:8000** in your browser.

---

## 🖥 Using the UI

1. **Drop or select a PDF** in the left sidebar.
2. Click **Index PDF** — the document is chunked and embedded locally.
3. **Ask questions** in the chat. The LLM only uses content from your PDF.

---

## 🔌 REST API

The FastAPI backend exposes these endpoints (see `/docs` for interactive docs):

### `POST /ingest`
Upload and index a PDF.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@my_document.pdf"
```

Response:
```json
{
  "session_id": "uuid-here",
  "message": "PDF indexed successfully (42 chunks).",
  "chunk_count": 42
}
```

### `POST /chat`
Ask a question (returns a **Server-Sent Events** stream).

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "uuid-here",
    "message": "What is the main topic of this document?",
    "chat_history": []
  }'
```

Streams:
```
data: {"token": "The"}
data: {"token": " main"}
data: {"token": " topic..."}
data: [DONE]
```

### `DELETE /session/{session_id}`
Free a session and its vector store from memory.

### `GET /health`
Health check.

---

## ⚙️ Configuration

Edit `app/rag.py` to change defaults:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "mistral"          # any model pulled in Ollama
EMBED_MODEL     = "nomic-ai/nomic-embed-text-v1"
CHUNK_SIZE      = 1500
CHUNK_OVERLAP   = 200
TOP_K           = 4                  # docs retrieved per query
```

---

## 🧠 How It Works

```
PDF Upload
   │
   ▼
PyMuPDF (text extraction)
   │
   ▼
RecursiveCharacterTextSplitter  → chunks
   │
   ▼
nomic-embed-text (HuggingFace)  → embeddings
   │
   ▼
FAISS in-memory vector store
   │
User Question ──────────────────────────────────────────────┐
   │                                                         │
   ▼                                                         │
History-aware retriever                                      │
(Ollama rephrases question using chat history)               │
   │                                                         │
   ▼                                                         │
FAISS similarity search → top-K relevant chunks             │
   │                                                         │
   ▼                                                         │
Ollama LLM (Mistral/LLaMA/etc.)                             │
streams answer based only on retrieved context ─────────────┘
```

This mirrors the RAG pipeline in the original `app/worker.ts`.

---

## 📁 Project Structure

```
pdf-chatbot-python/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes
│   └── rag.py           # RAG pipeline (ingest + chain)
├── templates/
│   └── index.html       # Single-page chat UI
├── .venv/               # created by `uv venv` (git-ignored)
├── run.py               # Uvicorn entry point
├── requirements.txt
└── .env.example
```

---

- **No browser required** for inference — everything runs server-side.
- **Persistent sessions** are held in memory (restart clears them; add Redis/DB for persistence).
- **Embeddings** run on CPU via sentence-transformers (GPU automatically used if available).
- The original supports WebLLM and Chrome Gemini Nano; this port targets **Ollama only** (the most practical local option).

---

## License

MIT
