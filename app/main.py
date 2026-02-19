"""
Fully Local PDF Chatbot - FastAPI Backend
Python port of jacoblee93/fully-local-pdf-chatbot

Uses Ollama for LLM + sentence-transformers for embeddings (all local).
"""

import os
import json
import asyncio
import tempfile
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag import RAGChain, ingest_pdf

app = FastAPI(title="Fully Local PDF Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# In-memory session store: session_id -> RAGChain
sessions: dict[str, RAGChain] = {}


# ---------- Schemas ----------

class ChatMessage(BaseModel):
    role: str  # "human" or "ai"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: List[ChatMessage] = []

class IngestResponse(BaseModel):
    session_id: str
    message: str
    chunk_count: int


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path(__file__).parent.parent / "templates" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content="<h1>PDF Chatbot API</h1><p>Use /docs for API docs.</p>")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and index a PDF file. Returns a session_id for subsequent chat calls."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    import uuid
    session_id = str(uuid.uuid4())

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        rag_chain, chunk_count = await asyncio.to_thread(ingest_pdf, tmp_path)
        sessions[session_id] = rag_chain
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        session_id=session_id,
        message=f"PDF indexed successfully ({chunk_count} chunks).",
        chunk_count=chunk_count,
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Stream a response given a user message and session context."""
    rag = sessions.get(request.session_id)
    if rag is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please ingest a PDF first.",
        )

    history = [(m.role, m.content) for m in request.chat_history]

    async def event_stream():
        try:
            async for chunk in rag.astream(request.message, history):
                # Server-Sent Events format
                data = json.dumps({"token": chunk})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted."}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/health")
async def health():
    return {"status": "ok"}
