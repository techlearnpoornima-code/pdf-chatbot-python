"""
RAG (Retrieval Augmented Generation) pipeline.

Mirrors the logic in app/worker.ts from the original JS project:
  1. Load & split PDF
  2. Embed chunks with sentence-transformers (Nomic embed, same model family)
  3. Store in an in-memory Chroma/FAISS vector store
  4. On each query: rephrase question using chat history, retrieve docs, answer

Supports Ollama as the local LLM backend (mirrors the JS Ollama option).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever


# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"  # or llama3, phi3, etc.

# Embedding model - mirrors "Xenova/nomic-embed-text-v1.5" from Transformers.js
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"   # pulled from HuggingFace

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 4  # number of docs to retrieve


# ─── Prompts (ported from worker.ts) ──────────────────────────────────────────

REPHRASE_PROMPT_TEXT = (
    "Given the above conversation, rephrase the following question into a "
    "standalone, natural language query with important keywords that a "
    "researcher could later pass into a search engine to get information "
    "relevant to the conversation. Do not respond with anything except the "
    "query.\n\n"
    "<question_to_rephrase>\n{input}\n</question_to_rephrase>"
)

RESPONSE_SYSTEM_TEMPLATE = (
    "You are an experienced researcher, expert at interpreting and answering "
    "questions based on provided sources. Using the provided context, answer "
    "the user's question to the best of your ability using the resources "
    "provided. Generate a concise answer for a given question based solely on "
    "the provided search results. You must only use information from the "
    "provided search results. Use an unbiased and journalistic tone. Combine "
    "search results together into a coherent answer. Do not repeat text. If "
    "there is nothing in the context relevant to the question at hand, just "
    'say "Hmm, I\'m not sure." Don\'t try to make up an answer. '
    "Anything between the following `context` html blocks is retrieved from a "
    "knowledge bank, not part of the conversation with the user.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "REMEMBER: If there is no relevant information within the context, just "
    'say "Hmm, I\'m not sure." Don\'t try to make up an answer.'
)


# ─── Singleton embedder (loaded once) ─────────────────────────────────────────

_embedder: HuggingFaceEmbeddings | None = None

def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


# ─── Public API ───────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> Tuple["RAGChain", int]:
    """Load, split, embed, and index a PDF. Returns (RAGChain, chunk_count)."""
    # 1. Load
    loader = PyMuPDFLoader(pdf_path)
    docs: List[Document] = loader.load()

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed + index
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)

    # 4. Build chain
    rag = RAGChain(vectorstore)
    return rag, len(chunks)


class RAGChain:
    """Conversational RAG chain backed by a FAISS vector store and Ollama LLM."""

    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self._chain = self._build_chain()

    def _build_chain(self):
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            streaming=True,
        )

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

        # History-aware retriever: rephrase question before retrieval
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", REPHRASE_PROMPT_TEXT),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, rephrase_prompt
        )

        # Answer synthesis
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", RESPONSE_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Full conversational retrieval chain
        chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        return chain

    def _convert_history(self, history: List[Tuple[str, str]]):
        messages = []
        for role, content in history:
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    async def astream(
        self, question: str, history: List[Tuple[str, str]]
    ) -> AsyncIterator[str]:
        """Stream answer tokens."""
        chat_history = self._convert_history(history)
        inputs = {"input": question, "chat_history": chat_history}

        async for chunk in self._chain.astream(inputs):
            # create_retrieval_chain yields dicts; the answer key streams tokens
            if "answer" in chunk:
                yield chunk["answer"]
