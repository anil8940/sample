"""Public RAG interface for ingestion and question answering."""

from __future__ import annotations

from langchain_core.documents import Document

from core.rag_graph import (
    ask_rag as graph_ask_rag,
    clear_rag_history as graph_clear_rag_history,
)
from core.rag_store import ingest_documents as store_ingest_documents, ingest_texts as store_ingest_texts


def ingest_documents(documents: list[Document]) -> int:
    """Split documents into retrieval-friendly chunks and store them in Qdrant."""
    return store_ingest_documents(documents)


def ingest_texts(texts: list[str], source: str) -> int:
    """Store plain-text documents in Qdrant."""
    return store_ingest_texts(texts, source)


def ask_rag(question: str, thread_id: str) -> tuple[str, list[dict[str, str]]]:
    """Ask a question using the RAG graph interface."""
    return graph_ask_rag(question, thread_id)


def clear_rag_history(thread_id: str) -> None:
    """Clear RAG memory for a thread."""
    graph_clear_rag_history(thread_id)
