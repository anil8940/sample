"""API routes for the LLM application."""

from io import BytesIO
import logging
from typing import Generator
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
import os
from pypdf import PdfReader

from models import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    QuestionRequest,
    QuestionResponse,
    RAGResponse,
    ThreadRequest,
)
from core.llm import llm, answer_with_history, stream_answer_with_history
from core import conversation
from config import settings
from core.rag import ask_rag, clear_rag_history, ingest_documents as ingest_rag_documents, ingest_texts
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the chat UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, 'r', encoding='utf-8') as f:
            return f.read()
    return f"<h1>HTML file not found at {ui_path}</h1>"


@router.get("/api/health", response_model=HealthResponse)
def read_root() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(message="LLM API is running")


@router.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Submit a question and get a response (with conversation history)."""
    # Get history from LangChain memory
    history_text = conversation.get_messages()
    
    # Get response with history
    response_text = answer_with_history(llm, history_text, request.question)
    
    # Add to LangChain memory
    conversation.add_user_message(request.question)
    conversation.add_ai_message(response_text)
    
    return QuestionResponse(question=request.question, response=response_text)


@router.post("/documents", response_model=DocumentIngestResponse, status_code=201)
def ingest_documents(request: DocumentIngestRequest) -> DocumentIngestResponse:
    """Add plain-text documents to the Qdrant knowledge base."""
    chunks_stored = ingest_texts(request.texts, request.source)
    return DocumentIngestResponse(chunks_stored=chunks_stored, source=request.source)


@router.post("/documents/pdf", response_model=DocumentIngestResponse, status_code=201)
async def ingest_pdf(file: UploadFile = File(...)) -> DocumentIngestResponse:
    """Extract text from an uploaded PDF and add its pages to the knowledge base."""
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF must be smaller than {settings.max_upload_size_mb} MB.",
        )

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as error:
        raise HTTPException(status_code=422, detail="The uploaded file is not a readable PDF.") from error

    pages = [
        Document(page_content=text, metadata={"source": filename, "page": page_number})
        for page_number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text()).strip()
    ]
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="No selectable text was found. Scanned PDFs need OCR before upload.",
        )

    try:
        chunks_stored = ingest_rag_documents(pages)
    except Exception as error:
        logger.exception("PDF ingestion failed for %s", filename)
        raise HTTPException(
            status_code=503,
            detail="The embedding service is temporarily unavailable. Please try the upload again shortly.",
        ) from error
    return DocumentIngestResponse(chunks_stored=chunks_stored, source=filename)


@router.post("/rag/ask", response_model=RAGResponse)
def ask_rag_question(request: QuestionRequest) -> RAGResponse:
    """Answer a question through the LangGraph retrieval workflow."""
    response_text, sources = ask_rag(request.question, request.thread_id)
    return RAGResponse(question=request.question, response=response_text, sources=sources)


@router.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest) -> StreamingResponse:
    """Submit a question and stream the response (with conversation history)."""
    # Get history from LangChain memory
    history_text = conversation.get_messages()
    
    def generate() -> Generator[str, None, None]:
        full_response = ""
        for chunk in stream_answer_with_history(llm, history_text, request.question):
            full_response += str(chunk)
            yield str(chunk)
        
        # Add to LangChain memory after streaming completes
        conversation.add_user_message(request.question)
        conversation.add_ai_message(full_response)
    
    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/clear-history")
def clear_conversation(request: ThreadRequest | None = None) -> dict:
    """Clear the legacy chat memory and this thread's LangGraph memory."""
    conversation.clear()
    clear_rag_history((request or ThreadRequest()).thread_id)
    return {"message": "Conversation history cleared"}
