"""API routes for the LLM application."""

from typing import Generator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse
import os

from models import QuestionRequest, QuestionResponse, HealthResponse
from core.llm import chain

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
    """Submit a question and get a response."""
    response_text = ""
    for chunk in chain.stream({"user_input": request.question}):
        if hasattr(chunk, 'content'):
            response_text += chunk.content
    return QuestionResponse(question=request.question, response=response_text)


@router.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest) -> StreamingResponse:
    """Submit a question and stream the response."""
    def generate() -> Generator[str, None, None]:
        for chunk in chain.stream({"user_input": request.question}):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")
