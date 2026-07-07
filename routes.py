"""API routes for the LLM application."""

from typing import Generator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse
import os

from models import QuestionRequest, QuestionResponse, HealthResponse
from core.llm import llm, answer_with_history, stream_answer_with_history
from core import conversation

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
def clear_conversation() -> dict:
    """Clear conversation history."""
    conversation.clear()
    return {"message": "Conversation history cleared"}
