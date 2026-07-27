"""Pydantic models for request/response validation."""

from typing import Any

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., description="The question to ask the LLM")
    thread_id: str = Field(default="default", min_length=1, description="LangGraph memory thread ID")


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    question: str
    response: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    message: str


class DocumentIngestRequest(BaseModel):
    """Plain-text documents to split and store in the vector database."""

    texts: list[str] = Field(..., min_length=1, description="Documents to ingest")
    source: str = Field(default="api", description="Source label included with citations")


class DocumentIngestResponse(BaseModel):
    chunks_stored: int
    source: str


class RAGResponse(QuestionResponse):
    sources: list[dict[str, Any]]


class ThreadRequest(BaseModel):
    thread_id: str = Field(default="default", min_length=1)
