"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., description="The question to ask the LLM")


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    question: str
    response: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    message: str
