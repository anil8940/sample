"""Main entry point for the   LLM API application."""

import uvicorn
from config import settings


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        watch=["*.py", "core/*.py"],
    )
