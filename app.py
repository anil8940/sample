"""FastAPI application factory."""

from dotenv import load_dotenv

# Load environment variables first before importing anything else
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        debug=settings.debug,
    )
    
    # Include routes
    app.include_router(router)
    
    # Serve static files (UI)
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")
    
    return app


app = create_app()
