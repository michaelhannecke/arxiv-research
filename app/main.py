from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.api.routes import api_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    settings.ensure_directories()
    yield
    logger.info("Shutting down application")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    index_file = static_path / "index.html"
    if index_file.exists():
        return index_file.read_text()
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>arXiv Document Processor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .status { color: green; }
            </style>
        </head>
        <body>
            <h1>arXiv Document Processor</h1>
            <p class="status">Service is running!</p>
            <p>API endpoints available at <a href="/docs">/docs</a></p>
        </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "storage_path": str(settings.storage.base_path),
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )