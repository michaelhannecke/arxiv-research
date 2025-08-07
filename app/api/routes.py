from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional
import logging
from datetime import datetime
import uuid

from app.api.schemas import (
    ProcessRequest,
    ProcessResponse,
    TaskStatus,
    HistoryItem,
    DocumentResponse,
)
from app.core.processor import document_processor, ProcessingError
from app.core.storage import storage_manager

logger = logging.getLogger(__name__)

api_router = APIRouter()

@api_router.post("/process", response_model=ProcessResponse)
async def process_paper(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process an arXiv paper by its ID.
    Downloads the PDF, extracts content, generates summaries, and stores results.
    """
    task_id = str(uuid.uuid4())
    
    # Validate arXiv ID format (done in schema validation)
    logger.info(f"Received processing request for arXiv ID: {request.arxiv_id}")
    
    # Queue background processing
    background_tasks.add_task(process_paper_task, task_id, request.arxiv_id)
    
    return ProcessResponse(
        status="accepted",
        task_id=task_id,
        message=f"Processing of arXiv paper {request.arxiv_id} has been queued",
    )

async def process_paper_task(task_id: str, arxiv_id: str):
    """Background task to process a paper using the document processor"""
    try:
        await document_processor.process_paper(arxiv_id, task_id)
        logger.info(f"Successfully completed processing for task {task_id}")
        
    except ProcessingError as e:
        logger.error(f"Processing error for task {task_id}: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in background task {task_id}: {e}", exc_info=True)

@api_router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    task = document_processor.get_task_status(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(
        status=task["status"],
        progress=task["progress"],
        result=task.get("result"),
        error=task.get("error"),
    )

@api_router.get("/history", response_model=List[HistoryItem])
async def get_processing_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Get the history of processed papers"""
    history = await storage_manager.get_history(limit=limit, offset=offset)
    
    return [HistoryItem(**item) for item in history]

@api_router.get("/document/{arxiv_id}", response_model=DocumentResponse)
async def get_document(arxiv_id: str):
    """Retrieve a processed document by arXiv ID"""
    content = await storage_manager.load_document(arxiv_id)
    
    if content is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get document info from history to populate response
    history = await storage_manager.get_history(limit=1000)  # Get all to find this document
    
    for item in history:
        if item["arxiv_id"] == arxiv_id:
            return DocumentResponse(
                arxiv_id=arxiv_id,
                title=item["title"],
                authors=item["authors"],
                processed_date=datetime.fromisoformat(item["processed_date"]),
                content=content,
                keywords=item["keywords"],
            )
    
    # Fallback if not found in history but document exists
    return DocumentResponse(
        arxiv_id=arxiv_id,
        title=f"Document {arxiv_id}",
        authors=[],
        processed_date=datetime.utcnow(),
        content=content,
        keywords=[],
    )

@api_router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a processing task"""
    success = document_processor.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    
    return {"message": "Task cancelled successfully"}