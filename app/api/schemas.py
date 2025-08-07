from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Dict, Any
import re

class ProcessRequest(BaseModel):
    """Request model for processing an arXiv paper"""
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2301.00001' or 'math.GT/0309136')")
    
    @validator("arxiv_id")
    def validate_arxiv_id(cls, v):
        """Validate arXiv ID format"""
        # New format: YYMM.NNNNN or YYMM.NNNNNvN
        new_format = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", v)
        # Old format: archive/YYMMNNN or archive.subj/YYMMNNN
        old_format = re.match(r"^[a-z\-]+(\.[A-Z]{2})?\/\d{7}$", v)
        
        if not (new_format or old_format):
            raise ValueError(
                "Invalid arXiv ID format. Expected format: 'YYMM.NNNNN' or 'archive/YYMMNNN'"
            )
        return v

class ProcessResponse(BaseModel):
    """Response model for paper processing request"""
    status: str = Field(..., description="Processing status (accepted, rejected)")
    task_id: str = Field(..., description="Unique task identifier")
    message: Optional[str] = Field(None, description="Status message")

class TaskStatus(BaseModel):
    """Model for task status information"""
    status: str = Field(..., description="Current status (pending, processing, completed, failed)")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    result: Optional[Dict[str, Any]] = Field(None, description="Processing result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")

class HistoryItem(BaseModel):
    """Model for processing history entries"""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of paper authors")
    processed_date: datetime = Field(..., description="Date when paper was processed")
    summary_path: str = Field(..., description="Path to the generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")

class DocumentResponse(BaseModel):
    """Model for retrieved document"""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of paper authors")
    processed_date: datetime = Field(..., description="Processing date")
    content: str = Field(..., description="Processed document content in Markdown")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")

class ArxivMetadata(BaseModel):
    """Model for arXiv paper metadata"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: datetime
    updated: datetime
    pdf_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None

class Section(BaseModel):
    """Model for document sections"""
    title: str = Field(..., description="Section title")
    level: int = Field(..., ge=1, le=6, description="Section level (1-6)")
    content: str = Field(..., description="Section content")
    summary: Optional[str] = Field(None, description="AI-generated summary")
    subsections: List["Section"] = Field(default_factory=list, description="Nested subsections")

class ProcessedDocument(BaseModel):
    """Model for a fully processed document"""
    metadata: ArxivMetadata
    sections: List[Section]
    full_summary: str = Field(..., description="Complete document summary")
    keywords: List[str] = Field(..., description="Extracted keywords")
    references: List[str] = Field(default_factory=list, description="Paper references")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float = Field(..., description="Processing time in seconds")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Update forward references for nested models
Section.model_rebuild()