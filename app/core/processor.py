import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import time

from app.services.arxiv_client import get_arxiv_client, ArxivAPIError
from app.services.pdf_processor import pdf_processor, PDFProcessingError
from app.services.anthropic_service import anthropic_service, AnthropicAPIError
from app.core.storage import storage_manager
from app.api.schemas import ProcessedDocument, Section
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class ProcessingError(Exception):
    """Base exception for processing errors"""
    pass

class DocumentProcessor:
    """Main processor that orchestrates arXiv fetching, PDF processing, and storage"""
    
    def __init__(self):
        self.processing_tasks = {}
    
    async def process_paper(self, arxiv_id: str, task_id: str) -> Dict[str, Any]:
        """Process a paper from arXiv ID to final stored document"""
        start_time = time.time()
        
        try:
            # Update task status
            self.processing_tasks[task_id] = {
                "status": "processing",
                "progress": 5,
                "stage": "Fetching metadata",
                "arxiv_id": arxiv_id,
                "started_at": datetime.utcnow()
            }
            
            logger.info(f"Starting processing for {arxiv_id}")
            
            # Step 1: Fetch metadata from arXiv
            async with await get_arxiv_client() as arxiv_client:
                metadata = await arxiv_client.fetch_metadata(arxiv_id)
                logger.info(f"Fetched metadata: {metadata.title}")
                
                self.processing_tasks[task_id]["progress"] = 20
                self.processing_tasks[task_id]["stage"] = "Downloading PDF"
                
                # Step 2: Download PDF
                pdf_bytes = await arxiv_client.download_pdf(arxiv_id)
                logger.info(f"Downloaded PDF: {len(pdf_bytes)} bytes")
            
            self.processing_tasks[task_id]["progress"] = 40
            self.processing_tasks[task_id]["stage"] = "Extracting text from PDF"
            
            # Step 3: Extract content from PDF
            pdf_content = await pdf_processor.extract_content(pdf_bytes)
            logger.info(f"Extracted content: {len(pdf_content['sections'])} sections")
            
            self.processing_tasks[task_id]["progress"] = 40
            self.processing_tasks[task_id]["stage"] = "Processing sections"
            
            # Step 4: Create sections from extracted content
            sections = []
            for section_data in pdf_content['sections']:
                section = Section(
                    title=section_data['title'],
                    level=section_data.get('level', 1),
                    content=pdf_processor.clean_text(section_data['content']),
                    summary=None  # Will be filled by AI
                )
                sections.append(section)
            
            self.processing_tasks[task_id]["progress"] = 50
            self.processing_tasks[task_id]["stage"] = "Generating AI summaries"
            
            # Step 5: Generate AI summaries
            try:
                # Context for AI summarization
                context = {
                    'paper_title': metadata.title,
                    'paper_categories': metadata.categories,
                    'paper_abstract': metadata.abstract
                }
                
                # Generate enhanced abstract summary
                enhanced_abstract = await anthropic_service.summarize_abstract(metadata.abstract, metadata)
                
                self.processing_tasks[task_id]["progress"] = 60
                self.processing_tasks[task_id]["stage"] = "Summarizing sections"
                
                # Summarize sections concurrently
                sections = await anthropic_service.batch_summarize_sections(sections, context)
                
                self.processing_tasks[task_id]["progress"] = 70
                self.processing_tasks[task_id]["stage"] = "Generating comprehensive summary"
                
                # Generate full document summary
                full_summary = await anthropic_service.generate_full_summary(sections, metadata)
                
                self.processing_tasks[task_id]["progress"] = 75
                self.processing_tasks[task_id]["stage"] = "Extracting keywords"
                
                # Extract keywords using AI
                full_text = "\n\n".join([s.content for s in sections])
                keywords = await anthropic_service.extract_keywords(full_text, metadata)
                
            except AnthropicAPIError as e:
                logger.warning(f"AI processing failed, using fallback: {e}")
                # Fallback to basic processing
                enhanced_abstract = f"**Abstract Summary**: {metadata.abstract[:500]}..."
                full_summary = enhanced_abstract
                keywords = self._extract_basic_keywords(metadata)
                
                # Clear any partial summaries
                for section in sections:
                    section.summary = None
            
            self.processing_tasks[task_id]["progress"] = 85
            self.processing_tasks[task_id]["stage"] = "Storing results"
            
            # Step 6: Create processed document
            processing_time = time.time() - start_time
            
            processed_doc = ProcessedDocument(
                metadata=metadata,
                sections=sections,
                full_summary=full_summary,
                keywords=keywords,
                references=pdf_content.get('references', []),
                processed_at=datetime.utcnow(),
                processing_time=processing_time
            )
            
            # Step 6: Save to storage
            document_path = await storage_manager.save_document(processed_doc)
            await storage_manager.save_pdf(arxiv_id, pdf_bytes)
            
            self.processing_tasks[task_id]["progress"] = 100
            self.processing_tasks[task_id]["status"] = "completed"
            self.processing_tasks[task_id]["stage"] = "Completed"
            self.processing_tasks[task_id]["result"] = {
                "arxiv_id": arxiv_id,
                "title": metadata.title,
                "document_path": document_path,
                "processing_time": processing_time,
                "sections_count": len(sections),
                "references_count": len(pdf_content.get('references', []))
            }
            
            logger.info(f"Successfully processed {arxiv_id} in {processing_time:.2f}s")
            
            return self.processing_tasks[task_id]["result"]
            
        except ArxivAPIError as e:
            logger.error(f"arXiv API error for {arxiv_id}: {e}")
            self._mark_task_failed(task_id, f"arXiv API error: {e}")
            raise ProcessingError(f"arXiv API error: {e}")
            
        except PDFProcessingError as e:
            logger.error(f"PDF processing error for {arxiv_id}: {e}")
            self._mark_task_failed(task_id, f"PDF processing error: {e}")
            raise ProcessingError(f"PDF processing error: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error processing {arxiv_id}: {e}", exc_info=True)
            self._mark_task_failed(task_id, f"Unexpected error: {e}")
            raise ProcessingError(f"Unexpected error: {e}")
    
    def _mark_task_failed(self, task_id: str, error_message: str):
        """Mark a task as failed with error message"""
        if task_id in self.processing_tasks:
            self.processing_tasks[task_id]["status"] = "failed"
            self.processing_tasks[task_id]["error"] = error_message
    
    def _extract_basic_keywords(self, metadata) -> list[str]:
        """Extract basic keywords from metadata (enhanced in Phase 4)"""
        keywords = []
        
        # Add categories as keywords
        keywords.extend(metadata.categories[:5])
        
        # Basic keyword extraction from title and abstract
        import re
        text = f"{metadata.title} {metadata.abstract}".lower()
        
        # Common academic/technical terms
        common_terms = [
            r'\b(machine learning|deep learning|neural network|artificial intelligence)\b',
            r'\b(algorithm|optimization|classification|regression)\b',
            r'\b(model|framework|approach|method|technique)\b',
            r'\b(analysis|evaluation|experiment|study|research)\b',
            r'\b(dataset|data|training|testing|validation)\b',
            r'\b(performance|accuracy|precision|recall)\b',
            r'\b(natural language processing|nlp|computer vision|cv)\b',
            r'\b(transformer|attention|embedding|representation)\b'
        ]
        
        for pattern in common_terms:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in keywords:
                    keywords.append(match)
        
        return keywords[:10]  # Limit to 10 keywords
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a processing task"""
        return self.processing_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a processing task"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            if task["status"] in ["pending", "processing"]:
                task["status"] = "cancelled"
                return True
        return False

# Singleton instance
document_processor = DocumentProcessor()