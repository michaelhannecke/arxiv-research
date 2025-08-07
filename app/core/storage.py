import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles
import hashlib

from app.core.config import get_settings
from app.api.schemas import ProcessedDocument, ArxivMetadata

logger = logging.getLogger(__name__)
settings = get_settings()

class StorageManager:
    """Manages file storage operations for processed documents"""
    
    def __init__(self):
        self.base_path = Path(settings.storage.base_path)
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.base_path / "papers",
            self.base_path / "summaries",
            self.base_path / "cache" / "arxiv",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_paper_path(self, arxiv_id: str) -> Path:
        """Get the storage path for a specific paper"""
        # Extract year and month from arXiv ID
        # New format: YYMM.NNNNN -> 2024/01/2401.00001
        # Old format: archive/YYMMNNN -> archive/0701/archive.0701001
        
        clean_id = arxiv_id.replace("/", ".")
        
        if "." in clean_id:
            # New format
            parts = clean_id.split(".")
            if len(parts[0]) == 4 and parts[0].isdigit():
                year = "20" + parts[0][:2]
                month = parts[0][2:4]
            else:
                # Fallback for unexpected formats
                year = datetime.now().strftime("%Y")
                month = datetime.now().strftime("%m")
        else:
            # Very old format or unknown
            year = datetime.now().strftime("%Y")
            month = datetime.now().strftime("%m")
        
        return self.base_path / "papers" / year / month / clean_id.replace(".", "_")
    
    async def save_document(self, document: ProcessedDocument) -> str:
        """Save a processed document to storage"""
        try:
            paper_path = self.get_paper_path(document.metadata.arxiv_id)
            paper_path.mkdir(parents=True, exist_ok=True)
            
            # Save markdown document
            markdown_path = paper_path / "paper.md"
            markdown_content = self._generate_markdown(document)
            async with aiofiles.open(markdown_path, "w", encoding="utf-8") as f:
                await f.write(markdown_content)
            
            # Save metadata as JSON
            metadata_path = paper_path / "metadata.json"
            metadata_dict = document.metadata.dict()
            # Convert datetime objects to strings
            for key in ["published", "updated"]:
                if key in metadata_dict and metadata_dict[key]:
                    metadata_dict[key] = metadata_dict[key].isoformat()
            
            async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(metadata_dict, indent=2))
            
            # Update history index
            await self._update_history(document)
            
            logger.info(f"Saved document {document.metadata.arxiv_id} to {paper_path}")
            return str(markdown_path)
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise
    
    async def load_document(self, arxiv_id: str) -> Optional[str]:
        """Load a processed document from storage"""
        try:
            paper_path = self.get_paper_path(arxiv_id)
            markdown_path = paper_path / "paper.md"
            
            if not markdown_path.exists():
                return None
            
            async with aiofiles.open(markdown_path, "r", encoding="utf-8") as f:
                content = await f.read()
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading document {arxiv_id}: {e}")
            return None
    
    async def save_pdf(self, arxiv_id: str, pdf_content: bytes) -> str:
        """Save a PDF file to storage"""
        try:
            paper_path = self.get_paper_path(arxiv_id)
            paper_path.mkdir(parents=True, exist_ok=True)
            
            pdf_path = paper_path / "original.pdf"
            async with aiofiles.open(pdf_path, "wb") as f:
                await f.write(pdf_content)
            
            logger.info(f"Saved PDF for {arxiv_id} to {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error saving PDF: {e}")
            raise
    
    async def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get processing history"""
        try:
            index_path = self.base_path / "summaries" / "index.json"
            
            if not index_path.exists():
                return []
            
            async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
                content = await f.read()
                history = json.loads(content)
            
            # Sort by date, most recent first
            history.sort(key=lambda x: x.get("processed_date", ""), reverse=True)
            
            # Apply pagination
            return history[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []
    
    async def _update_history(self, document: ProcessedDocument):
        """Update the history index with a new document"""
        try:
            index_path = self.base_path / "summaries" / "index.json"
            
            # Load existing history
            if index_path.exists():
                async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    history = json.loads(content)
            else:
                history = []
            
            # Add new entry
            entry = {
                "arxiv_id": document.metadata.arxiv_id,
                "title": document.metadata.title,
                "authors": document.metadata.authors,
                "processed_date": document.processed_at.isoformat(),
                "summary_path": str(self.get_paper_path(document.metadata.arxiv_id) / "paper.md"),
                "keywords": document.keywords,
            }
            
            # Remove duplicate if exists
            history = [h for h in history if h["arxiv_id"] != document.metadata.arxiv_id]
            history.append(entry)
            
            # Save updated history
            async with aiofiles.open(index_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(history, indent=2))
            
        except Exception as e:
            logger.error(f"Error updating history: {e}")
    
    def _generate_markdown(self, document: ProcessedDocument) -> str:
        """Generate markdown content from a processed document"""
        lines = []
        
        # Title and metadata
        lines.append(f"# {document.metadata.title}")
        lines.append("")
        lines.append(f"**Authors**: {', '.join(document.metadata.authors)}")
        lines.append(f"**arXiv ID**: {document.metadata.arxiv_id}")
        lines.append(f"**Published**: {document.metadata.published.strftime('%Y-%m-%d')}")
        lines.append(f"**Categories**: {', '.join(document.metadata.categories)}")
        lines.append(f"**Keywords**: {', '.join(document.keywords)}")
        lines.append("")
        
        if document.metadata.journal_ref:
            lines.append(f"**Journal Reference**: {document.metadata.journal_ref}")
            lines.append("")
        
        if document.metadata.doi:
            lines.append(f"**DOI**: {document.metadata.doi}")
            lines.append("")
        
        # Comprehensive AI Summary
        lines.append("## AI Summary")
        lines.append(document.full_summary)
        lines.append("")
        
        # Original abstract
        lines.append("## Original Abstract")
        lines.append(document.metadata.abstract)
        lines.append("")
        
        # Sections
        for section in document.sections:
            lines.extend(self._format_section(section))
        
        # References
        if document.references:
            lines.append("## References")
            lines.append("")
            for i, ref in enumerate(document.references, 1):
                lines.append(f"{i}. {ref}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append(f"*Processed on {document.processed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        lines.append(f"*Processing time: {document.processing_time:.2f} seconds*")
        
        return "\n".join(lines)
    
    def _format_section(self, section, level=2) -> List[str]:
        """Format a section and its subsections"""
        lines = []
        
        # Section header
        header_prefix = "#" * min(level, 6)
        lines.append(f"{header_prefix} {section.title}")
        lines.append("")
        
        # AI Summary if available
        if section.summary:
            lines.append("### AI Summary")
            lines.append(section.summary)
            lines.append("")
        
        # Original Content
        if section.summary:
            lines.append("### Original Content")
        
        lines.append(section.content)
        lines.append("")
        
        # Subsections
        for subsection in section.subsections:
            lines.extend(self._format_section(subsection, level + 1))
        
        return lines
    
    async def cache_arxiv_response(self, arxiv_id: str, response: Dict[str, Any]):
        """Cache an arXiv API response"""
        try:
            cache_path = self.base_path / "cache" / "arxiv"
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Create cache key from arxiv_id
            cache_key = hashlib.md5(arxiv_id.encode()).hexdigest()
            cache_file = cache_path / f"{cache_key}.json"
            
            # Add timestamp
            response["cached_at"] = datetime.utcnow().isoformat()
            
            async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(response, indent=2))
            
        except Exception as e:
            logger.error(f"Error caching arXiv response: {e}")
    
    async def get_cached_arxiv_response(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get cached arXiv API response if available and not expired"""
        try:
            cache_path = self.base_path / "cache" / "arxiv"
            cache_key = hashlib.md5(arxiv_id.encode()).hexdigest()
            cache_file = cache_path / f"{cache_key}.json"
            
            if not cache_file.exists():
                return None
            
            async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                content = await f.read()
                response = json.loads(content)
            
            # Check if cache is expired
            cached_at = datetime.fromisoformat(response["cached_at"])
            age = (datetime.utcnow() - cached_at).total_seconds()
            
            if age > settings.storage.cache_duration:
                return None
            
            return response
            
        except Exception as e:
            logger.error(f"Error loading cached response: {e}")
            return None

# Singleton instance
storage_manager = StorageManager()