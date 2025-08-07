import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import re

import httpx
from app.core.config import get_settings
from app.core.storage import storage_manager
from app.api.schemas import ArxivMetadata

logger = logging.getLogger(__name__)
settings = get_settings()

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0.0
    
    async def wait(self):
        """Wait if necessary to respect rate limit"""
        if self.rate_limit <= 0:
            return
        
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = asyncio.get_event_loop().time()

class ArxivAPIError(Exception):
    """Custom exception for arXiv API errors"""
    pass

class ArxivClient:
    """Client for interacting with the arXiv API"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(settings.arxiv.rate_limit)
        self.client = httpx.AsyncClient(
            timeout=settings.arxiv.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "arXiv Document Processor/1.0 (https://github.com/user/arxiv-research)"
            }
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize arXiv ID format"""
        # Remove any version suffix for consistency
        if 'v' in arxiv_id:
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        return arxiv_id
    
    async def fetch_metadata(self, arxiv_id: str) -> ArxivMetadata:
        """Fetch paper metadata from arXiv API"""
        normalized_id = self._normalize_arxiv_id(arxiv_id)
        
        # Check cache first
        cached_response = await storage_manager.get_cached_arxiv_response(normalized_id)
        if cached_response and "metadata" in cached_response:
            logger.info(f"Using cached metadata for {normalized_id}")
            return ArxivMetadata(**cached_response["metadata"])
        
        # Fetch from API
        await self.rate_limiter.wait()
        
        query_params = {
            'id_list': normalized_id,
            'max_results': 1
        }
        url = f"{settings.arxiv.api_base_url}?{urlencode(query_params)}"
        
        logger.info(f"Fetching metadata for {normalized_id} from {url}")
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse XML response
            metadata = self._parse_arxiv_response(response.text, normalized_id)
            
            # Cache the response
            await storage_manager.cache_arxiv_response(
                normalized_id,
                {"metadata": metadata.dict()}
            )
            
            return metadata
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching metadata: {e}")
            raise ArxivAPIError(f"Failed to fetch metadata: {e}")
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise ArxivAPIError(f"Failed to parse arXiv response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ArxivAPIError(f"Unexpected error fetching metadata: {e}")
    
    def _parse_arxiv_response(self, xml_content: str, arxiv_id: str) -> ArxivMetadata:
        """Parse arXiv API XML response"""
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find the entry
            entry = root.find('.//atom:entry', namespaces)
            if entry is None:
                raise ArxivAPIError(f"No entry found for arXiv ID {arxiv_id}")
            
            # Extract basic information
            title_elem = entry.find('atom:title', namespaces)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', namespaces)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('atom:author', namespaces):
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract dates
            published_elem = entry.find('atom:published', namespaces)
            published_str = published_elem.text if published_elem is not None else ""
            published = self._parse_datetime(published_str)
            
            updated_elem = entry.find('atom:updated', namespaces)
            updated_str = updated_elem.text if updated_elem is not None else published_str
            updated = self._parse_datetime(updated_str)
            
            # Extract categories
            categories = []
            for category_elem in entry.findall('atom:category', namespaces):
                term = category_elem.get('term')
                if term:
                    categories.append(term)
            
            # Extract additional arXiv-specific fields
            comment_elem = entry.find('arxiv:comment', namespaces)
            comment = comment_elem.text if comment_elem is not None else None
            
            journal_elem = entry.find('arxiv:journal_ref', namespaces)
            journal_ref = journal_elem.text if journal_elem is not None else None
            
            doi_elem = entry.find('arxiv:doi', namespaces)
            doi = doi_elem.text if doi_elem is not None else None
            
            # Extract PDF URL
            pdf_url = f"{settings.arxiv.pdf_base_url}/{arxiv_id}.pdf"
            
            return ArxivMetadata(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                comment=comment,
                journal_ref=journal_ref,
                doi=doi
            )
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            logger.debug(f"XML content: {xml_content}")
            raise ArxivAPIError(f"Invalid XML response from arXiv API")
    
    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse datetime from arXiv API format"""
        try:
            # arXiv uses ISO format: 2023-01-01T12:00:00Z
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Failed to parse datetime: {date_str}")
            return datetime.utcnow()
    
    async def download_pdf(self, arxiv_id: str) -> bytes:
        """Download PDF file for a paper"""
        normalized_id = self._normalize_arxiv_id(arxiv_id)
        
        await self.rate_limiter.wait()
        
        pdf_url = f"{settings.arxiv.pdf_base_url}/{normalized_id}.pdf"
        logger.info(f"Downloading PDF from {pdf_url}")
        
        try:
            response = await self.client.get(pdf_url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Check file size
            content_length = len(response.content)
            max_size_bytes = settings.processing.max_pdf_size_mb * 1024 * 1024
            
            if content_length > max_size_bytes:
                raise ArxivAPIError(
                    f"PDF too large: {content_length / 1024 / 1024:.1f}MB "
                    f"(max: {settings.processing.max_pdf_size_mb}MB)"
                )
            
            logger.info(f"Downloaded PDF: {content_length / 1024:.1f}KB")
            return response.content
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF: {e}")
            raise ArxivAPIError(f"Failed to download PDF: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {e}")
            raise ArxivAPIError(f"Unexpected error downloading PDF: {e}")
    
    async def search_papers(
        self, 
        query: str, 
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> List[ArxivMetadata]:
        """Search for papers using arXiv API"""
        await self.rate_limiter.wait()
        
        query_params = {
            'search_query': query,
            'max_results': min(max_results, settings.arxiv.max_results),
            'sortBy': sort_by
        }
        url = f"{settings.arxiv.api_base_url}?{urlencode(query_params)}"
        
        logger.info(f"Searching papers: {query}")
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse XML response and extract multiple entries
            root = ET.fromstring(response.text)
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers = []
            for entry in root.findall('.//atom:entry', namespaces):
                # Extract arXiv ID from the entry ID
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None:
                    entry_id = id_elem.text
                    # Extract arXiv ID from URL like http://arxiv.org/abs/2301.00001v1
                    match = re.search(r'abs/([^v]+)', entry_id)
                    if match:
                        arxiv_id = match.group(1)
                        # Use the existing parsing method
                        paper = self._parse_arxiv_entry(entry, arxiv_id, namespaces)
                        papers.append(paper)
            
            return papers
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error searching papers: {e}")
            raise ArxivAPIError(f"Failed to search papers: {e}")
        except Exception as e:
            logger.error(f"Unexpected error searching papers: {e}")
            raise ArxivAPIError(f"Unexpected error searching papers: {e}")
    
    def _parse_arxiv_entry(self, entry, arxiv_id: str, namespaces: Dict[str, str]) -> ArxivMetadata:
        """Parse a single entry from arXiv search results"""
        # Extract basic information
        title_elem = entry.find('atom:title', namespaces)
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
        
        summary_elem = entry.find('atom:summary', namespaces)
        abstract = summary_elem.text.strip() if summary_elem is not None else ""
        
        # Extract authors
        authors = []
        for author_elem in entry.findall('atom:author', namespaces):
            name_elem = author_elem.find('atom:name', namespaces)
            if name_elem is not None:
                authors.append(name_elem.text.strip())
        
        # Extract dates
        published_elem = entry.find('atom:published', namespaces)
        published_str = published_elem.text if published_elem is not None else ""
        published = self._parse_datetime(published_str)
        
        updated_elem = entry.find('atom:updated', namespaces)
        updated_str = updated_elem.text if updated_elem is not None else published_str
        updated = self._parse_datetime(updated_str)
        
        # Extract categories
        categories = []
        for category_elem in entry.findall('atom:category', namespaces):
            term = category_elem.get('term')
            if term:
                categories.append(term)
        
        # Extract additional fields
        comment_elem = entry.find('arxiv:comment', namespaces)
        comment = comment_elem.text if comment_elem is not None else None
        
        journal_elem = entry.find('arxiv:journal_ref', namespaces)
        journal_ref = journal_elem.text if journal_elem is not None else None
        
        doi_elem = entry.find('arxiv:doi', namespaces)
        doi = doi_elem.text if doi_elem is not None else None
        
        pdf_url = f"{settings.arxiv.pdf_base_url}/{arxiv_id}.pdf"
        
        return ArxivMetadata(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
            comment=comment,
            journal_ref=journal_ref,
            doi=doi
        )

# Singleton instance
async def get_arxiv_client() -> ArxivClient:
    """Get arXiv client instance"""
    return ArxivClient()