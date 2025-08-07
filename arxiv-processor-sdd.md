# Software Design Document: arXiv Document Processor

## 1. Introduction

### 1.1 Purpose
This document describes the design of a browser-based application that processes arXiv academic papers. The system downloads papers by their arXiv ID, extracts content, generates summaries using the Anthropic API, and stores the results locally in Markdown format.

### 1.2 Scope
The application consists of:
- A FastAPI web server providing a browser interface
- An MCP (Model Context Protocol) server for arXiv interactions
- Integration with Anthropic API for summarization
- Local file storage with configurable paths
- Processing history tracking

### 1.3 Definitions and Acronyms
- **MCP**: Model Context Protocol
- **arXiv**: Open-access repository of scientific papers
- **API**: Application Programming Interface
- **PDF**: Portable Document Format
- **SDD**: Software Design Document

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  Web Browser    │────▶│  FastAPI Server  │────▶│   MCP Server    │
│                 │     │                  │     │                 │
└─────────────────┘     └────────┬─────────┘     └────────┬────────┘
                                 │                         │
                                 │                         ▼
                                 │                 ┌─────────────────┐
                                 │                 │                 │
                                 ▼                 │  arXiv Website  │
                        ┌─────────────────┐        │                 │
                        │                 │        └─────────────────┘
                        │ Anthropic API   │
                        │                 │
                        └─────────────────┘
                                 
                        ┌─────────────────┐
                        │                 │
                        │ Local Storage   │
                        │   (Markdown)    │
                        └─────────────────┘
```

### 2.2 Component Architecture

```
arxiv-processor/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API endpoints
│   │   └── schemas.py       # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration management
│   │   ├── processor.py     # Main processing logic
│   │   └── storage.py       # File storage operations
│   ├── services/
│   │   ├── __init__.py
│   │   ├── anthropic_service.py  # Anthropic API integration
│   │   └── history_service.py    # History tracking
│   └── static/
│       ├── css/
│       ├── js/
│       └── index.html       # Frontend interface
├── mcp_server/
│   ├── __init__.py
│   ├── server.py            # MCP server implementation
│   ├── arxiv_client.py      # arXiv API/PDF handling
│   └── pdf_parser.py        # PDF extraction logic
├── config/
│   └── config.yaml          # Application configuration
├── tests/
│   ├── test_api.py
│   ├── test_processor.py
│   └── test_mcp_server.py
├── requirements.txt
├── README.md
└── setup.py
```

## 3. Detailed Design

### 3.1 FastAPI Web Server

#### 3.1.1 API Endpoints

```python
# GET /
# Returns the web interface (index.html)

# POST /api/process
# Body: { "arxiv_id": "2301.00001" }
# Returns: { "status": "success", "summary_path": "...", "task_id": "..." }

# GET /api/status/{task_id}
# Returns: { "status": "processing|completed|failed", "progress": 0-100 }

# GET /api/history
# Returns: [{ "arxiv_id": "...", "title": "...", "processed_date": "...", "summary_path": "..." }]

# GET /api/document/{arxiv_id}
# Returns the processed markdown content
```

#### 3.1.2 Request/Response Schemas

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ProcessRequest(BaseModel):
    arxiv_id: str
    
class ProcessResponse(BaseModel):
    status: str
    task_id: str
    message: Optional[str]
    
class TaskStatus(BaseModel):
    status: str  # "processing", "completed", "failed"
    progress: int
    result: Optional[dict]
    error: Optional[str]
    
class HistoryItem(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    processed_date: datetime
    summary_path: str
    keywords: List[str]
```

### 3.2 MCP Server Design

#### 3.2.1 Server Interface

```python
class ArxivMCPServer:
    def __init__(self, rate_limit: float = 1.0):
        self.rate_limiter = RateLimiter(rate_limit)
        
    async def fetch_metadata(self, arxiv_id: str) -> dict:
        """Fetch paper metadata from arXiv API"""
        
    async def download_pdf(self, arxiv_id: str) -> bytes:
        """Download PDF with rate limiting"""
        
    async def extract_content(self, pdf_bytes: bytes) -> dict:
        """Extract text and structure from PDF"""
```

#### 3.2.2 PDF Processing Pipeline

```python
class PDFProcessor:
    def extract_sections(self, pdf_bytes: bytes) -> List[Section]:
        """Extract sections with headers and content"""
        
    def extract_figures(self, pdf_bytes: bytes) -> List[Figure]:
        """Extract figure captions and references"""
        
    def extract_references(self, pdf_bytes: bytes) -> List[Reference]:
        """Extract bibliography"""
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
```

### 3.3 Anthropic Integration

#### 3.3.1 Summary Generation

```python
class AnthropicService:
    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key=api_key)
        
    async def summarize_section(self, section: Section) -> str:
        """Generate high-level summary for a section"""
        
    async def extract_keywords(self, full_text: str) -> List[str]:
        """Extract relevant keywords from the paper"""
        
    async def generate_abstract_summary(self, abstract: str) -> str:
        """Create an enhanced abstract summary"""
```

#### 3.3.2 Prompt Templates

```python
SECTION_SUMMARY_PROMPT = """
Summarize the following section from an academic paper. 
Provide a high-level overview focusing on key concepts, findings, and contributions.
Keep technical accuracy while making it accessible.

Section Title: {title}
Section Content: {content}

Summary:
"""

KEYWORD_EXTRACTION_PROMPT = """
Extract 10-15 key technical terms and concepts from this academic paper.
Focus on domain-specific terminology, methodologies, and novel contributions.

Paper Content: {content}

Keywords (comma-separated):
"""
```

### 3.4 Storage Design

#### 3.4.1 Directory Structure

```
configured_base_path/
├── papers/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── 2401.00001/
│   │   │   │   ├── paper.md
│   │   │   │   ├── metadata.json
│   │   │   │   └── original.pdf
├── summaries/
│   └── index.json  # History tracking
└── cache/
    └── arxiv/      # Cached API responses
```

#### 3.4.2 Markdown Output Format

```markdown
# {Paper Title}

**Authors**: {Author1}, {Author2}, ...
**arXiv ID**: {arxiv_id}
**Published**: {date}
**Categories**: {categories}
**Keywords**: {keyword1}, {keyword2}, ...

## Abstract Summary
{enhanced_abstract_summary}

## 1. Introduction
### Summary
{introduction_summary}

### Original Content
{introduction_content}

## 2. {Section Title}
### Summary
{section_summary}

### Original Content
{section_content}

...

## References
{formatted_references}
```

### 3.5 Configuration Management

#### 3.5.1 Configuration File (config.yaml)

```yaml
storage:
  base_path: "./data"
  cache_duration: 86400  # 24 hours in seconds

arxiv:
  api_base_url: "http://export.arxiv.org/api/query"
  pdf_base_url: "https://arxiv.org/pdf"
  rate_limit: 1.0  # requests per second
  timeout: 30

anthropic:
  model: "claude-3-opus-20240229"
  max_tokens: 4096
  temperature: 0.3

processing:
  max_pdf_size_mb: 50
  chunk_size: 1000  # tokens per section for summarization
  
server:
  host: "0.0.0.0"
  port: 8000
  reload: true
```

#### 3.5.2 Environment Variables (.env)

```bash
# ../secrets/.env
ANTHROPIC_API_KEY=sk-ant-...
APP_SECRET_KEY=your-secret-key-here
```

## 4. Data Flow

### 4.1 Processing Pipeline

```
1. User submits arXiv ID via web interface
   ↓
2. FastAPI creates background task
   ↓
3. MCP Server fetches metadata from arXiv API
   ↓
4. MCP Server downloads PDF (with rate limiting)
   ↓
5. PDF Parser extracts text and structure
   ↓
6. Content sent to Anthropic API for summarization
   ↓
7. Results formatted as Markdown
   ↓
8. Files saved to configured directory
   ↓
9. History updated in index.json
   ↓
10. User notified of completion
```

### 4.2 Error Handling Flow

```python
class ProcessingError(Exception):
    """Base exception for processing errors"""
    
class ArxivNotFoundError(ProcessingError):
    """Raised when arXiv ID is not found"""
    
class PDFExtractionError(ProcessingError):
    """Raised when PDF parsing fails"""
    
class AnthropicAPIError(ProcessingError):
    """Raised when Anthropic API calls fail"""
```

## 5. User Interface Design

### 5.1 Main Interface Components

```html
<!-- Simplified structure -->
<div class="container">
    <header>
        <h1>arXiv Document Processor</h1>
    </header>
    
    <section id="input-section">
        <input type="text" id="arxiv-id" placeholder="Enter arXiv ID (e.g., 2301.00001)">
        <button id="process-btn">Process Paper</button>
    </section>
    
    <section id="status-section">
        <div class="progress-bar"></div>
        <div class="status-message"></div>
    </section>
    
    <section id="history-section">
        <h2>Processing History</h2>
        <div id="history-list"></div>
    </section>
</div>
```

### 5.2 Frontend JavaScript Structure

```javascript
class ArxivProcessor {
    constructor() {
        this.apiBase = '/api';
        this.currentTaskId = null;
    }
    
    async processPaper(arxivId) {
        // Submit processing request
        // Start polling for status
        // Update UI with progress
    }
    
    async loadHistory() {
        // Fetch and display processing history
    }
    
    async viewDocument(arxivId) {
        // Load and display processed document
    }
}
```

## 6. Security Considerations

### 6.1 API Security
- API key stored in environment file outside project directory
- Rate limiting on all endpoints
- Input validation for arXiv IDs
- Sanitization of user inputs

### 6.2 File System Security
- Path traversal prevention
- File size limits
- Restricted file types
- Secure temporary file handling

## 7. Testing Strategy

### 7.1 Unit Tests
- PDF parsing accuracy
- Text extraction quality
- API response handling
- Configuration loading

### 7.2 Integration Tests
- End-to-end processing flow
- MCP server communication
- Anthropic API integration
- Storage operations

### 7.3 Mock Services
```python
class MockAnthropicService:
    """Mock for testing without API calls"""
    
class MockArxivService:
    """Mock for testing with sample PDFs"""
```

## 8. Deployment Considerations

### 8.1 Development Setup
```bash
# Clone repository
git clone <repository>
cd arxiv-processor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Create secrets directory and .env file
mkdir ../secrets
echo "ANTHROPIC_API_KEY=your-key-here" > ../secrets/.env

# Run application
python -m app.main
```

### 8.2 Production Deployment
- Use gunicorn/uvicorn for ASGI server
- Configure reverse proxy (nginx)
- Set up process manager (systemd/supervisor)
- Enable HTTPS
- Configure logging and monitoring

## 9. Future Enhancements

### 9.1 Potential Features
- Batch processing of multiple papers
- Export to different formats (LaTeX, HTML)
- Integration with reference managers (Zotero, Mendeley)
- Collaborative features
- Advanced search across processed papers
- Citation network visualization
- Automatic paper recommendations

### 9.2 Scalability Considerations
- Database integration for metadata
- Distributed task queue (Celery)
- Cloud storage options
- Caching layer (Redis)
- Horizontal scaling support