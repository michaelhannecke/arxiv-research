# arXiv Document Processor

A FastAPI-based application that processes arXiv academic papers by downloading PDFs, extracting content, generating AI-powered summaries using Claude, and storing results in structured Markdown format.

## Features

- 📄 Download papers from arXiv by ID
- 📝 Extract text and structure from PDFs
- 🤖 Generate intelligent summaries using Anthropic's Claude API
- 💾 Store processed documents in organized Markdown format
- 🔍 Track processing history
- 🌐 Web interface for easy interaction
- ⚡ Asynchronous processing with background tasks

## Project Status

Currently implemented (Phase 1-5):
- ✅ Core FastAPI application with health checks
- ✅ Configuration management system
- ✅ Storage module for file operations
- ✅ API endpoints structure
- ✅ Pydantic schemas for validation
- ✅ Project directory structure
- ✅ arXiv API integration with rate limiting
- ✅ PDF download and text extraction
- ✅ Document processing and storage
- ✅ Background task processing
- ✅ **AI Integration**: Anthropic Claude API for intelligent summaries
- ✅ **Enhanced Processing**: Section-by-section AI summarization
- ✅ **Smart Keyword Extraction**: AI-powered technical term identification
- ✅ **Comprehensive Summaries**: Multi-level document analysis
- ✅ **Web Interface**: Modern, responsive frontend with real-time progress
- ✅ **Document Viewer**: Interactive paper browser with search and filtering
- ✅ **Theme Support**: Dark/Light mode toggle with persistent preferences
- ✅ **Export Features**: Download processed papers as Markdown files

Next steps (Phase 6+):
- 🚧 Advanced PDF structure parsing (equations, figures, tables)
- 🚧 Citation network analysis
- 🚧 Multi-document processing and batch operations
- 🚧 User authentication and personalization

## Quick Start

### Prerequisites

- Python 3.9+
- uv package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Anthropic API key

### Quick Start Guide

Follow these simple steps to get the arXiv Document Processor running:

#### Step 1: Setup
```bash
# Clone and navigate to the repository
git clone <repository-url>
cd arxiv-research

# Run the automated setup script (recommended)
./run.sh
```

That's it! The `./run.sh` script will automatically:
- Create a Python 3.9 virtual environment
- Install all dependencies with `uv`
- Start the FastAPI server on http://localhost:8000
- Open the web interface in your browser

#### Step 2: Add AI Features (Optional)
For AI-powered summaries, add your Anthropic API key:
```bash
# Get your API key from https://console.anthropic.com
echo 'ANTHROPIC_API_KEY=your-api-key-here' > secrets/.env

# Restart the server to enable AI features
./run.sh
```

#### Manual Setup (Alternative)
If you prefer manual setup:

```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python python3.9
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --port 8000
```

#### Step 3: Access the Application
Once the server is running, you can access:
- **🌐 Web Interface**: http://localhost:8000 - Modern UI with real-time progress tracking
- **📚 API Documentation**: http://localhost:8000/docs - Interactive API explorer  
- **⚡ Health Check**: http://localhost:8000/health - Service status

#### Step 4: Process Your First Paper
1. Open http://localhost:8000 in your browser
2. Enter an arXiv ID (e.g., `1706.03762` for the famous "Attention Is All You Need" paper)
3. Click "Process Paper" and watch the real-time progress
4. Browse your processed papers in the History tab
5. View AI summaries and export as Markdown

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check endpoint
- `POST /api/process` - Process an arXiv paper
- `GET /api/status/{task_id}` - Check processing status
- `GET /api/history` - Get processing history
- `GET /api/document/{arxiv_id}` - Retrieve processed document

## Testing

### Basic Server Testing

Run the test script to verify the server is working:

```bash
# Start the server in one terminal
./run.sh

# In another terminal, run basic tests
source .venv/bin/activate
python tests/test_server.py
```

### arXiv Integration Testing

Test the complete arXiv processing workflow:

```bash
# Start server in one terminal:
./run.sh

# In another terminal, run the integration test:
./test.sh

# Or manually:
source .venv/bin/activate
python tests/test_arxiv_integration.py
```

This will:
- Download a real paper from arXiv (Attention Is All You Need)
- Extract text from the PDF
- Process and store the document
- Verify all API endpoints work correctly

Expected result: Successfully processes the Transformer paper in ~5-10 seconds with 22 sections and 42 references extracted.

### AI Integration Testing

Test the AI-powered summarization features:

```bash
# Start server in one terminal:
./run.sh

# In another terminal, test AI features:
source .venv/bin/activate
python tests/test_ai_integration.py
```

**With API Key**: Generates AI summaries, enhanced keywords, and intelligent section analysis.
**Without API Key**: Falls back to basic processing while maintaining the same output format.

To enable full AI features:
1. Get an Anthropic API key from https://console.anthropic.com
2. Add it to your environment: `echo 'ANTHROPIC_API_KEY=your-key-here' >> secrets/.env`
3. Restart the server

### Web Interface Demo

Explore the modern web interface with a guided tour:

```bash
# Start server, then run the demo
python tests/demo_web_interface.py
```

**Web Interface Features**:
- 🎨 **Modern Design**: Clean, responsive interface that works on all devices
- ⚡ **Real-time Progress**: Watch papers being processed with live updates
- 🧠 **AI Progress Tracking**: See detailed stages from PDF extraction to AI summarization
- 📚 **Smart History**: Browse all processed papers with rich metadata
- 🔍 **Instant Search**: Filter papers by title, author, or keywords in real-time
- 📖 **Document Viewer**: View AI summaries and original content side-by-side
- 🎨 **Dark/Light Themes**: Toggle themes with persistent preferences
- 💾 **Export Options**: Download processed papers as Markdown files
- ⌨️ **Keyboard Shortcuts**: Power user features (Ctrl+K to focus search)

## Project Structure

```
arxiv-research/
├── app/
│   ├── api/          # API routes and schemas
│   ├── core/         # Core functionality (config, storage)
│   ├── services/     # External service integrations
│   └── static/       # Web interface files
├── mcp_server/       # MCP server implementation
├── config/           # Configuration files
├── data/             # Processed documents storage
├── tests/            # Test files
├── requirements.txt  # Python dependencies
├── run.sh           # Startup script
└── README.md        # This file
```

## Configuration

Edit `config/config.yaml` to customize:
- Storage paths
- arXiv API settings
- Anthropic model parameters
- Server configuration
- Logging settings

## Development

### Code Style

Format code before committing:
```bash
black app/ tests/
ruff check app/ tests/
```

### Adding Features

See `arxiv-processor-sdd.md` for the complete architectural design and planned features.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]