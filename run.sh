#!/bin/bash

# Startup script for arXiv Document Processor

echo "Starting arXiv Document Processor..."
echo "======================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python python3.9
fi

# Activate virtual environment
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p logs data/{papers,summaries,cache/arxiv}

# Check for .env file
if [ ! -f "../secrets/.env" ]; then
    echo ""
    echo "WARNING: No .env file found at ../secrets/.env"
    echo "Please create one with your ANTHROPIC_API_KEY:"
    echo "  echo 'ANTHROPIC_API_KEY=your-key-here' > ../secrets/.env"
    echo ""
fi

# Start the server
echo ""
echo "Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo "Press CTRL+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000