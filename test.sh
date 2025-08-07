#!/bin/bash

echo "Running arXiv Integration Tests..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found. Please run ./run.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if server is running
echo "Checking if server is running..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Server is running"
else
    echo "ERROR: Server is not running. Please start it with:"
    echo "  ./run.sh"
    echo ""
    echo "Then run this test script in another terminal."
    exit 1
fi

# Run the integration test
echo ""
python tests/test_arxiv_integration.py