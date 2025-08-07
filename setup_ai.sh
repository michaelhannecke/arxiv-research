#!/bin/bash

echo "Setting up AI Integration for arXiv Document Processor"
echo "===================================================="

# Check if secrets directory exists
if [ ! -d "secrets" ]; then
    echo "Creating secrets directory..."
    mkdir -p secrets
fi

# Check if .env file exists
if [ -f "secrets/.env" ]; then
    echo "Found existing .env file"
    if grep -q "ANTHROPIC_API_KEY" secrets/.env; then
        echo "✓ ANTHROPIC_API_KEY already configured"
    else
        echo "⚠ ANTHROPIC_API_KEY not found in .env file"
        echo ""
        echo "Add your Anthropic API key to secrets/.env:"
        echo "echo 'ANTHROPIC_API_KEY=your-key-here' >> secrets/.env"
    fi
else
    echo "Creating .env file..."
    echo "Please add your Anthropic API key:"
    echo ""
    echo "1. Get your API key from: https://console.anthropic.com"
    echo "2. Run: echo 'ANTHROPIC_API_KEY=your-key-here' > secrets/.env"
    echo ""
    echo "Example:"
    echo "echo 'ANTHROPIC_API_KEY=sk-ant-...' > secrets/.env"
fi

echo ""
echo "After configuring your API key:"
echo "1. Restart the server: ./run.sh"
echo "2. Test AI features: python tests/test_ai_integration.py"
echo ""
echo "AI Features you'll get:"
echo "✓ Intelligent paper summaries"
echo "✓ Section-by-section AI analysis" 
echo "✓ Enhanced keyword extraction"
echo "✓ Technical content understanding"