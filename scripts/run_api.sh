#!/bin/bash

# Start the NeuralNav FastAPI backend
# This script activates the virtual environment and starts the API server

set -e

echo "🚀 Starting NeuralNav API..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
echo "Checking dependencies..."
if ! python -c "import fastapi" &> /dev/null; then
    echo "⚠️  Dependencies not found. Installing from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Check if Ollama is running
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama is not running on http://localhost:11434"
    echo "Please start it: ollama serve"
    echo ""
fi

# Start FastAPI
echo "Starting FastAPI backend on http://localhost:8000..."
echo ""
cd backend
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
