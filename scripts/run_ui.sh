#!/bin/bash

# Start the NeuralNav UI
# This script activates the virtual environment and starts the Streamlit UI

set -e

echo "🤖 Starting NeuralNav UI..."
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
if ! python -c "import streamlit" &> /dev/null; then
    echo "⚠️  Dependencies not found. Installing from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Check if FastAPI backend is running
echo "Checking if FastAPI backend is running..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Warning: FastAPI backend is not running on http://localhost:8000"
    echo "Please start it in another terminal: scripts/run_api.sh"
    echo ""
fi

# Start Streamlit
echo "Starting Streamlit UI on http://localhost:8501..."
echo ""

# Disable Streamlit's email collection prompt on first run
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

streamlit run ui/app.py --server.headless true
