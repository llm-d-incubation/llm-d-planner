#!/bin/bash

# Start the Planner UI
# This script installs dependencies with uv and starts the Streamlit UI

set -e

echo "🤖 Starting Planner UI..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found! Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync
echo "✅ Dependencies ready"
echo ""

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

uv run streamlit run ui/app.py --server.headless true
