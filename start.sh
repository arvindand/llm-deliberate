#!/bin/bash

# LLM Deliberate - Start Script
# Starts both backend and frontend servers

set -e

echo "🔬 Starting LLM Deliberate..."
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install Node.js first."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data/experiments

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "📡 Starting backend on http://localhost:8000..."
$PYTHON_CMD -m backend.main &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start frontend
echo "🎨 Starting frontend on http://localhost:5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ LLM Deliberate is running!"
echo ""
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Wait for either process to exit
wait
