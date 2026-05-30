#!/bin/bash

echo "📚 Paper Analyzer RAG - Full Stack Launch"
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create it with your OpenAI API key."
    exit 1
fi

echo ""
echo "🚀 Starting services..."
echo "=========================================="
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo "=========================================="
echo ""

python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

streamlit run streamlit_app.py

trap "kill $BACKEND_PID" EXIT
