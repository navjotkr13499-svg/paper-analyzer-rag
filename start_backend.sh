#!/bin/bash

echo "🚀 Starting Paper Analyzer Backend..."
source venv/bin/activate
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
