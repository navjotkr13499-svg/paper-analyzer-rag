![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-Classic-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 📚 Research Paper Analyzer - RAG System

An AI-powered system for analyzing research papers using Retrieval-Augmented Generation (RAG).

## 🎯 Features

- 📤 PDF Upload: Upload multiple research papers
- 🔍 Intelligent Search: Semantic search across papers
- 💬 Question Answering: Ask natural language questions
- 📚 Source Tracking: See which papers support each answer
- ⚡ Fast Processing: Optimized chunking and retrieval

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key

## 📊 Project Structure

paper-analyzer-rag/
├── src/
│   ├── main.py
│   ├── pdf_loader.py
│   └── rag_engine.py
├── config/
│   └── config.py
├── streamlit_app.py
├── requirements.txt
├── .env
└── README.md

## 🔑 API Endpoints

### Upload Paper
POST /upload

### Query Papers
POST /query

### Clear Database
DELETE /clear

## 🧪 Testing

Run unit tests:
python -m unittest tests/test_rag.py -v

---

Built with ❤️ using LangChain, FastAPI, and Streamlit
