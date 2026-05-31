"""Configuration settings"""
import os
import streamlit as st

# Get API key from Streamlit secrets (production) or env var (local)
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector database path
VECTOR_DB_PATH = "./chroma_db"

# LLM parameters
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# PDF processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
