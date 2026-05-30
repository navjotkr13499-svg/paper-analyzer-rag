import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Vector Database
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vectors")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Chunk Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Configuration
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048
