"""
FastAPI Backend for Paper Analyzer RAG
Provides REST API endpoints for PDF upload and queries
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.pdf_loader import PDFLoader
from src.rag_engine import RAGEngine
from config.config import HOST, PORT

# Initialize FastAPI app
app = FastAPI(
    title="📚 Paper Analyzer RAG",
    description="AI-powered research paper analyzer using RAG",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_loader = PDFLoader()
rag_engine = RAGEngine()

# Temp directory for uploads
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============== Pydantic Models ==============
class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str


class SourceDocument(BaseModel):
    """Source document model"""
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    sources: List[SourceDocument]


# ============== Routes ==============

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "✓ running",
        "service": "Paper Analyzer RAG",
        "version": "1.0.0"
    }


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """
    Upload a research paper (PDF)
    
    Returns:
        Success message with document info
    """
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process PDF
        chunks = pdf_loader.process_pdf(file_path)
        
        # Add to RAG system
        metadata = {"filename": file.filename, "source": "uploaded"}
        rag_engine.add_documents(chunks, metadata)
        
        return {
            "status": "✓ success",
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "message": f"Paper '{file.filename}' uploaded and indexed successfully"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """
    Query the uploaded papers
    
    Args:
        request: QueryRequest with 'question' field
        
    Returns:
        Answer with source documents
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        answer, sources = rag_engine.query(request.question)
        
        return QueryResponse(
            answer=answer,
            sources=[
                SourceDocument(
                    content=source["content"],
                    metadata=source["metadata"]
                )
                for source in sources
            ]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.delete("/clear")
async def clear_database():
    """Clear vector database"""
    try:
        rag_engine.clear_vector_store()
        
        # Clear uploads
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        return {
            "status": "✓ success",
            "message": "Database and uploads cleared"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing database: {str(e)}"
        )


# ============== Run Server ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=True
    )
