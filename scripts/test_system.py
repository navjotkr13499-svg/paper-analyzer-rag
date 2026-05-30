"""
System test script - tests RAG without API
Useful for local debugging
"""

import sys
sys.path.insert(0, '.')

from src.pdf_loader import PDFLoader
from src.rag_engine import RAGEngine

def test_rag_system():
    """Test RAG system end-to-end"""
    print("🧪 Testing RAG System...\n")
    
    # Initialize components
    loader = PDFLoader()
    rag = RAGEngine()
    
    # Test text chunking
    test_text = """
    Artificial Intelligence is transforming the world. Machine Learning enables computers
    to learn from data. Deep Learning uses neural networks with multiple layers.
    RAG (Retrieval-Augmented Generation) combines retrieval and generation models.
    This technique improves the quality of generated responses by retrieving relevant documents.
    """ * 10
    
    print("✓ Testing text chunking...")
    chunks = loader.chunk_text(test_text)
    print(f"  Created {len(chunks)} chunks\n")
    
    # Test adding to vector store
    print("✓ Testing vector store...")
    rag.add_documents(chunks, metadata={"test": "true", "source": "test_data"})
    print(f"  Vector store ready\n")
    
    # Test querying
    print("✓ Testing query...")
    question = "What is RAG and how does it work?"
    answer, sources = rag.query(question)
    
    print(f"Q: {question}")
    print(f"\nA: {answer[:200]}...\n")
    print(f"Sources found: {len(sources)}\n")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_rag_system()
