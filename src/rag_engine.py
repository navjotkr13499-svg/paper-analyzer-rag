"""
RAG (Retrieval-Augmented Generation) Engine
Handles embeddings, retrieval, and generation
"""

import os
import shutil
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from config.config import OPENAI_API_KEY, VECTOR_DB_PATH, LLM_TEMPERATURE, LLM_MAX_TOKENS


class RAGEngine:
    """RAG Engine for document retrieval and answer generation"""
    
    def __init__(self):
        """Initialize RAG components"""
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.qa_chain = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load vector database"""
        try:
            # Try to load existing vector store
            self.vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
            print(f"✓ Loaded existing vector store from {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"Creating new vector store: {str(e)}")
            self.vector_store = None
    
    def add_documents(self, chunks: List[str], metadata: dict = None):
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of text chunks
            metadata: Document metadata (e.g., title, source)
        """
        if metadata is None:
            metadata = {}
        
        # Create Document objects with metadata
        documents = [
            Document(page_content=chunk, metadata={**metadata, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        
        # Create or update vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        # Persist to disk
        #self.vector_store.persist()
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def create_qa_chain(self):
        """Create QA chain using retriever"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 similar chunks
        )
        
        llm = ChatOpenAI(
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
    
    def query(self, question: str) -> Tuple[str, List[dict]]:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if self.qa_chain is None:
            self.create_qa_chain()
        
        result = self.qa_chain.invoke({"query": question})
        
        # Format source documents
        sources = [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            }
            for doc in result.get("source_documents", [])
        ]
        
        return result["result"], sources
    
    def clear_vector_store(self):
        """Clear the vector store"""
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        self.vector_store = None
        self.qa_chain = None
        print("✓ Vector store cleared")
