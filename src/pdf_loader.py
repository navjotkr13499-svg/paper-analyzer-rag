"""
PDF Loading and Processing Module
Handles PDF upload, text extraction, and chunking
"""

import os
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFLoader:
    """Load and process PDF documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text = ""
        try:
            pdf_reader = PdfReader(file_path)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return text
        
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def process_pdf(self, file_path: str) -> List[str]:
        """
        Complete PDF processing: load → extract → chunk
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of processed text chunks
        """
        text = self.load_pdf(file_path)
        chunks = self.chunk_text(text)
        return chunks
