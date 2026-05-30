"""
Unit tests for RAG components
"""

import unittest
from src.pdf_loader import PDFLoader
import os


class TestPDFLoader(unittest.TestCase):
    """Test PDF loading functionality"""
    
    def setUp(self):
        self.loader = PDFLoader()
    
    def test_chunk_text(self):
        """Test text chunking"""
        test_text = "This is a test. " * 100
        chunks = self.loader.chunk_text(test_text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1200)  # CHUNK_SIZE + overlap
    
    def test_invalid_pdf(self):
        """Test handling of invalid PDF"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_pdf("nonexistent.pdf")


if __name__ == "__main__":
    unittest.main()
