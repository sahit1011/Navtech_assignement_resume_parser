"""
Basic PDF processor using PyPDF2 and pdfplumber
"""

import logging
from .base_processor import BaseFileProcessor

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class PDFProcessor(BaseFileProcessor):
    """Basic PDF processor with fallback options"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF using available libraries"""
        self.validate_file(file_path)
        
        # Try pdfplumber first (better quality)
        if PDFPLUMBER_AVAILABLE:
            try:
                return self._extract_with_pdfplumber(file_path)
            except Exception as e:
                self.logger.warning(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                return self._extract_with_pypdf2(file_path)
            except Exception as e:
                self.logger.warning(f"PyPDF2 failed: {e}")
        
        raise RuntimeError("No PDF processing library available")
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber"""
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2"""
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
