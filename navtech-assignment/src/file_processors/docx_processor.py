"""
DOCX file processor for Word documents
"""

import logging
from .base_processor import BaseFileProcessor

try:
    from docx import Document
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False


class DOCXProcessor(BaseFileProcessor):
    """DOCX file processor with fallback options"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.doc', '.docx']
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        self.validate_file(file_path)
        
        # Try python-docx first (better structure preservation)
        if PYTHON_DOCX_AVAILABLE:
            try:
                return self._extract_with_python_docx(file_path)
            except Exception as e:
                self.logger.warning(f"python-docx failed: {e}")
        
        # Fallback to docx2txt
        if DOCX2TXT_AVAILABLE:
            try:
                return self._extract_with_docx2txt(file_path)
            except Exception as e:
                self.logger.warning(f"docx2txt failed: {e}")
        
        raise RuntimeError("No DOCX processing library available")
    
    def _extract_with_python_docx(self, file_path: str) -> str:
        """Extract text using python-docx"""
        doc = Document(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        return "\n".join(text_parts)
    
    def _extract_with_docx2txt(self, file_path: str) -> str:
        """Extract text using docx2txt"""
        return docx2txt.process(file_path)
