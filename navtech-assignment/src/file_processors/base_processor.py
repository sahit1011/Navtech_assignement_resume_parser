"""
Base file processor and factory for handling different file types
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseFileProcessor(ABC):
    """Abstract base class for file processors"""
    
    def __init__(self):
        self.supported_extensions = []
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text from file"""
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return True
    
    def get_processing_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the processing applied"""
        return {
            "processor": self.__class__.__name__,
            "file_type": Path(file_path).suffix.lower(),
            "supported_extensions": self.supported_extensions
        }


class FileProcessorFactory:
    """Factory class for creating appropriate file processors"""
    
    @staticmethod
    def create_processor(file_path: str) -> BaseFileProcessor:
        """Create appropriate processor based on file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            try:
                from .improved_smart_pdf_processor import ImprovedSmartPDFProcessor
                return ImprovedSmartPDFProcessor()
            except ImportError:
                try:
                    from .pdf_processor import PDFProcessor
                    return PDFProcessor()
                except ImportError:
                    raise ImportError("No PDF processor available")
        
        elif file_ext in ['.doc', '.docx']:
            try:
                from .docx_processor import DOCXProcessor
                return DOCXProcessor()
            except ImportError:
                raise ImportError("DOCX processor not available")
        
        elif file_ext == '.txt':
            return TextProcessor()
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


class TextProcessor(BaseFileProcessor):
    """Simple text file processor"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.txt']
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from text file"""
        self.validate_file(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
