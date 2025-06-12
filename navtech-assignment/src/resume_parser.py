"""
Resume Parser class for NavTech Assignment
Extracted from main.py for Flask app compatibility
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.output_schema import ResumeData
from config.llm_config import LLMConfig, AVAILABLE_PROVIDERS
from src.file_processors.base_processor import FileProcessorFactory
from src.llm_providers.gemini_llm import GeminiLLMProvider
from src.llm_providers.openai_llm import OpenAILLMProvider
from src.llm_providers.openrouter_llm import OpenRouterLLMProvider

# Import enhanced transformer providers
try:
    from src.llm_providers.enhanced_transformer_llm import EnhancedTransformerProvider
    ENHANCED_TRANSFORMER_AVAILABLE = True
except ImportError:
    ENHANCED_TRANSFORMER_AVAILABLE = False

try:
    from src.llm_providers.improved_layoutlm_transformer import ImprovedLayoutLMProvider
    LAYOUTLM_TRANSFORMER_AVAILABLE = True
except ImportError:
    LAYOUTLM_TRANSFORMER_AVAILABLE = False


class ResumeParser:
    """Main resume parser class"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.providers = self._initialize_providers()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('resume_parser.log')
            ]
        )
        return logging.getLogger(__name__)

    def _initialize_providers(self) -> dict:
        """Initialize all LLM providers"""
        providers = {}

        try:
            providers['gemini'] = GeminiLLMProvider()
            self.logger.info(f"Gemini provider: {'Available' if providers['gemini'].is_available() else 'Not available'}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {e}")

        try:
            providers['openai'] = OpenAILLMProvider()
            self.logger.info(f"OpenAI provider: {'Available' if providers['openai'].is_available() else 'Not available'}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {e}")

        try:
            providers['openrouter'] = OpenRouterLLMProvider()
            self.logger.info(f"OpenRouter provider: {'Available' if providers['openrouter'].is_available() else 'Not available'}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenRouter provider: {e}")

        # Initialize Enhanced Smart PDF + Transformer provider (Best performing)
        try:
            if ENHANCED_TRANSFORMER_AVAILABLE:
                providers['smart_transformer'] = EnhancedTransformerProvider()
                self.logger.info(f"Enhanced Smart Transformer provider: {'Available' if providers['smart_transformer'].is_available() else 'Not available'}")
            else:
                self.logger.warning("Enhanced Transformer provider not available - missing dependencies")
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Smart Transformer provider: {e}")

        # Initialize Improved LayoutLM + Transformer provider (Best performing - 85.7% accuracy)
        try:
            if LAYOUTLM_TRANSFORMER_AVAILABLE:
                providers['layoutlm_transformer'] = ImprovedLayoutLMProvider()
                self.logger.info(f"Improved LayoutLM Transformer provider: {'Available' if providers['layoutlm_transformer'].is_available() else 'Not available'}")
            else:
                self.logger.warning("LayoutLM Transformer provider not available - missing dependencies")
        except Exception as e:
            self.logger.error(f"Failed to initialize LayoutLM Transformer provider: {e}")

        return providers

    def parse_resume(self, file_path: str, llm_provider: str = "gemini") -> ResumeData:
        """Parse resume file and extract structured data"""
        self.logger.info(f"Starting resume parsing: {file_path} with {llm_provider}")

        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")

        # Extract text from file
        try:
            # Choose the best processor for each model type
            if llm_provider in ['smart_transformer', 'layoutlm_transformer'] and file_path.lower().endswith('.pdf'):
                # For enhanced providers, use ImprovedSmartPDFProcessor for best results
                try:
                    from src.file_processors.improved_smart_pdf_processor import ImprovedSmartPDFProcessor
                    processor = ImprovedSmartPDFProcessor()
                    self.logger.info("Using ImprovedSmartPDFProcessor for enhanced models")
                except ImportError:
                    try:
                        from src.file_processors.smart_pdf_processor import SmartPDFProcessor
                        processor = SmartPDFProcessor()
                        self.logger.info("Using SmartPDFProcessor as fallback")
                    except ImportError:
                        processor = FileProcessorFactory.create_processor(file_path)
                        self.logger.warning("Enhanced processors not available, falling back to default")
            else:
                processor = FileProcessorFactory.create_processor(file_path)

            resume_text = processor.extract_text(file_path)
            self.logger.info(f"Extracted {len(resume_text)} characters from resume using {processor.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to extract text from file: {e}")
            raise

        # Get LLM provider
        if llm_provider not in self.providers:
            raise ValueError(f"Unknown LLM provider: {llm_provider}. Available providers: {list(self.providers.keys())}")

        provider = self.providers[llm_provider]

        # Check provider availability with specific error messages
        if not provider.is_available():
            if llm_provider == 'smart_transformer':
                raise ValueError("Enhanced Smart PDF + Transformer provider not available. Please check if enhanced transformer dependencies are installed.")
            elif llm_provider == 'layoutlm_transformer':
                raise ValueError("Enhanced LayoutLM + Transformer provider not available. Please check if LayoutLM dependencies are installed.")
            elif llm_provider == 'gemini':
                raise ValueError("Gemini provider not available. Please add GEMINI_API_KEY to your environment variables.")
            elif llm_provider == 'openai':
                raise ValueError("OpenAI provider not available. Please add OPENAI_API_KEY to your environment variables.")
            elif llm_provider == 'openrouter':
                raise ValueError("OpenRouter provider not available. Please add OPENROUTER_API_KEY to your environment variables.")
            else:
                raise ValueError(f"{llm_provider} provider not available. Please check your configuration.")

        # Extract structured data
        try:
            # For enhanced providers that can use PDF path, pass it along
            if llm_provider in ['smart_transformer', 'layoutlm_transformer']:
                resume_data = provider.extract_resume_data(resume_text, pdf_path=file_path)
            else:
                resume_data = provider.extract_resume_data(resume_text)

            self.logger.info(f"Successfully extracted resume data using {llm_provider}")
            return resume_data
        except Exception as e:
            self.logger.error(f"Failed to extract resume data with {llm_provider}: {e}")
            # Re-raise the exception with provider context
            raise ValueError(f"{llm_provider.title()} parsing failed: {str(e)}")

    def get_available_providers(self) -> list:
        """Get list of available LLM providers"""
        available = []
        for name, provider in self.providers.items():
            if provider and provider.is_available():
                available.append(name)
        return available

    def validate_output(self, resume_data: ResumeData) -> bool:
        """Validate extracted resume data"""
        try:
            # Convert to dict and validate structure
            data_dict = resume_data.to_dict()

            # Check required fields
            required_fields = ["first_name", "last_name", "email", "phone", "address", "summary", "skills", "education_history", "work_history"]

            for field in required_fields:
                if field not in data_dict:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            self.logger.info("Resume data validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False