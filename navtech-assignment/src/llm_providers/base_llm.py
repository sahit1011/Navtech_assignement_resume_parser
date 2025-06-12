"""
Base LLM Provider class for resume parsing
Provides common functionality for all LLM providers
"""

import json
import re
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

from config.output_schema import ResumeData, Address, Skill, Education, WorkExperience


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def extract_resume_data(self, resume_text: str, pdf_path: str = None) -> ResumeData:
        """Extract structured data from resume text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": self.__class__.__name__,
            "config": self.config,
            "available": self.is_available()
        }
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean and extract JSON from LLM response"""
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response_text.strip()
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            cleaned_response = self._clean_json_response(response_text)
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return {}
    
    def _get_fallback_data(self, resume_text: str) -> ResumeData:
        """Get fallback data when LLM extraction fails"""
        self.logger.warning("Using fallback data extraction")
        
        # Basic text extraction as fallback
        lines = resume_text.split('\n')
        
        # Try to extract basic information
        email = self._extract_email_fallback(resume_text)
        phone = self._extract_phone_fallback(resume_text)
        
        return ResumeData(
            first_name="",
            last_name="",
            email=email,
            phone=phone,
            address=Address(
                street="",
                city="",
                state="",
                zip_code="",
                country=""
            ),
            summary="",
            skills=[],
            education_history=[],
            work_history=[]
        )
    
    def _extract_email_fallback(self, text: str) -> str:
        """Extract email using regex fallback"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""
    
    def _extract_phone_fallback(self, text: str) -> str:
        """Extract phone using regex fallback"""
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        return ""
    
    def _create_resume_prompt(self, resume_text: str) -> str:
        """Create a standardized prompt for resume parsing"""
        return f"""
Please extract the following information from this resume and return it as a JSON object:

{{
  "first_name": "string",
  "last_name": "string", 
  "email": "string",
  "phone": "string",
  "address": {{
    "street": "string",
    "city": "string", 
    "state": "string",
    "zip_code": "string",
    "country": "string"
  }},
  "summary": "string",
  "skills": ["skill1", "skill2", ...],
  "education_history": [
    {{
      "institution": "string",
      "degree": "string",
      "field_of_study": "string", 
      "graduation_year": "string",
      "gpa": "string"
    }}
  ],
  "work_history": [
    {{
      "company": "string",
      "position": "string",
      "start_date": "string",
      "end_date": "string", 
      "description": "string"
    }}
  ]
}}

Resume text:
{resume_text}

Return only the JSON object, no additional text.
"""
