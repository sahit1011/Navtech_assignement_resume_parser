"""
Google Gemini LLM provider for resume parsing
"""

import os
import logging
from typing import Dict, Any

from .base_llm import BaseLLMProvider
from config.output_schema import ResumeData, Address, Skill, Education, WorkExperience

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {"model": "gemini-1.5-pro", "temperature": 0.1})
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model"""
        if not GEMINI_AVAILABLE:
            self.logger.error("Google Generative AI library not available")
            return
        
        if not self.api_key:
            self.logger.error("Gemini API key not found")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.config["model"])
            self.logger.info("Gemini model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return GEMINI_AVAILABLE and self.model is not None
    
    def extract_resume_data(self, resume_text: str, pdf_path: str = None) -> ResumeData:
        """Extract resume data using Gemini"""
        if not self.is_available():
            self.logger.error("Gemini model not available")
            return self._get_fallback_data(resume_text)
        
        try:
            prompt = self._create_resume_prompt(resume_text)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config["temperature"]
                )
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")
            
            # Parse JSON response
            data = self._parse_json_response(response.text)
            
            if not data:
                raise ValueError("Failed to parse JSON from Gemini response")
            
            # Create ResumeData object
            resume_data = ResumeData(
                first_name=data.get('first_name', ''),
                last_name=data.get('last_name', ''),
                email=data.get('email', ''),
                phone=data.get('phone', ''),
                address=Address(
                    street=data.get('address', {}).get('street', ''),
                    city=data.get('address', {}).get('city', ''),
                    state=data.get('address', {}).get('state', ''),
                    zip_code=data.get('address', {}).get('zip_code', ''),
                    country=data.get('address', {}).get('country', '')
                ),
                summary=data.get('summary', ''),
                skills=[Skill(name=skill, proficiency="") for skill in data.get('skills', [])],
                education_history=[
                    Education(
                        institution=edu.get('institution', ''),
                        degree=edu.get('degree', ''),
                        field_of_study=edu.get('field_of_study', ''),
                        graduation_year=edu.get('graduation_year', ''),
                        gpa=edu.get('gpa', '')
                    ) for edu in data.get('education_history', [])
                ],
                work_history=[
                    WorkExperience(
                        company=work.get('company', ''),
                        position=work.get('position', ''),
                        start_date=work.get('start_date', ''),
                        end_date=work.get('end_date', ''),
                        description=work.get('description', '')
                    ) for work in data.get('work_history', [])
                ]
            )
            
            self.logger.info("Successfully extracted resume data using Gemini")
            return resume_data
        
        except Exception as e:
            self.logger.error(f"Gemini extraction failed: {e}")
            return self._get_fallback_data(resume_text)
