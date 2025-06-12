"""
OpenAI GPT LLM provider for resume parsing
"""

import os
import logging
from typing import Dict, Any

from .base_llm import BaseLLMProvider
from config.output_schema import ResumeData, Address, Skill, Education, WorkExperience

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI GPT LLM provider"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {"model": "gpt-3.5-turbo", "temperature": 0.1})
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI library not available")
            return
        
        if not self.api_key:
            self.logger.error("OpenAI API key not found")
            return
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return OPENAI_AVAILABLE and self.client is not None
    
    def extract_resume_data(self, resume_text: str, pdf_path: str = None) -> ResumeData:
        """Extract resume data using OpenAI GPT"""
        if not self.is_available():
            self.logger.error("OpenAI client not available")
            return self._get_fallback_data(resume_text)
        
        try:
            prompt = self._create_resume_prompt(resume_text)
            
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured data from resumes. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config["temperature"]
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI")
            
            # Parse JSON response
            data = self._parse_json_response(response.choices[0].message.content)
            
            if not data:
                raise ValueError("Failed to parse JSON from OpenAI response")
            
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
            
            self.logger.info("Successfully extracted resume data using OpenAI")
            return resume_data
        
        except Exception as e:
            self.logger.error(f"OpenAI extraction failed: {e}")
            return self._get_fallback_data(resume_text)
