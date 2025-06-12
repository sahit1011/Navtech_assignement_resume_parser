"""
OpenRouter LLM provider for resume parsing
Provides access to multiple models through OpenRouter API
"""

import os
import json
import logging
import requests
from typing import Dict, Any

from .base_llm import BaseLLMProvider
from config.output_schema import ResumeData, Address, Skill, Education, WorkExperience


class OpenRouterLLMProvider(BaseLLMProvider):
    """OpenRouter LLM provider with multiple model support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "temperature": 0.1,
            "max_tokens": 6000
        }
        super().__init__(config or default_config)
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        return bool(self.api_key)
    
    def extract_resume_data(self, resume_text: str, pdf_path: str = None) -> ResumeData:
        """Extract resume data using OpenRouter"""
        if not self.is_available():
            raise ValueError("OpenRouter API key not available. Please set OPENROUTER_API_KEY environment variable or provide a custom API key.")
        
        try:
            prompt = self._create_resume_prompt(resume_text)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/navtech-assignment",
                "X-Title": "NavTech Resume Parser"
            }
            
            payload = {
                "model": self.config["model"],
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that extracts structured data from resumes. Return only valid JSON without any additional text or formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"]
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise ValueError(f"OpenRouter API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            if not response_data.get('choices') or not response_data['choices'][0].get('message', {}).get('content'):
                raise ValueError("Empty response from OpenRouter")
            
            content = response_data['choices'][0]['message']['content']
            self.logger.info("Received response from OpenRouter")
            self.logger.debug(f"Full response content length: {len(content)}")
            self.logger.debug(f"Full response content: {content}")

            # Parse JSON response
            data = self._parse_json_response(content)
            
            if not data:
                raise ValueError("Failed to parse JSON from OpenRouter response")
            
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
            
            self.logger.info("Successfully extracted resume data using OpenRouter")
            return resume_data
        
        except Exception as e:
            self.logger.error(f"OpenRouter extraction failed: {e}")
            raise ValueError(f"OpenRouter API error: {str(e)}")
