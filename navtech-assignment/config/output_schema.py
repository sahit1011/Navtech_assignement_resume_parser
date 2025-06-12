"""
Output schema definitions for resume parsing
Defines the structure of extracted resume data
"""

from typing import List, Optional
from pydantic import BaseModel


class Address(BaseModel):
    """Address information"""
    street: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    country: str = ""


class Skill(BaseModel):
    """Skill information"""
    name: str
    proficiency: str = ""


class Education(BaseModel):
    """Education information"""
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    graduation_year: str = ""
    gpa: str = ""


class WorkExperience(BaseModel):
    """Work experience information"""
    company: str = ""
    position: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""


class ResumeData(BaseModel):
    """Complete resume data structure"""
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    address: Address = Address()
    summary: str = ""
    skills: List[Skill] = []
    education_history: List[Education] = []
    work_history: List[WorkExperience] = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "address": {
                "street": self.address.street,
                "city": self.address.city,
                "state": self.address.state,
                "zip_code": self.address.zip_code,
                "country": self.address.country
            },
            "summary": self.summary,
            "skills": [{"name": skill.name, "proficiency": skill.proficiency} for skill in self.skills],
            "education_history": [
                {
                    "institution": edu.institution,
                    "degree": edu.degree,
                    "field_of_study": edu.field_of_study,
                    "graduation_year": edu.graduation_year,
                    "gpa": edu.gpa
                } for edu in self.education_history
            ],
            "work_history": [
                {
                    "company": work.company,
                    "position": work.position,
                    "start_date": work.start_date,
                    "end_date": work.end_date,
                    "description": work.description
                } for work in self.work_history
            ]
        }
