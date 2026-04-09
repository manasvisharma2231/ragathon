from pydantic import BaseModel, Field
from typing import List, Optional

class StudentProfile(BaseModel):
    # Mapping to: Academic_Score, DSA_Skill, Project_Quality, Experience_Score, OpenSource_Value, Soft_Skills, Tech_Stack_Score
    academic_score: float = Field(default=0.0, alias="cgpa")
    dsa_skill: float = Field(default=0.0)
    project_quality: float = Field(default=0.0)
    experience_score: float = Field(default=0.0)
    opensource_value: float = Field(default=0.0)
    soft_skills: float = Field(default=0.0)
    tech_stack_score: float = Field(default=0.0)
    
    # Internal text list for RAG search
    tech_stack: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True

class ExperienceMatch(BaseModel):
    student_name: str
    company: str
    cgpa: float
    tech_stack: List[str]
    experience_text: str
    outcome: str
    similarity_score: float