import fitz 
import docx
import json
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from .models import StudentProfile

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def parse_resume(file_bytes: bytes, filename: str) -> StudentProfile:
    text = ""
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")

    prompt = f"""
Given the following resume text, autonomously extract the student's profile into a strict JSON format.
If a value is not directly stated, estimate it conservatively based on the content (e.g., if no explicit open source mentioned, then 0).
Output ONLY valid JSON matching this schema exactly:
{{
    "cgpa": float (0.0 to 10.0),
    "tech_stack": [list of strings],
    "num_projects": int,
    "num_internships": int,
    "communication_score": int (1 to 5, assume 4 if the resume is well-formatted and clear),
    "open_source_score": int (0 to 2)
}}

Resume Text:
{text}
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        json_str = resp.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        return StudentProfile(**data)
    except Exception as e:
        print(f"Error parsing resume: {e}")
        return StudentProfile()