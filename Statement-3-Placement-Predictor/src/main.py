from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import os, json

from .models import StudentProfile, ExperienceMatch
from .profiler import ChatProfiler
from .resume_parser import parse_resume
from .regression_engine import predict_score
from .rag_engine import rag_engine

app = FastAPI(title="PlacementIQ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# For the hackathon, we use a global session. 
# Added a helper to ensure it's always initialized.
profiler_session = ChatProfiler()

class ChatRequest(BaseModel):
    message: str

class AnalysisResponse(BaseModel):
    profile: StudentProfile
    score: float
    top_experiences: list[ExperienceMatch]

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = "frontend/index.html"
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>index.html not found in frontend/</h1>", status_code=404)
    with open(index_path, "r") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def event_generator() -> AsyncGenerator[str, None]:
        # Handle the stream from the profiler
        async for chunk in profiler_session.astream_chat(req.message):
            yield chunk
        
        # After the stream finishes, check if we should signal the frontend to stop
        if profiler_session.is_complete():
            # We yield a specific separator so the frontend can split the text from the signal
            yield "\n[DONE]" 

    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/analyze-chat", response_model=AnalysisResponse)
async def analyze_chat():
    # 1. Extract JSON profile from the chat history
    profile = profiler_session.extract_profile()
    
    # 2. Get the ML prediction (Now fast because model is pre-loaded)
    try:
        score = predict_score(profile)
    except Exception as e:
        print(f"Regression Error: {e}")
        score = 0.0
        
    # 3. Get RAG matches from your PDF
    try:
        matches = rag_engine.get_top_matches(profile)
    except Exception as e:
        print(f"RAG Error: {e}")
        matches = []

    return AnalysisResponse(profile=profile, score=score, top_experiences=matches)

@app.post("/upload-resume", response_model=AnalysisResponse)
async def upload_resume(file: UploadFile = File(...)):
    contents = await file.read()
    # Resume parsing also uses Gemini, so it's subject to the same quota!
    profile = parse_resume(contents, file.filename)
    try:
        score = predict_score(profile)
    except Exception:
        score = 0.0
    matches = rag_engine.get_top_matches(profile)
    return AnalysisResponse(profile=profile, score=score, top_experiences=matches)

@app.post("/reset")
async def reset_session():
    global profiler_session
    profiler_session = ChatProfiler()
    return {"status": "reset"}