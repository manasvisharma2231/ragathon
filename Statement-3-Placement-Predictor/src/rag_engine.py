import json
import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .models import StudentProfile, ExperienceMatch

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "./data/faiss_index.bin"
METADATA_PATH = "./data/senior_experiences_metadata.json"

class ExperienceRAG:
    def __init__(self):
        self._encoder = None  # Lazy load for speed
        self.chunk_metadata = []
        self._load_metadata()
        
        self.index = None
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)

    @property
    def encoder(self):
        if self._encoder is None:
            print(f"Loading SentenceTransformer ({MODEL_NAME})...")
            self._encoder = SentenceTransformer(MODEL_NAME)
        return self._encoder

    def _load_metadata(self):
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)

    def get_top_matches(self, profile: StudentProfile, k: int = 3) -> List[ExperienceMatch]:
        if not self.index:
            return []
            
        stack_text = ", ".join(profile.tech_stack)
        query = f"Candidate with skills in {stack_text} and CGPA {profile.cgpa}. Look for interview reports."
        
        query_vector = self.encoder.encode([query])
        
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        matches = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunk_metadata):
                text_chunk = self.chunk_metadata[idx]
                
                dist = float(distances[0][i])
                sim_score = max(0.0, min(100.0, 100.0 - (dist * 10)))
                
                # Mock high-level metadata as we are chunking raw text now
                matches.append(ExperienceMatch(
                    student_name="Verified Senior",
                    company="Matched Company",
                    cgpa=0.0,
                    tech_stack=[],
                    experience_text=text_chunk,
                    outcome="Successful",
                    similarity_score=round(sim_score, 2)
                ))
                
        return matches

rag_engine = ExperienceRAG()