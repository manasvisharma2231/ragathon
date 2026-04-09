import os
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import json
import faiss
from sentence_transformers import SentenceTransformer
import joblib
from src.regression_engine import train_model_from_csv
from src.rag_engine import rag_engine

def build_rag_index():
    print("--- Building RAG Index from PDF ---")
    pdf_path = "./data/INTERVIEW EXPERIENCES.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    # 1. Extract and Chunk Text
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    # Split by double newline or chunks of 1000 chars
    chunks = [c.strip() for c in full_text.split("\n\n") if len(c.strip()) > 50]
    
    if not chunks:
        # Fallback to character chunks if no double newlines
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]

    print(f"Extracted {len(chunks)} relevant experience chunks.")

    # 2. Embed and Build FAISS
    # Using lazy encoder
    embeddings = rag_engine.encoder.encode(chunks)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, "./data/faiss_index.bin")
    
    # Save the chunk metadata
    with open("./data/senior_experiences_metadata.json", "w") as f:
        json.dump(chunks, f)
        
    print("✅ RAG Index and Metadata built.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # 1. Train Regression Model from user CSV
    csv_path = "./data/normalized_placement_data.csv"
    if os.path.exists(csv_path):
        print("--- Training Regression Model from CSV ---")
        train_model_from_csv(csv_path)
    else:
        print(f"Warning: {csv_path} not found. Skipping regression training.")

    # 2. Build RAG Store from user PDF
    build_rag_index()
    
    print("\n🚀 All real-data models optimized and ready.")