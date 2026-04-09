import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from .models import StudentProfile

MODEL_PATH = "./data/model.pkl"

# --- GLOBAL MODEL CACHE ---
# This loads the model into RAM when the server starts
if os.path.exists(MODEL_PATH):
    print("🚀 [PlacementIQ] Loading pre-trained model into memory...")
    GLOBAL_MODEL = joblib.load(MODEL_PATH)
else:
    print("⚠️ [PlacementIQ] No model found at /data/model.pkl. Using heuristic fallback.")
    GLOBAL_MODEL = None

def train_model_from_csv(csv_path):
    """
    Utility function to train and save the model. 
    You should only run this once or via a separate script.
    """
    df = pd.read_csv(csv_path)
    
    target_col = "Readiness_Score"
    X = pd.DataFrame()
    X['Academic_Score'] = pd.to_numeric(df['Academic_Score'], errors='coerce').fillna(7.0)
    X['DSA_Skill'] = pd.to_numeric(df['DSA_Skill'], errors='coerce').fillna(5)
    X['Project_Quality'] = pd.to_numeric(df['Project_Quality'], errors='coerce').fillna(5)
    X['Experience_Score'] = pd.to_numeric(df['Experience_Score'], errors='coerce').fillna(0)
    X['OpenSource_Value'] = pd.to_numeric(df['OpenSource_Value'], errors='coerce').fillna(0)
    X['Soft_Skills'] = pd.to_numeric(df['Soft_Skills'], errors='coerce').fillna(5)
    X['Tech_Stack_Score'] = pd.to_numeric(df['Tech_Stack_Score'], errors='coerce').fillna(5)
    
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(50)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}.")

def predict_score(profile: StudentProfile) -> float:
    """
    Uses the pre-loaded global model to predict the score instantly.
    """
    global GLOBAL_MODEL
    
    # If model isn't loaded, use the math-based fallback
    if GLOBAL_MODEL is None:
        # Simple weighted average for safety
        score = (profile.academic_score * 8) + (profile.dsa_skill * 2)
        return min(max(round(score, 2), 0), 100)
        
    # Prepare the data exactly how the model expects it
    X_input = pd.DataFrame([{
        "Academic_Score": profile.academic_score,
        "DSA_Skill": profile.dsa_skill,
        "Project_Quality": profile.project_quality,
        "Experience_Score": profile.experience_score,
        "OpenSource_Value": profile.opensource_value,
        "Soft_Skills": profile.soft_skills,
        "Tech_Stack_Score": profile.tech_stack_score
    }])
    
    # Prediction happens in milliseconds because GLOBAL_MODEL is already in RAM
    predicted_score = GLOBAL_MODEL.predict(X_input)
    return min(max(round(predicted_score, 2), 0), 100)