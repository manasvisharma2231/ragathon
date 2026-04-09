import os
import google.generativeai as genai
from dotenv import load_dotenv # You might need to pip install python-dotenv

load_dotenv() # This looks for a .env file in your folder

# Get the key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment!")
else:
    genai.configure(api_key=api_key)
    # Now try listing the models
    for m in genai.list_models():
        print(m.name)