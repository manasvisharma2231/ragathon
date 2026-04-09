Lucknow Foodie Guide: IIITL Edition:
Welcome to the Lucknow Foodie Guide, an intelligent, context-aware recommendation engine designed specifically for the students of IIIT Lucknow. This bot leverages RAG (Retrieval-Augmented Generation) to help you navigate the local food scene, from the historic lanes of Chowk to the modern cafes in Phoenix Palassio.

System Architecture:
The project implements a modular RAG pipeline to ensure high-precision recommendations based on a curated local dataset:

1. Context Loading: A structured restaurants.json file containing details for 30+ local eateries.
2. Chunking: The data is processed using a Recursive Character Text Splitter ($Chunk Size: 500, Overlap: 50$) to maintain the integrity of restaurant profiles.
3. Embeddings: Text chunks are transformed into numerical vectors using OpenAI Embeddings (text-embedding-3-small).
4. Vector Store: Vectors are indexed and stored using FAISS (Facebook AI Similarity Search) for rapid semantic retrieval.5. 5.5. Generation: When a user queries the bot, the system retrieves the most relevant restaurant data and uses an LLM to synthesize a helpful, "vibe-checked" response.

Setup Instructions->

1. Prerequisites
Python 3.9 or higher
An active OpenAI API Key

2. Environment Configuration
Create a .env file in the Statement-2-Lucknow-Foodie-Guide/ directory. 

3. Installation:
Navigate to the project directory and install the necessary dependencies:
cd Statement-2-Lucknow-Foodie-Guide
pip install langchain openai faiss-cpu python-dotenv

4. Running the Project:
Step 1: Data Ingestion
Run the ingestion script to process the restaurants.json file and create the FAISS vector database.

Step 2: Run the main script directly using python src/main.py

Usage Examples->

1. Vibe-based: "Where can I find a chill cafe to study near IIITL?"

2. Budget-focused: "Suggest a budget-friendly Biryani spot in Gomti Nagar."

3. Dish-specific: "Who serves the best Basket Chaat around here?"

4. Dietary: "Find me the best non-veg kebabs in the city."

FOLDER STRUCTURE ->

RAGATHON/
└── Statement-2-Lucknow-Foodie-Guide/
    ├── dataset/
    │   ├── .gitkeep
    │   └── restaurants.json
    ├── src/
    │   ├── .gitkeep
    │   ├── ingest.py
    │   └── main.py
    ├── static/
    │   ├── images/
    │   ├── app.js
    │   ├── index.html
    │   └── styles.css
    ├── .env.example
    ├── .gitkeep
    └── run.sh