import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_career_paths():
    return pd.read_csv("data/career_paths.csv").to_dict(orient="records")

def embed_text(text):
    return embedder.encode([text])[0]

def extract_user_profile(user_data):
    return " ".join(user_data.get(k, "") for k in ["interests", "skills", "personality_traits", "career_goals"])

def match_career(user_vector, career_data):
    vectors = [embed_text(" ".join(path["careers"])) for path in career_data]
    similarities = cosine_similarity([user_vector], vectors)[0]
    best_idx = similarities.argmax()
    return career_data[best_idx], similarities[best_idx]
