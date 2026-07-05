import os
import numpy as np
from sentence_transformers import SentenceTransformer

from catalogue import create_assessment_text

model = None

def get_model():
    """
    Loads the SentenceTransformer model from the local directory.
    Initializes the model only once (lazy loading) to save memory and time.
    """
    global model
    if model is None:
        model = SentenceTransformer(
            "model/all-MiniLM-L6-v2"
        )
    return model

def build_assessment_embeddings(catalogue):
    """
    Generates vector embeddings for all assessments in the catalogue.
    - Loads pre-computed embeddings from 'data/assessment_embeddings.npy' if available (fast).
    - Otherwise, computes them using the SentenceTransformer model and saves them for future use.
    """
    npy_path = "data/assessment_embeddings.npy"
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)

    m = get_model()
    embeddings = []

    for item in catalogue:
        text = create_assessment_text(item)

        embedding = m.encode(text)

        embeddings.append(embedding)

    return embeddings