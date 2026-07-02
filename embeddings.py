import os
import numpy as np
from sentence_transformers import SentenceTransformer

from catalogue import create_assessment_text

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer(
            "model/all-MiniLM-L6-v2"
        )
    return model

def build_assessment_embeddings(catalogue):
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