from sentence_transformers import SentenceTransformer

from catalogue import create_assessment_text

model = SentenceTransformer(
    "model/all-MiniLM-L6-v2"
)

def build_assessment_embeddings(catalogue):

    embeddings = []

    for item in catalogue:
        text = create_assessment_text(item)

        embedding = model.encode(text)

        embeddings.append(embedding)

    return embeddings