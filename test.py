from catalogue import load_catalogue
from semantic_recommender import semantic_recommend
from embeddings import build_assessment_embeddings

catalogue = load_catalogue()
embeddings = build_assessment_embeddings(catalogue)

results = semantic_recommend(
    "Java Developer",
    catalogue,
    embeddings
)

for item in results[:10]:
    print(item["name"], " - ", item["reason"])