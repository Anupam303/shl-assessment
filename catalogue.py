import json

def load_catalogue():
    """
    Loads the assessment catalogue from the JSON file.
    Returns an empty list if the file is not found.
    """
    try:
        with open('data/shl_catalogue.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def create_assessment_text(item):
    """
    Merges all relevant fields from an assessment item into a single string.
    This combined text is used to generate vector embeddings for semantic search.
    """
    return (
        item.get("name", "") + " " +
        item.get("description", "") + " " +
        " ".join(item.get("keys", [])) + " " +
        " ".join(item.get("job_levels", []))
    )