import json

def load_catalogue():
    try:
        with open('data/shl_catalogue.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def create_assessment_text(item):

    return (
        item.get("name", "") + " " +
        item.get("description", "") + " " +
        " ".join(item.get("keys", [])) + " " +
        " ".join(item.get("job_levels", []))
    )