import re
import numpy as np

from embeddings import get_model
from recommendation import build_reason

def contains_word(text, word):
    """
    Checks if a specific word exists as a standalone word in the text.
    Uses regex word boundaries (\b) so that 'java' matches 'Java' but not 'javascript'.
    """
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))

def cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors (lists of numbers).
    Formula: (a . b) / (||a|| * ||b||)
    Returns a score between -1 and 1 representing how conceptually similar they are.
    """
    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

from collections import Counter

def should_filter_item(query_text, query_words, item_text, item_words):
    """
    Guardrail function to prevent partial match false positives.
    Example: Prevents 'C' matching 'C++' or 'Java' matching 'JavaScript'.
    """
    # Rule 1: A query word is a partial/substring match of an item word,
    # but the query word does not exist as a standalone word in the item.
    for qw in query_words:
        if len(qw) < 2:
            continue
        for iw in item_words:
            if qw in iw and qw != iw:
                if not contains_word(item_text, qw):
                    return True

    # Rule 2: An item word is a partial/substring match of a query word,
    # but the item word does not exist as a standalone word in the query.
    for iw in item_words:
        if len(iw) < 2:
            continue
        for qw in query_words:
            if iw in qw and iw != qw:
                if not contains_word(query_text, iw):
                    return True
                    
    return False

def semantic_recommend(query, catalogue, assessment_embeddings):
    """
    Searches the assessment catalogue using a combination of:
    1. Rare-keyword filtering (boosting specific technologies).
    2. Substring guardrails (preventing false matches).
    3. Dense semantic search (comparing embeddings via Cosine Similarity).
    """
    query_lower = query.lower()
    # Tokenize the query into words (ignoring punctuation)
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    # 1. Compute document frequency for all words in the catalogue dynamically
    # This tells us how common each word is across all assessments.
    doc_count = len(catalogue)
    doc_freqs = Counter()
    for item in catalogue:
        item_text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            " ".join(item.get("keys", []))
        ).lower()
        words = set(re.findall(r'\b\w+\b', item_text))
        for w in words:
            doc_freqs[w] += 1

    # 2. Identify rare/specific terms in the query (appearing in < 15% of the catalogue)
    # E.g., if a user specifically mentions a rare tech like 'c++', we want to pay close attention to it.
    rare_query_words = []
    for qw in query_words:
        if len(qw) >= 2:
            df = doc_freqs.get(qw, 0)
            if 0 < df < 0.15 * doc_count:
                rare_query_words.append(qw)

    # Convert the user query into a vector embedding
    query_embedding = get_model().encode(query)

    scores = []

    # Iterate over the catalogue and pre-computed embeddings
    for item, assessment_embedding in zip(catalogue, assessment_embeddings):

        # Merge fields to form the item text
        item_text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            " ".join(item.get("keys", []))
        ).lower()
        
        # Enforce that if the query contains rare/specific terms, the item must match at least one of them.
        # This acts as a hard filter to keep search results highly relevant.
        if rare_query_words:
            if not any(qw in item_text for qw in rare_query_words):
                continue
        
        item_words = re.findall(r'\b\w+\b', item_text)
        
        # Prevent partial/substring matches (e.g. java vs javascript, c vs c++)
        if should_filter_item(query_lower, query_words, item_text, item_words):
            continue

        # Calculate semantic closeness using cosine similarity
        score = cosine_similarity(query_embedding, assessment_embedding)

        scores.append((
            score,
            item
        ))

    # Sort results in descending order of similarity score (highest match first)
    scores.sort(key=lambda x: x[0], reverse=True)

    results = []

    # Format the top 10 matches for the response
    for score, item in scores[:10]:
        keys = [k.lower() for k in item.get("keys", [])]
        
        # Determine if the test is a Personality ("P") or Knowledge ("K") test based on keywords
        is_personality = any("personality" in k or "behavior" in k or "opq" in k for k in keys)
        test_type = "P" if is_personality else "K"

        results.append({
            "name": item["name"],
            "url": item["link"],
            "duration": item["duration"],
            "remote": item["remote"],
            "adaptive": item["adaptive"],
            "reason": build_reason(item),
            "test_type": test_type,
        })
    return results

    