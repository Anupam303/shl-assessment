import re
import numpy as np

from catalogue import create_assessment_text
from embeddings import get_model
from recommendation import build_reason

def contains_word(text, word):
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))

def cosine_similarity(a,b):

    return np.dot(a,b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

from collections import Counter

def should_filter_item(query_text, query_words, item_text, item_words):
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

    query_lower = query.lower()
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    # 1. Compute document frequency for all words in the catalogue dynamically
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
    rare_query_words = []
    for qw in query_words:
        if len(qw) >= 2:
            df = doc_freqs.get(qw, 0)
            if 0 < df < 0.15 * doc_count:
                rare_query_words.append(qw)

    query_embedding = get_model().encode(query)

    scores = []

    for item, assessment_embedding in zip(catalogue, assessment_embeddings):

        # Get item text for language verification filtering
        item_text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            " ".join(item.get("keys", []))
        ).lower()
        
        # Enforce that if the query contains specific terms, the item must match at least one of them
        if rare_query_words:
            if not any(qw in item_text for qw in rare_query_words):
                continue
        
        item_words = re.findall(r'\b\w+\b', item_text)
        
        # Generic check to prevent partial/substring matches (e.g. java vs javascript, c vs c++)
        if should_filter_item(query_lower, query_words, item_text, item_words):
            continue

        score = cosine_similarity(query_embedding,assessment_embedding)



        scores.append((
            score,
            item
        ))

    scores.sort(key=lambda x: x[0], reverse=True)


    results = []

    for score, item in scores[:10]:
        keys = [k.lower() for k in item.get("keys", [])]
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

    