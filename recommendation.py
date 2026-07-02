def is_word_in_text(word, text):
    word_len = len(word)
    text_len = len(text)
    idx = 0
    while True:
        idx = text.find(word, idx)
        if idx == -1:
            return False
        if idx > 0 and text[idx - 1].isalnum():
            idx += 1
            continue
        if idx + word_len < text_len and text[idx + word_len].isalnum():
            idx += 1
            continue
        return True

def recommend(query, catalogue):

    query = query.lower()

    scores = []
    for item in catalogue:
         
         score = 0

         text = (
            item.get("name", "") + " " +
            item.get("description", "") + " " +
            " ".join(item.get("keys", []))
         ).lower()

         for word in query.split():
             if is_word_in_text(word, text):
                 score += 1
         
         if score > 0:        
            scores.append((score, item))

    scores.sort(key=lambda x: x[0], reverse=True)

    return [assessment[1] for assessment in scores[:10]]

def build_reason(item):

    return (
        f"Recommended because it evaluates "
        f"{', '.join(item.get('keys', []))} "
        f"and is suitable for "
        f"{', '.join(item.get('job_levels', []))}."
    )


    
