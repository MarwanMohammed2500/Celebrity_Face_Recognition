from comparator import cosine_similarity

def match_face(query_embedding, database, threshold=0.3):
    best_match = None
    best_score = -1

    for name, embedding in database.items():
        for emb in embedding:
            sim = (cosine_similarity(query_embedding, emb))
            if sim > best_score:
                best_score = sim
                best_match = name if sim >= threshold else None

    return best_match, best_score