def cosine_similarity(a, b):
    a = a.squeeze()
    b = np.array(b).squeeze()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))