import hashlib
import numpy as np

def compute_embedding(text: str, dim: int = 768) -> list:
    """Zero-disk, deterministic 768-dimension hashing vector."""
    tokens = text.lower().split()
    vector = np.zeros(dim, dtype=np.float32)
    
    for token in tokens:
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        index = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vector[index] += sign
        
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    else:
        vector[0] = 1.0
    return vector.tolist()
