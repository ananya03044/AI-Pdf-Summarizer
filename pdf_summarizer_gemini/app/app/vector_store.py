import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS index from embeddings.
    Returns the index ready for similarity search.
    """
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def search_index(index, query_embedding: np.ndarray, top_k: int = 5):
    """
    Searches the FAISS index and returns indices of top_k similar vectors.
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]
