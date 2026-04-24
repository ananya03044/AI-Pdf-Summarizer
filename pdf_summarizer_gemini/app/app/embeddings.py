from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer()

def embed_texts(texts: list[str], api_key: str = None) -> np.ndarray:
    """
    Convert text into embeddings using TF-IDF (no external API, no torch).
    """
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()