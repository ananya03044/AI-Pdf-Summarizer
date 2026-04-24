import numpy as np
from .vector_store import search_index


def run_pipeline(
    chunks: list[str],
    embeddings: np.ndarray,
    index,
    model,
    agents: dict
) -> str:

    query_embedding = np.mean(embeddings, axis=0, keepdims=True)

    top_indices = search_index(index, query_embedding, 5)
    retrieved_chunks = [chunks[i] for i in top_indices]

    context = "\n".join(retrieved_chunks)

    analyzer = agents["analyzer"]
    summarizer = agents["summarizer"]
    editor = agents["editor"]

    analysis = analyzer(model, context)
    summary = summarizer(model, context)   # ✅ FIXED
    final_summary = editor(model, summary)

    return final_summary