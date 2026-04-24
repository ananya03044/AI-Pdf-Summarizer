import os
from dotenv import load_dotenv

from .pdf_loader import load_pdf
from .chunker import chunk_text
from .embeddings import embed_texts
from .vector_store import build_faiss_index
from .agents import (
    get_model,
    analyzer_agent,
    summarizer_agent,
    editor_agent,
)
from .pipeline import run_pipeline


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    print("📄 Loading PDF...")
    text = load_pdf("data/the-state-of-fashion-2025-v2.pdf")

    print("✂️ Chunking...")
    chunks = chunk_text(text)

    print("🔢 Creating embeddings...")
    embeddings = embed_texts(chunks, api_key)

    print("📦 Building vector store...")
    index = build_faiss_index(embeddings)

    print("🤖 Loading Gemini model...")
    model = get_model(api_key)

    agents = {
        "analyzer": analyzer_agent,
        "summarizer": summarizer_agent,
        "editor": editor_agent,
    }

    print("🧠 Running RAG pipeline...")
    summary = run_pipeline(
        chunks=chunks,
        embeddings=embeddings,
        index=index,
        model=model,
        agents=agents,
    )

    os.makedirs("output", exist_ok=True)
    with open("output/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("✅ DONE! Summary saved to output/summary.txt")


if __name__ == "__main__":
    main()




