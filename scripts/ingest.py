"""
Ingestion pipeline entry point.

Usage:
    python scripts/ingest.py --backend faiss
    python scripts/ingest.py --backend chroma
    python scripts/ingest.py --backend both

This script performs the full ingestion flow:
    Documents -> Loader -> Chunking -> Embeddings -> Vector Store + BM25 index
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.ingestion.loaders import load_documents
from src.ingestion.chunking import chunk_documents
from src.ingestion.vectorstore import build_vector_store
from src.retrieval.keyword_search import build_bm25_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest company documents into the knowledge base.")
    parser.add_argument(
        "--backend",
        choices=["faiss", "chroma", "both"],
        default="both",
        help="Which vector store backend(s) to build (default: both).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Folder containing source documents (default: data/raw).",
    )
    args = parser.parse_args()

    start = time.time()

    print(f"[ingest] Loading documents from {args.data_dir} ...")
    documents = load_documents(Path(args.data_dir))
    print(f"[ingest] Loaded {len(documents)} raw document page(s)/section(s).")

    print("[ingest] Chunking documents ...")
    chunks = chunk_documents(documents)
    print(f"[ingest] Produced {len(chunks)} chunks.")

    backends = ["faiss", "chroma"] if args.backend == "both" else [args.backend]
    for backend in backends:
        print(f"[ingest] Building {backend.upper()} vector store (embedding all chunks)...")
        build_vector_store(chunks, backend=backend)
        print(f"[ingest] {backend.upper()} vector store built and persisted.")

    print("[ingest] Building BM25 keyword index ...")
    build_bm25_index(chunks)
    print("[ingest] BM25 index built and persisted.")

    elapsed = time.time() - start
    print(f"[ingest] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
