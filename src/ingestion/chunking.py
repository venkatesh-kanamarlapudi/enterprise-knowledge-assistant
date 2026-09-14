"""
Chunking utilities: splits loaded Documents into retrieval-sized chunks
while preserving and enriching source metadata for citations.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CONFIG


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks suitable for embedding.

    Each resulting chunk keeps the original `source` metadata and gets a
    stable `chunk_id` so it can be referenced consistently across the
    vector store, BM25 index, and reranker.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunking.chunk_size,
        chunk_overlap=CONFIG.chunking.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    per_source_counter = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = per_source_counter.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}::chunk_{idx}"
        per_source_counter[source] = idx + 1

    return chunks
