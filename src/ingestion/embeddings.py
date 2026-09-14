"""
Embedding model factory.

Uses a local, free sentence-transformers model by default so the project
does not depend on any paid API for embeddings. This keeps ingestion fully
reproducible offline, regardless of which LLM provider is selected for
answer generation.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CONFIG


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=CONFIG.embedding.model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
