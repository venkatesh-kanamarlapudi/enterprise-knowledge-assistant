"""
Vector store abstraction supporting both FAISS and ChromaDB as
interchangeable local backends. The rest of the application talks to this
module through `build_vector_store` / `load_vector_store` only, so the
retrieval layer never needs to know which backend is active.
"""

from pathlib import Path
from typing import List, Literal

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from src.config import CONFIG, FAISS_DIR, CHROMA_DIR
from src.ingestion.embeddings import get_embedding_model

VectorBackend = Literal["faiss", "chroma"]


def _faiss_path(persist_dir: Path) -> Path:
    return persist_dir / "index.faiss"


def build_vector_store(documents: List[Document], backend: VectorBackend) -> None:
    """Build and persist a fresh vector store from the given chunks."""
    embeddings = get_embedding_model()

    if backend == "faiss":
        store = FAISS.from_documents(documents, embeddings)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(FAISS_DIR))
    elif backend == "chroma":
        # Wipe any existing collection so re-ingestion doesn't duplicate chunks.
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=CONFIG.vector_store.collection_name,
            persist_directory=str(CHROMA_DIR),
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")


def load_vector_store(backend: VectorBackend):
    """Load a previously built vector store for querying."""
    embeddings = get_embedding_model()

    if backend == "faiss":
        index_file = _faiss_path(FAISS_DIR)
        if not index_file.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python scripts/ingest.py --backend faiss` first."
            )
        return FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )

    if backend == "chroma":
        if not any(CHROMA_DIR.iterdir()) if CHROMA_DIR.exists() else True:
            raise FileNotFoundError(
                "Chroma index not found. Run `python scripts/ingest.py --backend chroma` first."
            )
        return Chroma(
            collection_name=CONFIG.vector_store.collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    raise ValueError(f"Unknown vector store backend: {backend}")


def vector_store_exists(backend: VectorBackend) -> bool:
    """Check whether a persisted index already exists for the given backend."""
    try:
        if backend == "faiss":
            return _faiss_path(FAISS_DIR).exists()
        if backend == "chroma":
            return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
    except Exception:  # noqa: BLE001
        return False
    return False
