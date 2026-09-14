"""
Document loading utilities.

Supports loading a heterogeneous folder of company documents in multiple
formats (.pdf, .docx, .md, .txt) into a unified list of LangChain Document
objects, each tagged with source metadata for later citation.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt"}

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}


def _load_single_file(path: Path) -> List[Document]:
    """Load a single file into one or more Document objects."""
    ext = path.suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file extension: {ext} ({path.name})")

    if ext == ".txt":
        loader = loader_cls(str(path), encoding="utf-8")
    else:
        loader = loader_cls(str(path))

    docs = loader.load()

    # Normalize metadata so downstream code can always rely on `source`
    # containing just the filename (used for citations in the UI).
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = ext.lstrip(".")
    return docs


def load_documents(data_dir: Path) -> List[Document]:
    """
    Load every supported document from `data_dir`.

    Returns a flat list of LangChain Document objects, each carrying
    `metadata["source"]` (the original filename) for citation purposes.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    all_docs: List[Document] = []
    skipped: List[str] = []

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(path.name)
            continue
        try:
            all_docs.extend(_load_single_file(path))
        except Exception as exc:  # noqa: BLE001 - surfaced to caller/log
            print(f"[loaders] Failed to load {path.name}: {exc}")

    if skipped:
        print(f"[loaders] Skipped unsupported files: {skipped}")

    if not all_docs:
        raise RuntimeError(
            f"No documents could be loaded from {data_dir}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return all_docs
