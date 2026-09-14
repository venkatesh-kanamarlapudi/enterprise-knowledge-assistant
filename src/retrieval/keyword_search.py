"""
BM25 keyword search.

Maintains its own lightweight persisted index (a pickled list of chunk
Documents plus a fitted BM25Okapi object) so that keyword search works
independently of whichever vector store backend (FAISS/Chroma) is active.
"""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import BM25_STORE_PATH

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


@dataclass
class BM25Store:
    documents: List[Document]
    bm25: BM25Okapi


def build_bm25_index(documents: List[Document]) -> None:
    """Fit a BM25 index over the chunk corpus and persist it to disk."""
    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    BM25_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump({"documents": documents, "bm25": bm25}, f)


def _load_bm25_store() -> BM25Store:
    if not BM25_STORE_PATH.exists():
        raise FileNotFoundError(
            "BM25 index not found. Run `python scripts/ingest.py` first."
        )
    with open(BM25_STORE_PATH, "rb") as f:
        raw = pickle.load(f)
    return BM25Store(documents=raw["documents"], bm25=raw["bm25"])


def bm25_search(query: str, top_k: int) -> List[Tuple[Document, float]]:
    """Return the top_k (Document, bm25_score) pairs for the query."""
    store = _load_bm25_store()
    tokenized_query = _tokenize(query)
    scores = store.bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    return [(store.documents[i], float(scores[i])) for i in ranked_indices]
