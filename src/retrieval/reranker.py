"""
Reranking stage.

Re-scores the fused (vector + BM25) candidates with a cross-encoder that
looks at the (query, chunk) pair jointly. Cross-encoders are more accurate
than the bi-encoder used for initial retrieval because they attend across
the full query-document pair rather than comparing independent embeddings,
at the cost of being too slow to run over the entire corpus. This is why it
runs only on the small candidate set produced by hybrid search.

Uses a local, free `sentence-transformers` CrossEncoder so no paid API is
required for this step, independent of which LLM provider is selected.
"""

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import CONFIG


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(CONFIG.reranker.model_name)


def rerank(query: str, candidates: List[Document]) -> List[Document]:
    """
    Re-score `candidates` against `query` with a cross-encoder and return
    the top `CONFIG.retrieval.final_top_k` documents, best first.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_k = CONFIG.retrieval.final_top_k
    return [doc for doc, _score in scored[:top_k]]
