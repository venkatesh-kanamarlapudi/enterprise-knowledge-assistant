"""
Hybrid search: combines dense vector/semantic search with sparse BM25
keyword search using Reciprocal Rank Fusion (RRF).

RRF is used (rather than a raw weighted-score sum) because vector similarity
scores and BM25 scores live on different, incomparable scales. RRF only
relies on each retriever's *rank order*, which makes fusion robust without
needing score normalization.
"""

from typing import List

from langchain_core.documents import Document

from src.config import CONFIG
from src.retrieval.keyword_search import bm25_search

RRF_K = 60  # standard smoothing constant used in the RRF formula


def _chunk_key(doc: Document) -> str:
    """Stable identity for a chunk, used for de-duplication during fusion."""
    return doc.metadata.get("chunk_id") or (
        doc.metadata.get("source", "unknown") + "::" + doc.page_content[:50]
    )


def hybrid_retrieve(query: str, vector_store) -> List[Document]:
    """
    Run vector search and BM25 search in parallel, then fuse the two ranked
    lists with weighted Reciprocal Rank Fusion.

    Returns a de-duplicated, fused list of candidate chunks ordered from
    most to least relevant, ready to be handed to the reranker.
    """
    retrieval_cfg = CONFIG.retrieval

    vector_hits = vector_store.similarity_search(query, k=retrieval_cfg.vector_top_k)
    bm25_hits = bm25_search(query, top_k=retrieval_cfg.bm25_top_k)
    bm25_docs = [doc for doc, _score in bm25_hits]

    alpha = retrieval_cfg.hybrid_alpha  # weight on vector search, (1-alpha) on BM25

    fused_scores = {}
    doc_lookup = {}

    for rank, doc in enumerate(vector_hits):
        key = _chunk_key(doc)
        doc_lookup[key] = doc
        fused_scores[key] = fused_scores.get(key, 0.0) + alpha * (1.0 / (RRF_K + rank + 1))

    for rank, doc in enumerate(bm25_docs):
        key = _chunk_key(doc)
        doc_lookup[key] = doc
        fused_scores[key] = fused_scores.get(key, 0.0) + (1 - alpha) * (
            1.0 / (RRF_K + rank + 1)
        )

    ranked_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)
    fused_docs = [doc_lookup[k] for k in ranked_keys[: retrieval_cfg.rerank_candidate_k]]

    return fused_docs
