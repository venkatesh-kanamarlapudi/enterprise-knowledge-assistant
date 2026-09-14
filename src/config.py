"""
Central configuration for the Enterprise Knowledge Assistant.

All tunable parameters are defined here (or read from environment variables)
so the rest of the codebase never hard-codes values.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
FAISS_DIR = INDEX_DIR / "faiss"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_STORE_PATH = INDEX_DIR / "bm25_store.pkl"

FAISS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))


@dataclass
class RetrievalConfig:
    # Number of candidates pulled from each retriever before fusion.
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", 10))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", 10))
    # Weight given to vector search vs. BM25 during Reciprocal Rank Fusion.
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", 0.5))
    # Number of fused candidates handed to the reranker.
    rerank_candidate_k: int = int(os.getenv("RERANK_CANDIDATE_K", 10))
    # Final number of chunks handed to the LLM after reranking.
    final_top_k: int = int(os.getenv("FINAL_TOP_K", 4))


@dataclass
class MemoryConfig:
    # Number of most recent conversation turns kept verbatim in the prompt.
    max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", 6))


@dataclass
class LLMConfig:
    provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")  # openai | gemini
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 800))


@dataclass
class EmbeddingConfig:
    # Local, free, sentence-transformers embedding model (no API key required).
    model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


@dataclass
class VectorStoreConfig:
    backend: str = os.getenv("DEFAULT_VECTOR_STORE", "faiss")  # faiss | chroma
    collection_name: str = os.getenv("CHROMA_COLLECTION", "employee_knowledge_base")


@dataclass
class RerankerConfig:
    # Local cross-encoder reranker (no API key required).
    model_name: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@dataclass
class AppConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")


CONFIG = AppConfig()

SYSTEM_PROMPT = """You are the Enterprise Knowledge Assistant for NimbusTech Solutions.
Answer the employee's question using ONLY the information contained in the
"Context" section below, together with the recent conversation history for
resolving references (e.g. "it", "that", "what about carry-forward").

Rules you must follow:
1. Ground every factual claim in the provided context. Do not use outside knowledge.
2. If the answer is not present in the context, clearly say:
   "I could not find this information in the available company documents."
   Do not guess or fabricate an answer.
3. Be concise and specific. Prefer bullet points for policies with multiple rules.
4. Do not mention "the context" or "the documents" explicitly in your answer;
   just answer naturally as an HR/IT knowledge assistant would.
5. Never invent a source or a document name that was not given to you.
"""
