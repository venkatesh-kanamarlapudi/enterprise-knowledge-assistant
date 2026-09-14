"""
End-to-end RAG pipeline orchestration.

Flow:
  User Query
    -> (memory) condense into a standalone question
    -> Hybrid Search (vector + BM25, fused with RRF)
    -> Reranking (cross-encoder)
    -> Prompt assembly (system prompt + history + context)
    -> LLM answer generation
    -> Source citation extraction
    -> memory update
"""

from dataclasses import dataclass, field
from typing import List

from src.config import CONFIG, SYSTEM_PROMPT
from src.ingestion.vectorstore import load_vector_store, VectorBackend
from src.retrieval.hybrid_search import hybrid_retrieve
from src.retrieval.reranker import rerank
from src.memory.conversation_memory import ConversationMemory
from src.llm.llm_provider import get_llm, Provider

ANSWER_PROMPT_TEMPLATE = """{system_prompt}

Conversation history (for resolving references only):
{history}

Context:
{context}

Question: {question}

Answer:"""

NOT_FOUND_MARKER = "I could not find this information in the available company documents."


@dataclass
class RAGResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    grounded: bool = True


def _format_context(chunks) -> str:
    if not chunks:
        return "(no relevant context was found)"
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, vector_backend: VectorBackend, llm_provider: Provider):
        self.vector_backend = vector_backend
        self.llm_provider = llm_provider
        self.vector_store = load_vector_store(vector_backend)
        self.llm = get_llm(llm_provider)
        self.memory = ConversationMemory()

    def reset_memory(self) -> None:
        self.memory.clear()

    def ask(self, question: str) -> RAGResponse:
        # 1. Resolve conversational references into a standalone question.
        standalone_question = self.memory.condense_question(question, self.llm)

        # 2. Hybrid retrieval (vector + BM25 fused via RRF).
        candidates = hybrid_retrieve(standalone_question, self.vector_store)

        # 3. Reranking with a cross-encoder to sharpen relevance.
        top_chunks = rerank(standalone_question, candidates)

        # 4. Build the final prompt and call the LLM.
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            history=self.memory.as_history_text(),
            context=_format_context(top_chunks),
            question=question,
        )
        response = self.llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response)).strip()

        # 5. Determine grounding / sources.
        grounded = NOT_FOUND_MARKER.lower() not in answer_text.lower()
        sources = sorted({doc.metadata.get("source", "unknown") for doc in top_chunks}) if grounded else []

        # 6. Update conversation memory with this turn.
        self.memory.add_turn(question, answer_text)

        return RAGResponse(answer=answer_text, sources=sources, grounded=grounded)
