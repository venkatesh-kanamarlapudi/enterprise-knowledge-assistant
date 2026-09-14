# Enterprise Knowledge Assistant — Advanced RAG

A production-oriented Retrieval-Augmented Generation (RAG) application that
lets employees ask natural-language questions about company policy documents
(leave, IT, travel, benefits, code of conduct, HR handbook, FAQs) and get
grounded, cited answers through a Streamlit chat interface.

---

## 1. Problem Statement

Employees routinely need to look up scattered policy documents (PDF, Word,
Markdown, plain text) to answer simple questions like *"How many casual
leaves do I get?"* or *"What's the notice period during probation?"*. This is
slow, error-prone, and doesn't handle natural follow-up questions like
*"What about carry-forward?"*.

This project builds an **Employee Knowledge Assistant** that ingests a
heterogeneous collection of company documents, retrieves the most relevant
passages using a hybrid (semantic + keyword) search pipeline, reranks them
for precision, and generates a grounded answer with source citations —
explicitly refusing to answer when the information isn't in the documents.

## 2. Solution Overview

The assistant implements the full advanced-RAG flow requested:

| Requirement | Implementation |
|---|---|
| Document ingestion (2+ formats) | PDF, DOCX, Markdown, and TXT loaders (`src/ingestion/loaders.py`) |
| Chunking | `RecursiveCharacterTextSplitter` (`src/ingestion/chunking.py`) |
| Embeddings | Local, free `sentence-transformers/all-MiniLM-L6-v2` (`src/ingestion/embeddings.py`) |
| Vector store | **FAISS** and **ChromaDB**, switchable at runtime (`src/ingestion/vectorstore.py`) |
| Hybrid search | Vector search + BM25, fused with Reciprocal Rank Fusion (`src/retrieval/hybrid_search.py`) |
| Reranking | Local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) (`src/retrieval/reranker.py`) |
| Conversational memory | Follow-up question condensation + history window (`src/memory/conversation_memory.py`) |
| Source citations | Every answer lists the exact source document(s) used |
| Hallucination handling | System prompt + explicit "not found" fallback, sources hidden when unfound |
| UI | Streamlit chat app with history, reset, and source display (`app.py`) |
| LLM provider switch | OpenAI **and** Google Gemini, selectable from the sidebar (`src/llm/llm_provider.py`) |
| Vector DB switch | FAISS **and** ChromaDB, selectable from the sidebar |

## 3. Architecture

```
Documents (PDF/DOCX/MD/TXT)
        │
        ▼
  Document Loader
        │
        ▼
     Chunking
        │
        ▼
    Embeddings  ───────────────► BM25 Index (keyword)
        │
        ▼
  Vector Store (FAISS / ChromaDB)
        │
        │             ┌── Vector / Semantic Search ──┐
User Query ──(memory)──┤                              ├──► Hybrid Fusion (RRF)
        │              └── BM25 Keyword Search ───────┘            │
        │                                                          ▼
        │                                                     Reranking
        │                                                  (Cross-Encoder)
        │                                                          │
        │                                                          ▼
        │                                                  Relevant Context
        │                                                          │
        └───────────────────────► LLM (OpenAI / Gemini) ◄──────────┘
                                          │
                                          ▼
                          Grounded Answer + Source Citations
                                          │
                                          ▼
                                   Streamlit Chat UI
```

A rendered diagram is also available at [`docs/architecture.svg`](docs/architecture.svg).

### Why these design choices?

- **Reciprocal Rank Fusion (RRF)** combines the vector and BM25 result lists
  by rank rather than raw score, since cosine similarity and BM25 scores are
  on incomparable scales — RRF avoids fragile score normalization.
- **Cross-encoder reranking** is applied only to the small fused candidate
  set (not the whole corpus) because cross-encoders are far more accurate
  at judging query–passage relevance than bi-encoder embeddings, but too
  slow to run over an entire index.
- **Local embeddings + local reranker** (sentence-transformers, no API key)
  keep the ingestion and retrieval path fully free and reproducible offline,
  regardless of which paid LLM provider is chosen for the final answer.
- **Query condensation** rewrites follow-up questions (e.g. "What about
  carry-forward?") into standalone questions using the active LLM and
  recent history, so retrieval always receives a self-contained query.

## 4. Technology Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain
- **Vector Stores:** FAISS, ChromaDB
- **Keyword Search:** rank_bm25 (BM25Okapi)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Reranker:** sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- **LLMs:** OpenAI GPT (`gpt-4o-mini`) and Google Gemini (`gemini-1.5-flash`)
- **UI:** Streamlit
- **Document loaders:** pypdf, docx2txt, unstructured (Markdown), plain text

## 5. Project Structure

```
enterprise-knowledge-assistant/
├── app.py                          # Streamlit UI entry point
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── raw/                        # Sample company documents (PDF/DOCX/MD/TXT)
│   └── index/                      # Generated FAISS/Chroma/BM25 indexes (gitignored)
├── docs/
│   ├── architecture.svg
│   └── architecture.mmd
├── sample_outputs/
│   └── sample_conversation.md      # Example Q&A transcript with citations
├── scripts/
│   └── ingest.py                   # CLI: builds vector + BM25 indexes
└── src/
    ├── config.py                   # Central configuration (env-driven)
    ├── rag_pipeline.py              # End-to-end orchestration
    ├── ingestion/
    │   ├── loaders.py               # Multi-format document loading
    │   ├── chunking.py              # Recursive character chunking
    │   ├── embeddings.py            # Local embedding model factory
    │   └── vectorstore.py           # FAISS / Chroma build & load
    ├── retrieval/
    │   ├── keyword_search.py        # BM25 index & search
    │   ├── hybrid_search.py         # RRF fusion of vector + BM25
    │   └── reranker.py              # Cross-encoder reranking
    ├── memory/
    │   └── conversation_memory.py   # Follow-up question condensation
    └── llm/
        └── llm_provider.py          # OpenAI / Gemini provider factory
```

## 6. Setup Instructions

### 6.1 Prerequisites

- Python 3.10 or higher
- An OpenAI API key and/or a Google AI Studio (Gemini) API key
  (only the provider(s) you intend to use)

### 6.2 Installation

```bash
git clone <your-repo-url>
cd enterprise-knowledge-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 6.3 Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set at least one of:

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

All other variables (chunk size, retrieval `top_k`, model names, etc.) have
sensible defaults and do not need to be changed to run the app. See
`.env.example` for the full list of tunable parameters.

## 7. How to Run

### 7.1 Ingest the sample documents

This builds the FAISS index, the ChromaDB index, and the BM25 keyword index
from the documents in `data/raw/`:

```bash
python scripts/ingest.py --backend both
```

Run it once after cloning, and again any time you add/change documents in
`data/raw/`. Use `--backend faiss` or `--backend chroma` to build only one
vector store if you don't need both.

### 7.2 Launch the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). Use the
sidebar to switch between **OpenAI / Gemini** and **FAISS / ChromaDB** at any
time — the app reloads the corresponding index/model automatically.

## 8. Sample Data

`data/raw/` ships with 7 realistic sample policy documents for a fictional
company, **NimbusTech Solutions**, spanning 4 different formats so the
evaluator can run the app immediately without sourcing their own data:

| Document | Format |
|---|---|
| Leave_Policy.pdf | PDF |
| IT_Policy.pdf | PDF |
| Code_of_Conduct.pdf | PDF |
| Employee_Handbook.docx | DOCX |
| Travel_Policy.docx | DOCX |
| Benefits_Documentation.md | Markdown |
| Company_FAQs.txt | Plain text |

## 9. Sample Inputs & Outputs

See [`sample_outputs/sample_conversation.md`](sample_outputs/sample_conversation.md)
for a full example conversation (including a follow-up question that
exercises conversational memory, and a question deliberately outside the
documents' scope to demonstrate hallucination handling).

Example:

> **User:** What is the leave policy?
> **Assistant:** Employees can avail several types of leave: Earned Leave
> (1.5 days/month, 18 days/year)... *(grounded in Leave_Policy.pdf)*
>
> Sources: `Leave_Policy.pdf`
>
> **User:** What about carry-forward?
> **Assistant:** Up to 10 days of unused Earned Leave can be carried forward
> to the next calendar year... *(system understood "carry-forward" refers to
> the leave policy discussed above)*
>
> Sources: `Leave_Policy.pdf`

## 10. Key Design Decisions

- **Hybrid search over pure vector search:** policy documents contain many
  exact terms (e.g. "26 weeks", "POSH Act", specific INR amounts) that dense
  embeddings alone can under-retrieve; BM25 complements semantic search for
  these cases.
- **RRF fusion instead of score-weighted fusion:** avoids the need to
  normalize incomparable similarity/BM25 scales.
- **Reranking after fusion, not before:** cross-encoders are precise but
  expensive; running them only on ~10 fused candidates keeps latency low
  while still sharpening final relevance.
- **Local embeddings/reranker, pluggable LLM:** keeps the retrieval pipeline
  fully reproducible and free, while still letting users choose their
  preferred (paid) LLM for generation.
- **Explicit hallucination guard:** the system prompt instructs the model to
  say *"I could not find this information in the available company
  documents"* rather than guess, and the UI suppresses source citations for
  such answers so a "not found" response never appears falsely grounded.
- **Provider/backend switch resets conversation memory:** switching LLM or
  vector store mid-conversation starts a fresh session to avoid mixing
  contexts generated by different models/backends.

## 11. Limitations

- Local embedding/reranker models trade some accuracy for being free and
  offline; a hosted embedding API (e.g. OpenAI `text-embedding-3-large`)
  could improve retrieval quality further.
- BM25 index is rebuilt in full on every ingestion run (no incremental
  updates); acceptable at this document-collection scale.
- ChromaDB and FAISS indexes are stored on local disk (`data/index/`) and
  are not synced across machines — re-run `scripts/ingest.py` on a new
  machine/environment.
- The app does not implement user authentication or per-user document
  permissions; all users see all ingested documents.
- Conversation memory is in-process (Streamlit session state) and is not
  persisted across app restarts.

## 12. Demonstration

A sample conversation transcript is provided in
`sample_outputs/sample_conversation.md`. For a live/video demonstration,
record a short walkthrough showing: document ingestion, a basic Q&A, a
follow-up question exercising memory, a source-citation check, switching
LLM providers, switching vector store backends, and an out-of-scope
question triggering the "not found" response.
