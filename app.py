"""
Streamlit UI for the Enterprise Knowledge Assistant.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.config import CONFIG
from src.ingestion.vectorstore import vector_store_exists
from src.llm.llm_provider import MissingAPIKeyError
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="\U0001F4DA",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sidebar: provider / backend selection
# --------------------------------------------------------------------------
st.sidebar.title("\u2699\ufe0f Configuration")

llm_provider = st.sidebar.radio(
    "LLM Provider",
    options=["openai", "gemini"],
    format_func=lambda p: {"openai": "OpenAI (GPT)", "gemini": "Google Gemini"}[p],
    index=0 if CONFIG.llm.provider == "openai" else 1,
    help="Choose which LLM generates the final answer. Requires the "
    "corresponding API key to be set in your .env file.",
)

vector_backend = st.sidebar.radio(
    "Vector Store",
    options=["faiss", "chroma"],
    format_func=lambda b: {"faiss": "FAISS (local file index)", "chroma": "ChromaDB (local persistent DB)"}[b],
    index=0 if CONFIG.vector_store.backend == "faiss" else 1,
    help="Choose which vector database backend serves semantic search.",
)

st.sidebar.divider()

if st.sidebar.button("\U0001F5D1\ufe0f Clear conversation", use_container_width=True):
    if "pipeline" in st.session_state:
        st.session_state.pipeline.reset_memory()
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Documents are indexed offline via `scripts/ingest.py`. "
    "This app only performs retrieval + answer generation."
)

# --------------------------------------------------------------------------
# Pipeline initialization (cached per provider/backend combination)
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, sources}

pipeline_key = f"{vector_backend}::{llm_provider}"

if not vector_store_exists(vector_backend):
    st.error(
        f"No {vector_backend.upper()} index found. Please run "
        f"`python scripts/ingest.py --backend {vector_backend}` first, "
        "then reload this app."
    )
    st.stop()

if st.session_state.get("pipeline_key") != pipeline_key:
    try:
        with st.spinner(f"Loading {vector_backend.upper()} index and {llm_provider} model..."):
            st.session_state.pipeline = RAGPipeline(
                vector_backend=vector_backend, llm_provider=llm_provider
            )
        st.session_state.pipeline_key = pipeline_key
        # Switching provider/backend starts a fresh conversation for clarity.
        st.session_state.chat_history = []
    except MissingAPIKeyError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize the knowledge assistant: {exc}")
        st.stop()

pipeline: RAGPipeline = st.session_state.pipeline

# --------------------------------------------------------------------------
# Main chat interface
# --------------------------------------------------------------------------
st.title("\U0001F4DA Enterprise Knowledge Assistant")
st.caption(
    "Ask questions about company policies \u2014 leave, IT, travel, benefits, "
    "code of conduct, and more. Answers are grounded in your company documents."
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("\U0001F4C4 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

user_question = st.chat_input("Ask about leave, IT, travel, benefits, code of conduct...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            try:
                response = pipeline.ask(user_question)
                st.markdown(response.answer)
                if response.sources:
                    with st.expander("\U0001F4C4 Sources"):
                        for source in response.sources:
                            st.markdown(f"- `{source}`")
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                    }
                )
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "Something went wrong while answering your question. "
                    f"Details: {exc}"
                )
