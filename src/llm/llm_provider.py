"""
LLM provider factory.

Lets the application switch between OpenAI and Google Gemini chat models at
runtime (e.g. via a Streamlit sidebar selector) behind a single interface,
so the RAG pipeline code never needs to know which provider is active.
"""

from typing import Literal

from src.config import CONFIG

Provider = Literal["openai", "gemini"]


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider's API key is not configured."""


def get_llm(provider: Provider):
    """
    Return a LangChain chat model instance for the requested provider.

    Raises MissingAPIKeyError with a user-friendly message if the relevant
    API key is not set, so the Streamlit UI can surface a clean error
    instead of a raw stack trace.
    """
    if provider == "openai":
        if not CONFIG.openai_api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Add it to your .env file to use OpenAI."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=CONFIG.llm.openai_model,
            temperature=CONFIG.llm.temperature,
            max_tokens=CONFIG.llm.max_tokens,
            api_key=CONFIG.openai_api_key,
        )

    if provider == "gemini":
        if not CONFIG.google_api_key:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is not set. Add it to your .env file to use Gemini."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=CONFIG.llm.gemini_model,
            temperature=CONFIG.llm.temperature,
            max_output_tokens=CONFIG.llm.max_tokens,
            google_api_key=CONFIG.google_api_key,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
