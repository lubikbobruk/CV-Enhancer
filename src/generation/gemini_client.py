"""Thin wrapper around Gemini via langchain-google-genai."""

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, MODEL_NAME


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


def test_call(prompt: str) -> str:
    """Test to verify the API is working."""
    response = get_llm().invoke(prompt)
    return response.content
