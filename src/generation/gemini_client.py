"""Thin wrapper around Gemini via langchain-google-genai."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY, MODEL_NAME
from src.generation.prompts import build_enhance_prompt
from src.generation.schemas import EnhancementResult


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2)


def test_call(prompt: str) -> str:
    """Test to verify the API is working."""
    response = get_llm().invoke(prompt)
    return response.content


def enhance_chunks(
    full_cv: str,
    job_ad: str,
    selected: list[tuple[int, str]]) -> EnhancementResult:
    """Rewrite the selected (chunk_id, text) chunks toward the job ad."""
    if not selected:
        return EnhancementResult(rewrites=[])

    system, user = build_enhance_prompt(full_cv, job_ad, selected)
    # force gemini to return pydantic 
    structured = get_llm().with_structured_output(EnhancementResult)
    return structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
