"""Shared Gemini configuration for RAG, MCP tool use, and connection checks.

Select the answer/tool-calling model with GEMINI_MODEL in the project's .env.
Embedding and reranker models have separate settings and do not use this file.
"""

from pathlib import Path

from dotenv import dotenv_values
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_gemini_model(
    *, max_tokens: int = 4096, timeout: float = 60
) -> ChatGoogleGenerativeAI:
    """Read fresh project settings on each call; never log the API key.

Creating this object does not generate an answer or consume a Gemini request.
There is no default model: a missing setting produces a clear configuration
error instead of silently choosing an unintended model.
"""
    settings = dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    model_name = (settings.get("GEMINI_MODEL") or "").strip()
    api_key = (settings.get("GOOGLE_API_KEY") or "").strip()
    if not model_name:
        raise ValueError("Add GEMINI_MODEL to the project's .env file.")
    if not api_key or api_key in {"paste_your_key_here", "your_api_key_here"}:
        raise ValueError("Add your GOOGLE_API_KEY to the project's .env file.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        vertexai=False,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=0,
    )
