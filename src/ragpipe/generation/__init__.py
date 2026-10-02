"""Generation layer: prompts + LLM clients + response caching."""

from .llm import CachedLLM, ExtractiveClient, LLMClient, OpenAIClient, build_llm
from .prompts import CONTEXT_TEMPLATE, SYSTEM_PROMPT, PromptBuilder

__all__ = [
    "CONTEXT_TEMPLATE",
    "SYSTEM_PROMPT",
    "CachedLLM",
    "ExtractiveClient",
    "LLMClient",
    "OpenAIClient",
    "PromptBuilder",
    "build_llm",
]
