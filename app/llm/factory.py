from app.llm.base import LLMClient


def get_llm_client(provider: str, model: str | None = None) -> LLMClient:
    """Instantiate the appropriate LLM client for the given provider."""
    p = provider.lower().strip()

    if p == "anthropic":
        from app.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model or "claude-sonnet-4-6")

    elif p == "openai":
        from app.llm.openai_client import OpenAIClient
        return OpenAIClient(model=model or "gpt-4.1-mini")

    elif p == "gemini":
        from app.llm.gemini_client import GeminiClient
        return GeminiClient(model=model or "gemini-2.0-flash")

    elif p == "ollama":
        from app.llm.ollama_client import OllamaClient
        return OllamaClient(model=model or "llama3.2:3b")

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Choose from: anthropic, openai, gemini, ollama")
