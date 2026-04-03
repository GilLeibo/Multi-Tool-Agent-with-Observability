from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Tool API keys
    openweathermap_api_key: str = ""
    tavily_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://ollama:11434"

    # Database
    database_url: str = "sqlite:////app/data/agent.db"

    # Agent
    max_agent_iterations: int = 10

    # App
    log_level: str = "INFO"
    app_version: str = "1.0.0"


settings = Settings()
