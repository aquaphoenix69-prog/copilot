from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "copilotpass"
    neo4j_database: str = "neo4j"

    # LLM provider — any OpenAI-compatible /v1/chat/completions server.
    # Tested with: vLLM (default in design), Groq, Ollama, TGI, llama.cpp server.
    # Switch providers by editing these three values in .env — code does not change.
    llm_provider: str = "vllm"  # informational only: "vllm" | "groq" | "ollama" | "openai"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_api_key: str = "EMPTY"

    interpreter_model: str = "microsoft/deberta-v3-base"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    stt_model: str = "nvidia/parakeet-tdt-0.6b-v2"

    dataset_path: str = "data/final_master_dataset_complete_final.json"

    device: str = "cuda"
    log_level: str = "INFO"


settings = Settings()
