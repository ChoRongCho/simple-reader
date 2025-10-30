import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Literal

# .env 로드
load_dotenv()

Provider = Literal["openai", "ollama"]

@dataclass
class EnvConfig:
    provider: Provider
    openai_model: str
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]
    ollama_model: Optional[str]
    ollama_base_url: Optional[str]
    temperature: float
    max_tokens: int

def load_env() -> EnvConfig:
    return EnvConfig(
        provider=os.getenv("PROVIDER", "openai"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
    )

if __name__ == "__main__":
    cfg = load_env()
    print(cfg)
