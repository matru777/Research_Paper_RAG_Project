import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Settings:
    """
    Centralized configuration management.
    All secrets are validated and stored here.
    """
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    def validate(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("CRITICAL: OPENAI_API_KEY is missing from .env")
        if not self.QDRANT_URL:
            raise ValueError("CRITICAL: QDRANT_URL is missing from .env")

settings = Settings()

def get_fast_llm():
    """Returns the fast model for routing and simple grading."""
    return ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0, 
        api_key=settings.OPENAI_API_KEY
    )
def get_strong_llm():
    """Returns the strong model for answer generation."""
    return ChatOpenAI(
        model="gpt-4o", 
        temperature=0, 
        api_key=settings.OPENAI_API_KEY,
        streaming=True
    )