import os

from dotenv import load_dotenv


load_dotenv()


def groq_api_key_configured() -> bool:
    """Return whether a usable Groq key is available without exposing its value."""
    return bool(get_groq_api_key())


def get_groq_api_key() -> str:
    """Read the Groq key from the environment without exposing it to the UI."""
    key = os.getenv("GROQ_API_KEY", "").strip()

    return key


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-120b"
