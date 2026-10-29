"""Settings loaded from the .env file (C:\\LLM\\.env). The API key is never returned or logged."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")   # the .env in the LLM folder (in .gitignore)

API_KEY = os.getenv("GEMINI_API_KEY")                 # loaded, but never printed, logged or returned
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "5"))
MAX_RESUME_TOKENS = int(os.getenv("MAX_RESUME_TOKENS", "1500"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))   # local CPU model is slow
ALLOWED_TYPES = {".pdf", ".txt"}
# Estimated cost if the same tokens were sent to a hosted model (Gemini 3.8 Flash, USD per 1M tokens)
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0.75"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "3.75"))


def public_settings():
    """Safe to show in /health: says IF a key is loaded, never the key itself."""
    return {"api_key_loaded": bool(API_KEY), "max_file_mb": MAX_FILE_MB,
            "max_resume_tokens": MAX_RESUME_TOKENS, "llm_timeout_seconds": LLM_TIMEOUT_SECONDS}


def estimate_cost(input_tokens, output_tokens):
    return round(input_tokens / 1_000_000 * PRICE_IN_PER_M + output_tokens / 1_000_000 * PRICE_OUT_PER_M, 6)