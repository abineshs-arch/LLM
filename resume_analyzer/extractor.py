"""Resume extraction: schema with constraints + validated JSON with retry (uses local_llm.py)."""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from local_llm import MODEL, call_json

BASE = Path(__file__).parent
PROMPT_VERSION = "extract_v1"
SYSTEM_PROMPT = (BASE / "prompts" / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
PREFILL = '{\n  "candidate_name":'


class ResumeData(BaseModel):
    """The extraction schema with constraints."""
    candidate_name: Optional[str] = None
    skills: list[str] = Field(min_length=1)                                  # must not be empty
    years_of_experience: Optional[float] = Field(default=None, ge=0, le=50)  # between 0 and 50
    education: list[str] = []
    strengths: list[str] = []
    gaps: list[str] = []


class ExtractionError(Exception):
    pass


def extract_with_retry(resume_text, label="extract"):
    """Returns {"success": True, "data": ResumeData, "attempts": n} or a failure object."""
    return call_json(SYSTEM_PROMPT, f"Resume:\n{resume_text}", ResumeData, PREFILL, label)


def extract(resume_text, label="extract"):
    """Same as Part 1 (used by task2_extract.py and app.py): returns (data, None) or raises."""
    result = extract_with_retry(resume_text, label)
    if not result["success"]:
        raise ExtractionError(result["message"])
    return result["data"], None