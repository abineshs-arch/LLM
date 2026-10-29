"""Compare extracted resume data with a job description and score it."""
import json
from pathlib import Path

from pydantic import BaseModel, Field

from local_llm import call_json

BASE = Path(__file__).parent
PROMPT_VERSION = "match_v1"
SYSTEM_PROMPT = (BASE / "prompts" / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
PREFILL = '{\n  "score":'

JOBS = {
    "python_developer": """Job: Python Developer (Backend)
Required skills: Python, Django or FastAPI, SQL / PostgreSQL, REST APIs, Git
Nice to have: Docker, AWS, CI/CD
Experience: 3+ years of backend development""",
    "data_analyst": """Job: Data Analyst
Required skills: SQL, Excel, Power BI or Tableau, Python, data visualisation
Nice to have: statistics, business reporting, stakeholder communication
Experience: 2+ years in data analysis""",
}


class MatchResult(BaseModel):
    score: int = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str = Field(min_length=10)


def match_jd(resume: dict, job_description: str, label="match"):
    """Score a resume against any job description text. The name is removed first (fairness)."""
    profile = {k: v for k, v in resume.items() if k != "candidate_name"}
    user = f"JOB DESCRIPTION:\n{job_description}\n\nCANDIDATE PROFILE (JSON):\n{json.dumps(profile, indent=2)}"
    return call_json(SYSTEM_PROMPT, user, MatchResult, PREFILL, label, max_new_tokens=300)


def match(resume: dict, job_id: str, label="match"):
    """Used by task4_match.py: score against one of the 2 saved jobs."""
    return match_jd(resume, JOBS[job_id], label)