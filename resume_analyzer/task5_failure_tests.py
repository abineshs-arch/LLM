"""Part 3 - Task 2: test 5 failure cases (plus a fake success) with FastAPI's TestClient.
The LLM is replaced by quick fake functions here, so the tests take seconds instead of minutes."""
import time
from pathlib import Path

from fastapi.testclient import TestClient

import app as api
import config
from extractor import ResumeData
from matcher import JOBS, MatchResult

BASE = Path(__file__).parent
client = TestClient(api.app)
JD = JOBS["python_developer"]
RESUME_01 = (BASE / "resumes" / "resume_01.pdf").read_bytes()
USAGE = {"input_tokens": 400, "output_tokens": 100}


def fake_extract(text, label):
    return {"success": True, "attempts": 1, "usage": USAGE,
            "data": ResumeData(candidate_name="Ravi Kumar", skills=["Python", "Django"], years_of_experience=4)}


def fake_match(resume, jd, label):
    return {"success": True, "attempts": 1, "usage": USAGE,
            "data": MatchResult(score=78, matched_skills=["Python", "Django"], missing_skills=["Git"],
                                reasoning="Has Python and Django with 4 years of backend work.")}


def fake_bad_json(text, label):
    return {"success": False, "attempts": 3, "usage": USAGE, "error": "json: Invalid JSON: EOF while parsing"}


def fake_slow(text, label):
    time.sleep(3)
    return fake_extract(text, label)


def post(name, data, **kw):
    files = {"file": (name, data, "application/octet-stream")} if name else None
    form = {"job_description": kw.get("jd", JD)}
    if kw.get("text"):
        form["resume_text"] = kw["text"]
    return client.post("/analyze", files=files, data=form)


def main():
    rows = []

    def record(case, response):
        body = response.json()
        message = body.get("detail") if "detail" in body else f"score {body.get('score')}"
        rows.append((case, response.status_code, str(message)))
        print(f"{case:42} -> {response.status_code} | {message} | X-Request-ID {response.headers.get('X-Request-ID')}")

    api.extract_with_retry, api.match_jd = fake_extract, fake_match
    record("0. Normal request (fake LLM)", post("resume_01.pdf", RESUME_01))
    record("1. Wrong file type (.docx)", post("resume.docx", b"fake word file"))
    record("2. File too large (6 MB)", post("big.txt", b"a" * 6 * 1024 * 1024))
    long_text = "Python developer with Django and SQL experience. " * 400
    record("3. Resume over the token limit", post(None, None, text=long_text))
    record("4. Scanned PDF (no text)", post("scanned.pdf", (BASE / "resumes" / "scanned.pdf").read_bytes()))

    api.extract_with_retry = fake_bad_json
    record("5. LLM returns invalid JSON 3 times", post("resume_01.pdf", RESUME_01))

    api.extract_with_retry = fake_slow
    config.LLM_TIMEOUT_SECONDS = 1
    record("6. LLM timeout (1 s limit, retried once)", post("resume_01.pdf", RESUME_01))

    print("\n| Failure case | Status code | Message |\n|---|---|---|")
    for case, code, msg in rows:
        print(f"| {case} | {code} | {msg} |")
    print(f"\nAPI key loaded from .env: {bool(config.API_KEY)} (the key itself is never printed)")


if __name__ == "__main__":
    main()