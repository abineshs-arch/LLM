"""Part 4 - Task 1: pytest tests for the API with a mocked LLM (no real model is loaded)."""
import pytest
from fastapi.testclient import TestClient

import app as api
import local_llm
from matcher import JOBS

JD = JOBS["python_developer"]
GOOD_RESUME = (b"Ravi Kumar\nPython Developer, Chennai\nBackend developer with 4 years of experience.\n"
               b"Skills: Python, Django, FastAPI, PostgreSQL, Git\nEducation: B.E. Computer Science, 2021")
EXTRACT_JSON = (' "Ravi Kumar", "skills": ["Python", "Django", "FastAPI", "PostgreSQL", "Git"], '
                '"years_of_experience": 4, "education": ["B.E. Computer Science, 2021"], '
                '"strengths": ["4 years of backend work"], "gaps": []}')
MATCH_JSON = (' 80, "matched_skills": ["Python", "Django", "PostgreSQL", "Git"], "missing_skills": ["REST APIs"], '
              '"reasoning": "Has the main backend skills and 4 years of experience."}')


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace local_llm.chat. The list gives the extraction answers, one per attempt."""
    calls = []

    def install(extract_answers):
        answers = list(extract_answers)

        def chat(messages, prefill="", max_new_tokens=450):
            calls.append(prefill)
            if "score" in prefill:                                   # matching step
                return prefill + MATCH_JSON, 300, 60, 0.01
            return prefill + answers.pop(0), 250, 80, 0.01          # extraction step
        monkeypatch.setattr(local_llm, "chat", chat)
        return calls
    return install


def post(client, name, content, jd=JD):
    return client.post("/analyze", data={"job_description": jd},
                       files={"file": (name, content, "text/plain")})


def test_health_does_not_leak_api_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "api_key_loaded" in r.json()
    if api.config.API_KEY:
        assert api.config.API_KEY not in r.text


def test_success(client, fake_llm):
    calls = fake_llm([EXTRACT_JSON])
    r = post(client, "resume.txt", GOOD_RESUME)
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_name"] == "Ravi Kumar"
    assert body["score"] == 80
    assert body["usage"]["input_tokens"] == 250 + 300
    assert "X-Request-ID" in r.headers
    assert len(calls) == 2                                          # 1 extraction + 1 matching


def test_invalid_json_then_success(client, fake_llm):
    calls = fake_llm([' "Ravi Kumar", "skills": ["Python"',          # attempt 1: broken JSON
                      EXTRACT_JSON])                                  # attempt 2: valid
    r = post(client, "resume.txt", GOOD_RESUME)
    assert r.status_code == 200
    assert r.json()["score"] == 80
    assert len(calls) == 3                                          # 2 extraction attempts + 1 matching


def test_invalid_json_three_times_returns_502(client, fake_llm):
    fake_llm(['oops'] * 3)
    r = post(client, "resume.txt", GOOD_RESUME)
    assert r.status_code == 502
    assert "valid JSON after 3 attempts" in r.json()["detail"]


def test_empty_file_returns_422(client, fake_llm):
    calls = fake_llm([EXTRACT_JSON])
    r = post(client, "resume.pdf", b"")
    assert r.status_code == 422
    assert r.json()["detail"] == "the uploaded file is empty"
    assert calls == []                                              # LLM never called


def test_wrong_type_returns_415(client, fake_llm):
    calls = fake_llm([EXTRACT_JSON])
    r = post(client, "resume.docx", b"not a real word file")
    assert r.status_code == 415
    assert calls == []


def test_too_many_tokens_returns_413(client, fake_llm):
    fake_llm([EXTRACT_JSON])
    r = post(client, "resume.txt", b"Python developer with Django experience. " * 400)
    assert r.status_code == 413