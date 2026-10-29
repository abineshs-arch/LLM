# AI Resume Analyzer

Upload a resume (PDF or text) and a job description. The API extracts the candidate's details as structured JSON,
scores the fit from 0 to 100 with matched and missing skills and a short reasoning, and reports token usage and
estimated cost. **The score is a sorting aid for a recruiter - a human must review every decision.**

All 8 sample resumes are **dummy data** (fictional people) created by `make_resume.py`.

## Architecture
```
                upload (PDF / TXT) + job description
                               |
                         POST /analyze  (app.py)
                               |
  validate: type (415), size (413), empty / scanned (422), token limit (413)
                               |
          reader.py  - PyMuPDF text extraction + cleaning (headers, page numbers, spaces)
                               |
          extractor.py - prompt extract_v2 + JSON mode + Pydantic ResumeData
                               |        (retry up to 3x with the validation error)
          matcher.py  - prompt match_v1 + Pydantic MatchResult (name removed before scoring)
                               |
          local_llm.py - Qwen2.5-1.5B-Instruct on CPU, temperature 0, timeout, logging with request id
                               |
     JSON: candidate_name, score, matched_skills, missing_skills, reasoning, resume, usage + cost
```

| File | Purpose |
|---|---|
| `app.py` | FastAPI: `GET /health`, `POST /parse`, `POST /analyze` |
| `config.py` | Settings from `.env` (API key never returned or logged) |
| `reader.py` | `read_resume(path)` - PDF/TXT reading, cleaning, validation |
| `extractor.py`, `prompts/extract_v2.txt` | Resume -> structured JSON (`ResumeData`) |
| `matcher.py`, `prompts/match_v1.txt` | Resume JSON + job description -> score (`MatchResult`) |
| `local_llm.py` | Local model, JSON retry with feedback, token logging |
| `test_api.py` | 7 pytest tests with a mocked LLM |
| `task6_evaluate.py` | Golden-set evaluation (precision / recall / accuracy, score consistency) |
| `truth.json` | Golden set: correct name, skills, years and education for the 8 resumes |

## Setup
```powershell
cd C:\LLM
python -m pip install -r resume_analyzer/requirements.txt
copy .env.example .env          # then edit .env (never commit it - it is in .gitignore)
python resume_analyzer/make_resume.py     # creates the dummy resumes and truth.json
```
`.env.example`:
```
GEMINI_API_KEY=your-api-key-here
MAX_FILE_MB=5
MAX_RESUME_TOKENS=1500
LLM_TIMEOUT_SECONDS=600
PRICE_IN_PER_M=0.75
PRICE_OUT_PER_M=3.75
```
The model (`Qwen/Qwen2.5-1.5B-Instruct`, about 3 GB) downloads automatically on the first request.

## Run
```powershell
python -m uvicorn app:app --app-dir resume_analyzer
# open http://127.0.0.1:8000/docs  (POST /analyze has a ready example job description)
```

### Example curl (Windows: use curl.exe)
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@resume_analyzer/resumes/resume_01.pdf" \
  -F "job_description=Job: Python Developer (Backend). Required skills: Python, Django or FastAPI, SQL / PostgreSQL, REST APIs, Git. Experience: 3+ years"
```
Response (real run, about 10 minutes on a laptop CPU):
```json
{"request_id": "2ff3544e", "candidate_name": "Ravi Kumar", "score": 85,
 "matched_skills": ["Python", "Django", "FastAPI", "SQL / PostgreSQL", "Docker", "AWS"],
 "missing_skills": [], "reasoning": "The candidate has all the required skills ...",
 "usage": {"input_tokens": 884, "output_tokens": 222, "estimated_cost_usd": 0.001495},
 "note": "Automated score - must be reviewed by a human before any decision."}
```

### Errors
| Case | Status |
|---|---|
| Wrong file type | 415 |
| File > 5 MB or resume > 1500 tokens | 413 |
| Empty file, scanned PDF, too-short text | 422 |
| Model returns invalid JSON 3 times / crashes | 502 |
| Model timeout or busy | 503 (+ `Retry-After`) |

## Tests and evaluation
```powershell
python -m pytest resume_analyzer -v --cov=resume_analyzer --cov-config=resume_analyzer/.coveragerc --cov-report=term-missing
python resume_analyzer/task6_evaluate.py
```
- **7 tests passed** (mocked LLM): success, invalid JSON then success, 3x invalid JSON -> 502, empty file -> 422,
  wrong type -> 415, token limit -> 413, `/health` does not leak the key. Coverage 67% (core modules 62-100%).

| Prompt | Skill precision | Skill recall | Field accuracy (name, years, education) |
|---|---|---|---|
| extract_v1 | 1.00 | 0.94 | 0.96 |
| **extract_v2** (fix: copy every skill incl. the last one, years rules, gap examples) | **1.00** | **0.98** | 0.96 |

- Biggest weakness of v1: the **last skill in a list was dropped** (Git, HRMS). v2 fixed it for 2 of 3 resumes.
- Still wrong: resume_01 misses Git; the fresher's 6-month internship is still `null` instead of 0.5.
- **Score consistency:** 16 resume/job pairs scored twice at temperature 0 -> **average difference 0.0 points**.

## Limitations
- A small 1.5B model on CPU: ~2-5 minutes per model call, about 10 minutes per `/analyze` request.
- Scores are **consistent but not always correct**: a data analyst ranked #1 for the Python Developer job,
  and some matched skills were invented (HR and Sales candidates "matched" Python and Django).
- Scanned PDFs need OCR (not supported); only English resumes were tested.
- Retry-with-feedback can push the model to invent data to pass validation (seen on an empty resume),
  so empty or unreadable files are rejected before the model is called.
- The cost shown is an estimate for Gemini 3.8 Flash prices; the local model itself is free.

## Fairness note
1. The model can invent or miss skills, so a strong candidate can be ranked low for no real reason.
2. LLMs learn from biased data and may favour certain names, colleges, genders, career gaps or writing styles;
   the name is removed before scoring, but bias can hide in other fields.
3. The score looks precise but is not calibrated - use it only to **sort for review, never to reject** automatically.
4. A human recruiter checks every shortlist against the original resume, reviews rejected candidates too,
   sees the reasoning and missing skills, and keeps an audit log; candidates are told AI is used and can ask for a human review.

## Demo Q&A
**Can it be wrong?** Yes. It sometimes misses skills (Git), mis-reads experience (fresher = null) and can invent
matched skills; the ranking for the Python Developer job put a data analyst first. That is why every response says
a human must review it, the reasoning and missing skills are shown, and we measure precision/recall on a golden set
before changing prompts.

**How do you stop bias?** We remove the candidate's name before scoring, tell the model to judge only skills and
experience and ignore gender, age, location and college, score against a fixed job description with a written
scoring guide, keep temperature 0 so the same resume always gets the same score, log every request, and never
auto-reject - a human reviews both shortlisted and rejected candidates. Next step: test with pairs of resumes that
differ only in name, gender or college and check that the scores stay the same.