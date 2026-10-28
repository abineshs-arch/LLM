# AI Resume Analyzer (Weekly Project - Part 1)

FastAPI app that reads a resume (PDF or text), cleans it and extracts structured data with an LLM.
Branch: `feature/resume-analyzer`. All 8 resumes are **dummy data** created by `make_resumes.py` (fictional people).

| File | Purpose |
|---|---|
| `make_resumes.py` | Creates 8 dummy resumes (5 PDF + 3 TXT), a scanned PDF, an empty PDF and `truth.json` |
| `reader.py` | `read_resume(path)` - PyMuPDF for PDF, plain read for text, cleaning, validation |
| `task1_report.py` | Characters and tokens per resume |
| `prompts/extract_v1.txt` | Versioned system prompt |
| `extractor.py` | Schema (Pydantic) + extraction with JSON mode and temperature 0 |
| `task2_extract.py` | Runs all 8 resumes, saves `outputs/extract_v1/*.json`, counts errors for 3 resumes |
| `app.py` | FastAPI: `GET /health`, `POST /parse`, `POST /extract` |

## Task 1 - Resume input handling
- PDF: PyMuPDF `page.get_text()`; text: normal file read
- Cleaning: extra spaces, repeated header/footer lines (a line on every page, e.g. "CONFIDENTIAL - CANDIDATE PROFILE"),
  page numbers ("Page 1 of 2"), double blank lines
- Validation: less than 50 characters -> error; if the PDF has images but no text -> "scanned file - needs OCR";
  unsupported file types and empty uploads are rejected (HTTP 422)

| File | Result |
|---|---|
| resume_01.pdf | 539 characters, 143 tokens |
| resume_02.pdf | 496 characters, 123 tokens |
| resume_03.pdf | 436 characters, 118 tokens |
| resume_04.pdf | 598 characters, 159 tokens |
| resume_05.pdf | 473 characters, 124 tokens |
| resume_06.txt | 368 characters, 86 tokens |
| resume_07.txt | 288 characters, 62 tokens |
| resume_08.txt | 468 characters, 120 tokens |
| empty.pdf | Error - no text found (empty PDF) |
| scanned.pdf | Error - no text found (scanned file - needs OCR) |

## Task 2 - Extraction prompt and schema

### Schema (`ResumeData` in `extractor.py`)
```json
{
  "candidate_name": "string or null",
  "skills": ["string"],
  "years_of_experience": "number or null",
  "education": ["string"],
  "strengths": ["string"],
  "gaps": ["string"]
}
```

### Prompt
`prompts/extract_v1.txt` - Role, Task, Output format and Constraints
("never invent facts", "use null when missing", "do not guess years of experience", "no text outside the JSON").

### Run settings
- Model: **Qwen2.5-1.5B-Instruct**, local on CPU (the Gemini API is blocked from Python on my network)
- **Temperature 0** (`do_sample=False`, greedy) - extraction must be repeatable
- **JSON mode**: the answer is forced to start with `{"candidate_name":`, the first complete JSON object is cut out,
  then validated with `json.loads` + the Pydantic schema

### Results
**Valid JSON for 8/8 resumes** (saved in `outputs/extract_v1/`), about 2-12 minutes per resume on CPU.

**Manual check of 3 resumes (compared with `truth.json`): 3 errors in total**

| Resume | Errors | Details |
|---|---|---|
| resume_01.pdf | 1 | Missed the skill "Git" |
| resume_03.pdf | 2 | years_of_experience = null (resume says 0.5 years); missed "Git" |
| resume_07.txt | 0 | Correctly returned null years and empty education - did not invent facts |

**Other observation:** `gaps` was empty for all three, even for resume_07 (no education, no dates).
Strengths were good and quoted real achievements (e.g. "Closed a Rs 2 crore deal with a logistics client").

**Next prompt version (v2):** tell the model to list every skill (including the last one in a list), to use the
"Total experience" line or internship months, and give examples of gaps (missing education, no dates, no numbers).

## Run
```powershell
python resume_analyzer/make_resumes.py
python resume_analyzer/task1_report.py
python resume_analyzer/task2_extract.py
python -m uvicorn app:app --app-dir resume_analyzer     # then open http://127.0.0.1:8000/docs
```

---

# AI Resume Analyzer - Part 2: Validation, job matching and scoring

## Task 1 - Validation and retry (`local_llm.py`, `extractor.py`, `task3_validation.py`)
- `ResumeData` (Pydantic) with constraints: `years_of_experience` between 0 and 50, `skills` list not empty
- Output parsed with `model_validate_json`; on error the **error message is sent back** to the model, up to **3 attempts**
- After 3 failures a clear **failure object** is returned: `{"success": false, "error": ..., "attempts": 3, "message": ...}`
- Every attempt is logged with input/output tokens and time (console + `logs/llm_calls.log`)

| Test | Log | Result |
|---|---|---|
| Messy resume ("exp 2018 -> now, approx 7 yrs", "team = 60+ yrs combined") | Attempt 1: success, 376 in + 106 out tokens | years = 7.0 (did not take 60); but "bsc maths" became "BCS (Bachelor of Computer Science)" - hallucination |
| Empty resume | Attempt 1: validation error (`skills: List should have at least 1 item`), 292 in + 34 out tokens. Attempt 2: success, 374 in + 59 out tokens | "Passed" only by **inventing** skills ("Python", "Data Analysis") and strengths |

**Lesson:** retry-with-feedback makes the model satisfy the schema, even by making things up.
Empty or scanned files must be rejected **before** the LLM (as `read_resume` does), and constraints like
"skills not empty" must never be the only quality check.

## Task 2 - Job description matching (`prompts/match_v1.txt`, `matcher.py`, `task4_match.py`)
- 2 job descriptions: **Python Developer** and **Data Analyst** (required skills, nice-to-have, experience)
- Matching prompt returns `score` (0-100), `matched_skills`, `missing_skills`, `reasoning` (validated with Pydantic)
- The candidate **name is removed** before scoring; the prompt says to ignore name, gender, age, location and college
- 8 resumes x 2 jobs x **2 runs** at temperature 0 = 32 calls - all valid JSON on the first attempt

### Ranking - Python Developer
| Rank | Candidate | Years | Run 1 | Run 2 | Consistent |
|---|---|---|---|---|---|
| 1 | Meena Iyer | 7 | 95 | 95 | yes |
| 2 | Ravi Kumar | 4 | 85 | 85 | yes |
| 3 | Dr. Fatima Khan | 9 | 85 | 85 | yes |
| 4 | Arjun S | - | 48 | 48 | yes |
| 5 | Karthik Raman | 5 | 48 | 48 | yes |
| 6 | Priya Natarajan | 3 | 48 | 48 | yes |
| 7 | Vikram Desai | - | 48 | 48 | yes |
| 8 | Anitha George | 6 | 45 | 45 | yes |

### Ranking - Data Analyst
| Rank | Candidate | Years | Run 1 | Run 2 | Consistent |
|---|---|---|---|---|---|
| 1 | Meena Iyer | 7 | 90 | 90 | yes |
| 2 | Dr. Fatima Khan | 9 | 85 | 85 | yes |
| 3 | Ravi Kumar | 4 | 75 | 75 | yes |
| 4 | Karthik Raman | 5 | 48 | 48 | yes |
| 5 | Priya Natarajan | 3 | 48 | 48 | yes |
| 6 | Anitha George | 6 | 43 | 43 | yes |
| 7 | Arjun S | - | 38 | 38 | yes |
| 8 | Vikram Desai | - | 38 | 38 | yes |

Full table with matched/missing skills and reasoning: `match_results.md`.

### Findings
- **Consistency: 16/16 pairs gave the same score twice** - but temperature 0 only makes the model repeatable, not correct.
- **Python Developer ranking is wrong at the top:** Meena (data analyst, no Django/FastAPI) ranked above Ravi (the real backend developer).
- **Invented matches:** Priya (HR) and Vikram (Sales) were "matched" with Python, Django and SQL, which are not in their resumes.
- The Data Analyst ranking is more sensible (Meena #1). Many candidates get the same "default" score (48), so the
  small 1.5B model cannot separate weak fits. A larger model, a deterministic keyword check of `matched_skills`
  against the extracted skills, and calibration examples would be the next steps.

## Fairness note - risks of automated resume scoring
1. The model can invent or miss skills (as seen above), so a strong candidate can be ranked low for no real reason.
2. LLMs learn from biased data and may favour certain names, colleges, genders, career gaps or writing styles - we removed the name, but bias can hide in other fields.
3. The score looks precise (95 vs 85) but is not calibrated; it should only be used to **sort for review, never to reject** a candidate automatically.
4. A human recruiter must check every shortlist against the original resume, review rejected candidates too, see the reasoning and missing skills, and keep an audit log - candidates should be told AI is used and be able to ask for a human review.

## API
`POST /match/{job_id}` (job_id = `python_developer` or `data_analyst`) - upload a resume, it is extracted and scored;
the response includes a note that a human must review it. `GET /health` lists the available jobs.