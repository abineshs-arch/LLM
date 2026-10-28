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