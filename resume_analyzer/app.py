"""AI Resume Analyzer - FastAPI app (Part 3).
POST /analyze: resume file (PDF/TXT) or resume text + job description -> structured data, score, tokens, cost."""
import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import config
from extractor import extract_with_retry
from matcher import JOBS, match_jd
from reader import ResumeError, clean_text, count_tokens, read_resume

log = logging.getLogger("resume_analyzer")
app = FastAPI(title="AI Resume Analyzer", version="0.4",
              description="Upload a resume and a job description. Scores must be reviewed by a human.")
llm_lock = asyncio.Lock()             # the local model can do one job at a time ("rate limit")
EXAMPLE_JD = JOBS["python_developer"]


class LLMError(Exception):
    """Raised with an HTTP status code and a friendly message."""
    def __init__(self, status, message):
        self.status, self.message = status, message


# ---------------- request id + logging ----------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.rid = uuid.uuid4().hex[:8]
    start = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.rid
    log.info(f"req={request.state.rid} {request.method} {request.url.path} -> {response.status_code} "
             f"in {time.time() - start:.1f} s")
    return response


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    headers = {"Retry-After": "60"} if exc.status == 503 else {}
    return JSONResponse(status_code=exc.status, headers=headers,
                        content={"detail": exc.message, "request_id": request.state.rid})


# ---------------- helpers ----------------
async def resume_text_from_request(file: Optional[UploadFile], resume_text: Optional[str]) -> str:
    """Validate the upload (type, size, empty) or the pasted text and return cleaned text."""
    if file is None and not resume_text:
        raise HTTPException(422, "send a resume file or resume_text")
    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in config.ALLOWED_TYPES:
            raise HTTPException(415, f"unsupported file type '{suffix}', use .pdf or .txt")
        data = await file.read()
        if not data:
            raise HTTPException(422, "the uploaded file is empty")
        if len(data) > config.MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, f"file too large (max {config.MAX_FILE_MB:g} MB)")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                text = read_resume(path)
            except ResumeError as e:
                raise HTTPException(422, str(e))
    else:
        text = clean_text([resume_text])
        if len(text) < 50:
            raise HTTPException(422, "resume_text is too short (min 50 characters)")
    tokens = count_tokens(text)
    if tokens > config.MAX_RESUME_TOKENS:
        raise HTTPException(413, f"resume too long: {tokens} tokens (limit {config.MAX_RESUME_TOKENS})")
    return text


async def run_llm(fn, *args, retries=1):
    """Run one LLM step in a thread with a timeout. Retries once on timeout or busy, then a friendly error."""
    for attempt in range(retries + 1):
        if llm_lock.locked():
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            raise LLMError(503, "The AI model is busy with another request (rate limit). Please try again in a minute.")
        async with llm_lock:
            try:
                result = await asyncio.wait_for(run_in_threadpool(fn, *args), timeout=config.LLM_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                if attempt < retries:
                    continue
                raise LLMError(503, f"The AI model did not answer within {config.LLM_TIMEOUT_SECONDS:g} seconds. "
                                    "Please try again later.")
            except Exception as e:                       # any crash inside the model call
                log.info(f"LLM crashed: {type(e).__name__}")
                raise LLMError(502, "The AI model returned an error. Please try again later.")
        if not result["success"]:
            raise LLMError(502, f"The AI model did not return valid JSON after {result['attempts']} attempts "
                                f"({result['error']}).")
        return result


# ---------------- endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok", "model_busy": llm_lock.locked(), **config.public_settings()}


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    text = await resume_text_from_request(file, None)
    return {"filename": file.filename, "characters": len(text), "tokens": count_tokens(text), "preview": text[:300]}


@app.post("/analyze")
async def analyze(request: Request,
                  job_description: str = Form(..., min_length=20, examples=[EXAMPLE_JD]),
                  file: Optional[UploadFile] = File(None),
                  resume_text: Optional[str] = Form(None)):
    """Example: upload resumes/resume_01.pdf and paste the Python Developer job description."""
    rid = request.state.rid
    text = await resume_text_from_request(file, resume_text)
    extracted = await run_llm(extract_with_retry, text, f"req={rid} extract")
    data = extracted["data"]
    scored = await run_llm(match_jd, data.model_dump(), job_description, f"req={rid} match")
    result = scored["data"]
    in_tok = extracted["usage"]["input_tokens"] + scored["usage"]["input_tokens"]
    out_tok = extracted["usage"]["output_tokens"] + scored["usage"]["output_tokens"]
    return {"request_id": rid,
            "candidate_name": data.candidate_name, "score": result.score,
            "matched_skills": result.matched_skills, "missing_skills": result.missing_skills,
            "reasoning": result.reasoning, "resume": data.model_dump(),
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                      "estimated_cost_usd": config.estimate_cost(in_tok, out_tok),
                      "cost_note": "local model is free; cost shown as if run on Gemini 3.8 Flash"},
            "note": "Automated score - must be reviewed by a human before any decision."}