"""AI Resume Analyzer - FastAPI app.
GET /health, POST /parse, POST /extract, POST /match/{job_id}."""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from extractor import PROMPT_VERSION, extract_with_retry
from matcher import JOBS, match
from reader import ResumeError, count_tokens, read_resume

app = FastAPI(title="AI Resume Analyzer", version="0.3")
MAX_BYTES = 5 * 1024 * 1024            # 5 MB upload limit


async def upload_to_text(file: UploadFile) -> str:
    """Save the upload to a temporary file, read it with read_resume, return the cleaned text."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 5 MB)")
    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            return read_resume(path)
        except ResumeError as e:
            raise HTTPException(status_code=422, detail=str(e))


async def extract_or_fail(text):
    result = await run_in_threadpool(extract_with_retry, text, "api-extract")
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result)       # the clear failure object
    return result["data"]


@app.get("/health")
def health():
    return {"status": "ok", "prompt_version": PROMPT_VERSION, "jobs": list(JOBS)}


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    text = await upload_to_text(file)
    return {"filename": file.filename, "characters": len(text),
            "tokens": count_tokens(text), "preview": text[:300]}


@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    """Slow on CPU (1-3 minutes)."""
    data = await extract_or_fail(await upload_to_text(file))
    return data.model_dump()


@app.post("/match/{job_id}")
async def match_resume(job_id: str, file: UploadFile = File(...)):
    """Extract the resume, then score it against python_developer or data_analyst (slow on CPU)."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"unknown job, use one of {list(JOBS)}")
    data = await extract_or_fail(await upload_to_text(file))
    result = await run_in_threadpool(match, data.model_dump(), job_id, "api-match")
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result)
    return {"job": job_id, "candidate": data.candidate_name, **result["data"].model_dump(),
            "note": "Automated score - must be reviewed by a human before any decision."}