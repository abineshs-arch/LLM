"""AI Resume Analyzer - FastAPI app.
GET /health, POST /parse (text, characters, tokens) and POST /extract (structured JSON)."""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from extractor import PROMPT_VERSION, ExtractionError, ResumeData, extract
from reader import ResumeError, count_tokens, read_resume

app = FastAPI(title="AI Resume Analyzer", version="0.2")
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


@app.get("/health")
def health():
    return {"status": "ok", "prompt_version": PROMPT_VERSION}


@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    text = await upload_to_text(file)
    return {"filename": file.filename, "characters": len(text),
            "tokens": count_tokens(text), "preview": text[:300]}


@app.post("/extract", response_model=ResumeData)
async def extract_resume(file: UploadFile = File(...)):
    """First call loads the model (slow). The model runs in a thread so the server stays responsive."""
    text = await upload_to_text(file)
    try:
        data, _ = await run_in_threadpool(extract, text)
    except ExtractionError as e:
        raise HTTPException(status_code=502, detail=f"extraction failed: {e}")
    return data