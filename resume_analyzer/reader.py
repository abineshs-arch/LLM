"""Task 1: read a resume (PDF or text), clean it, and validate it."""
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

ENC = tiktoken.get_encoding("o200k_base")
MIN_CHARS = 50                         # less than this = no usable text
PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?$", re.IGNORECASE)


class ResumeError(Exception):
    """Raised with a clear message when a resume cannot be used."""


def clean_text(pages):
    """pages = list of page texts. Removes repeated headers/footers, page numbers and extra spaces."""
    page_lines = [[" ".join(line.split()) for line in page.splitlines()] for page in pages]
    repeated = set()
    if len(pages) > 1:                 # a line that appears on every page is a header or footer
        counts = Counter(line for lines in page_lines for line in set(lines) if line)
        repeated = {line for line, c in counts.items() if c == len(pages)}
    out = []
    for lines in page_lines:
        for line in lines:
            if line in repeated or PAGE_NUMBER.match(line):
                continue
            if line == "" and (not out or out[-1] == ""):
                continue               # no double blank lines
            out.append(line)
    return "\n".join(out).strip()


def read_resume(path):
    path = Path(path)
    if not path.exists():
        raise ResumeError("file not found")
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(path)
        except Exception as e:
            raise ResumeError(f"cannot open PDF ({e})")
        with doc:
            pages = [page.get_text("text") for page in doc]
            has_images = any(page.get_images() for page in doc)
        text = clean_text(pages)
        if len(text) < MIN_CHARS:
            if has_images:
                raise ResumeError("no text found (scanned file - needs OCR)")
            raise ResumeError("no text found (empty PDF)")
    elif ext in (".txt", ".md"):
        text = clean_text([path.read_text(encoding="utf-8", errors="replace")])
        if len(text) < MIN_CHARS:
            raise ResumeError("no text found (empty text file)")
    else:
        raise ResumeError(f"unsupported file type '{ext}' (use .pdf or .txt)")
    return text


def count_tokens(text):
    return len(ENC.encode(text))