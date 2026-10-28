"""Task 2: turn cleaned resume text into structured JSON with a local LLM (JSON mode, temperature 0)."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

BASE = Path(__file__).parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"          # small open model, runs on the laptop CPU
PROMPT_VERSION = "extract_v1"
SYSTEM_PROMPT = (BASE / "prompts" / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
PREFILL = '{\n  "candidate_name":'             # JSON mode: the answer is forced to start as our JSON


class ResumeData(BaseModel):
    """The extraction schema."""
    candidate_name: Optional[str]
    skills: list[str]
    years_of_experience: Optional[float]
    education: list[str]
    strengths: list[str]
    gaps: list[str]


class ExtractionError(Exception):
    pass


_model = {}


def load_model():
    """Load the model once and reuse it (first time downloads about 3 GB)."""
    if not _model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _model["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _model["llm"] = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        _model["torch"] = torch
    return _model


def first_json_object(text):
    """Return the first complete {...} object in text (braces inside strings are ignored)."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def generate(resume_text):
    m = load_model()
    tok, llm, torch = m["tok"], m["llm"], m["torch"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Resume:\n{resume_text}"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PREFILL
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=450, do_sample=False,      # do_sample=False = temperature 0
                           pad_token_id=tok.eos_token_id)
    return PREFILL + tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract(resume_text):
    """Returns (ResumeData, raw_model_text). Raises ExtractionError if the JSON is invalid."""
    raw = generate(resume_text)
    candidate = first_json_object(raw)
    if candidate is None:
        raise ExtractionError("model did not return a complete JSON object")
    try:
        return ResumeData(**json.loads(candidate)), raw
    except json.JSONDecodeError as e:
        raise ExtractionError(f"invalid JSON: {e.msg}")
    except ValidationError as e:
        raise ExtractionError(f"JSON does not match the schema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")