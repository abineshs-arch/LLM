"""Shared local LLM helper: load once, chat, and get validated JSON with retry + logging."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
import time
from pathlib import Path

from pydantic import ValidationError

BASE = Path(__file__).parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

(BASE / "logs").mkdir(exist_ok=True)
log = logging.getLogger("resume_analyzer")
if not log.handlers:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(BASE / "logs" / "llm_calls.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)

_model = {}


def load_model():
    """Load the model once and reuse it."""
    if not _model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _model["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _model["llm"] = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        _model["torch"] = torch
    return _model


def chat(messages, prefill="", max_new_tokens=450):
    """One call at temperature 0 (greedy). Returns (text, input_tokens, output_tokens, seconds)."""
    m = load_model()
    tok, llm, torch = m["tok"], m["llm"], m["torch"]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
    inputs = tok(prompt, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    text = prefill + tok.decode(new, skip_special_tokens=True)
    return text, inputs["input_ids"].shape[1], len(new), time.time() - start


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


def short_error(e):
    """'years_of_experience: Input should be less than or equal to 50' from a Pydantic error."""
    first = e.errors()[0]
    field = ".".join(str(x) for x in first["loc"]) or "json"
    return f"{field}: {first['msg']}"


def call_json(system, user, schema, prefill, label, max_attempts=3, max_new_tokens=450):
    """Ask for JSON, validate with schema.model_validate_json, retry with the error message.
    Returns {"success": True, "data": obj, "attempts": n, "usage": {...}} or a failure object."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    error = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(1, max_attempts + 1):
        raw, in_tok, out_tok, secs = chat(messages, prefill, max_new_tokens)
        usage["input_tokens"] += in_tok
        usage["output_tokens"] += out_tok
        candidate = first_json_object(raw) or raw
        try:
            data = schema.model_validate_json(candidate)
            log.info(f"[{label}] Attempt {attempt}: success | {in_tok} in + {out_tok} out tokens | {secs:.0f} s")
            return {"success": True, "data": data, "attempts": attempt, "usage": usage}
        except ValidationError as e:
            error = short_error(e)
            log.info(f"[{label}] Attempt {attempt}: validation error ({error}) | "
                     f"{in_tok} in + {out_tok} out tokens | {secs:.0f} s")
            messages += [{"role": "assistant", "content": candidate},
                         {"role": "user", "content": f"Your JSON is not valid: {error}. "
                                                     "Fix this and return only the corrected JSON object."}]
    log.info(f"[{label}] FAILED after {max_attempts} attempts")
    return {"success": False, "error": error, "attempts": max_attempts, "usage": usage,
            "message": f"Could not get valid JSON after {max_attempts} attempts. Last error: {error}"}