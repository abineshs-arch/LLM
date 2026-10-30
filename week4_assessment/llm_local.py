"""Local LLM helper for the Week 4 assessment (Qwen2.5-1.5B-Instruct on CPU, already downloaded).
Every answer is cached in results/cache.json, so a stopped script continues where it stopped."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from pathlib import Path

BASE = Path(__file__).parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_FILE = BASE / "results" / "cache.json"
CACHE_FILE.parent.mkdir(exist_ok=True)
_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
_model = {}


def load():
    if not _model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {MODEL} ...")
        _model["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _model["llm"] = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        _model["torch"] = torch
    return _model


def ask(key, prompt, system=None, temperature=0.0, top_p=1.0, max_new_tokens=200, prefill="", seed=0):
    """Returns dict(text, in_tokens, out_tokens, seconds). temperature 0 = greedy (repeatable)."""
    if key in _cache:
        return _cache[key]
    m = load()
    tok, llm, torch = m["tok"], m["llm"], m["torch"]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    inputs = tok(text, return_tensors="pt")
    settings = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if temperature > 0:
        settings.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    else:
        settings.update(do_sample=False)
    torch.manual_seed(seed)
    start = time.time()
    with torch.no_grad():
        out = llm.generate(**inputs, **settings)
    new = out[0][inputs["input_ids"].shape[1]:]
    result = {"text": (prefill + tok.decode(new, skip_special_tokens=True)).strip(),
              "in_tokens": int(inputs["input_ids"].shape[1]), "out_tokens": int(len(new)),
              "seconds": round(time.time() - start, 1)}
    _cache[key] = result
    CACHE_FILE.write_text(json.dumps(_cache, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"   [{key}] {result['in_tokens']} in + {result['out_tokens']} out tokens | {result['seconds']} s")
    return result


def cell(text, limit=160):
    """One table cell: single line, no pipes, shortened."""
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit] + "..."