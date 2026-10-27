"""Week 4 Day 3 (part 2) - chain-of-thought, temperature/top-p, structured JSON output, fixing weak prompts.

Runs a small open model (Qwen2.5-0.5B-Instruct) locally on the CPU, so temperature and top_p
can really be changed and every experiment runs automatically. Results -> day3_part2_results.md
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import re
import time
from pathlib import Path

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = BASE / "day3_part2_results.md"

print(f"Loading {MODEL} (first time downloads about 1 GB)...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
report = [f"# Week 4 Day 3 (part 2) results - model {MODEL} (local, CPU)", ""]


def generate(prompt, system=None, temperature=0.0, top_p=1.0, max_new_tokens=200, prefill="", seed=0):
    """One chat call. temperature 0 = greedy (always the same answer)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    inputs = tok(text, return_tensors="pt")
    settings = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, top_k=0)
    if temperature > 0:
        settings.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        settings.update(do_sample=False)
    torch.manual_seed(seed)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **settings)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = prefill + tok.decode(new_tokens, skip_special_tokens=True)
    return answer.strip(), len(new_tokens), time.time() - start


def cell(text, limit=150):
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit] + "..."


# ======================================================================
# TASK 1a - chain-of-thought on 5 word problems
# ======================================================================
PROBLEMS = [
    ("A shop has 48 apples. It sells 3 boxes of 12 apples each and then receives 20 more apples. "
     "How many apples does it have now?", 32),
    ("Priya earns Rs 600 per day. She works 5 days a week for 4 weeks and spends Rs 4500. "
     "How many rupees does she save?", 7500),
    ("A tank holds 500 litres and is 40% full. 120 litres are added and then 50 litres are used. "
     "How many litres are in the tank now?", 270),
    ("A class has 30 students and 2/5 of them are girls. Each boy brings 3 notebooks and each girl "
     "brings 2 notebooks. How many notebooks are there in total?", 78),
    ("I have 3 times as many marbles as Arun. Together we have 48 marbles. How many marbles do I have?", 36),
]


def last_number(text):
    m = re.search(r"answer\s*[:=]\s*\**\s*(?:rs\.?\s*)?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return float(nums[-1].replace(",", "")) if nums else None


def task1a():
    print("\n=== TASK 1a: chain-of-thought ===")
    modes = {
        "Direct answer": "Answer with only the final number. No explanation.",
        "Think step by step": "Think step by step. Then write the final answer on the last line as 'Answer: <number>'.",
    }
    rows, summary = [], {}
    for mode, instruction in modes.items():
        correct = tokens = 0
        secs = 0.0
        for i, (q, truth) in enumerate(PROBLEMS, 1):
            ans, n, s = generate(f"{q}\n\n{instruction}", max_new_tokens=300)
            got = last_number(ans)
            ok = got is not None and abs(got - truth) < 0.01
            correct += ok
            tokens += n
            secs += s
            rows.append((mode, i, truth, got, "yes" if ok else "NO", n))
            print(f"  {mode:20} P{i}: expected {truth}, got {got} -> {'OK' if ok else 'WRONG'} ({n} tokens)")
        summary[mode] = (correct, tokens, secs)

    report.extend(["## Task 1a - Chain-of-thought (5 word problems, greedy)", "",
                   "| Mode | Correct | Output tokens (total) | Time (s) |", "|---|---|---|---|"])
    for mode, (c, t, s) in summary.items():
        report.append(f"| {mode} | {c}/5 | {t} | {s:.1f} |")
    report.extend(["", "| Mode | Problem | Expected | Model answer | Correct | Tokens |", "|---|---|---|---|---|---|"])
    report.extend(f"| {m} | {i} | {t} | {g} | {ok} | {n} |" for m, i, t, g, ok, n in rows)
    report.append("")


# ======================================================================
# TASK 1b/1c - temperature and top-p
# ======================================================================
CREATIVE = "Write a catchy one-sentence slogan for a small filter-coffee shop in Chennai."


def sample_runs(temperature, top_p, runs=5):
    outs = []
    for seed in range(runs):
        ans, _, _ = generate(CREATIVE, temperature=temperature, top_p=top_p, max_new_tokens=40, seed=seed)
        outs.append(ans.split("\n")[0].strip('" '))
    return outs


def task1bc():
    print("\n=== TASK 1b: temperature (top_p = 1.0) ===")
    report.extend(["## Task 1b - Temperature (same creative prompt, 5 runs each, top_p = 1.0)",
                   f"Prompt: *{CREATIVE}*", "",
                   "| Temperature | Different answers (of 5) | Answers |", "|---|---|---|"])
    for t in [0.0, 0.7, 1.2]:
        outs = sample_runs(t, 1.0)
        print(f"  T={t}: {len(set(outs))}/5 different")
        for o in outs:
            print(f"     - {o}")
        report.append(f"| {t} | {len(set(outs))} | " + "<br>".join(cell(o, 110) for o in outs) + " |")

    print("\n=== TASK 1c: top_p at temperature 1.0 ===")
    report.extend(["", "## Task 1c - Top-p (temperature 1.0, 5 runs each)", "",
                   "| top_p | Different answers (of 5) | Answers |", "|---|---|---|"])
    for p in [0.3, 0.95]:
        outs = sample_runs(1.0, p)
        print(f"  top_p={p}: {len(set(outs))}/5 different")
        for o in outs:
            print(f"     - {o}")
        report.append(f"| {p} | {len(set(outs))} | " + "<br>".join(cell(o, 110) for o in outs) + " |")
    report.append("")


# ======================================================================
# TASK 2a - structured JSON output: plain prompting vs JSON mode
# ======================================================================
RESUMES = [
    "Ravi Kumar is a Python developer from Chennai with 4 years of experience. He works with Django, "
    "PostgreSQL and Docker, and recently started using AWS.",
    "Meena Iyer has spent the last seven years as a data analyst. Her toolkit: SQL, Excel, Power BI and some Python.",
    "Arjun S. - fresher, graduated 2025. Internship (6 months) in React and Node.js. Knows Git. Total experience: 0.5 years.",
    "Dr. Fatima Khan, ML engineer, 9 yrs. Skills include PyTorch, NLP, computer vision, MLOps (Kubeflow) and team leadership.",
    "Karthik worked 2 years in QA (manual testing, Selenium, JIRA) and then 3 years in DevOps with Jenkins, Kubernetes "
    "and Terraform.",
]
SCHEMA = '{"name": string, "skills": [string], "years_experience": number}'


class Resume(BaseModel):
    name: str
    skills: list[str]
    years_experience: float


def json_mode(paragraph):
    """Local 'JSON mode': strict JSON system prompt + the answer is forced to start with '{"name": "'
    + we cut the text at the matching closing brace. Hosted APIs (response_format / Structured outputs)
    do this even more strictly by constraining every generated token."""
    system = f"You are a JSON API. Reply with one JSON object matching {SCHEMA} and nothing else."
    ans, n, s = generate(f"Resume:\n{paragraph}", system=system, prefill='{"name": "', max_new_tokens=150)
    depth, end = 0, None
    for i, ch in enumerate(ans):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0 and ch == "}":
            end = i + 1
            break
    return (ans[:end] if end else ans), n, s


def check(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return "invalid JSON", f"json.loads: {e.msg}"
    try:
        Resume(**data) if isinstance(data, dict) else Resume.model_validate(data)
        return "valid", ""
    except (ValidationError, TypeError) as e:
        return "JSON ok, schema FAIL", cell(str(e).split("\n")[1] if "\n" in str(e) else str(e), 80)


def task2a():
    print("\n=== TASK 2a: plain prompting vs JSON mode ===")
    plain_prompt = "Extract the person's name, skills and years of experience from this resume and return JSON.\n\n"
    rows, invalid = [], {"Plain prompting": 0, "JSON mode": 0}
    for i, para in enumerate(RESUMES, 1):
        for mode in ["Plain prompting", "JSON mode"]:
            if mode == "Plain prompting":
                ans, n, s = generate(plain_prompt + para, max_new_tokens=150)
            else:
                ans, n, s = json_mode(para)
            status, why = check(ans)
            invalid[mode] += status != "valid"
            rows.append((i, mode, status, why, ans))
            print(f"  Resume {i} | {mode:15} | {status} {why}")

    report.extend(["## Task 2a - Structured output: plain prompting vs JSON mode", "",
                   f"Schema: `{SCHEMA}`, every output checked with `json.loads` and a Pydantic model.", "",
                   "| Method | Invalid outputs (json.loads or Pydantic failed) |", "|---|---|",
                   f"| Plain prompting | {invalid['Plain prompting']}/5 |",
                   f"| JSON mode (strict system prompt + forced '{{' start + cut at closing brace) | {invalid['JSON mode']}/5 |",
                   "", "| Resume | Method | Result | Reason | Output |", "|---|---|---|---|---|"])
    report.extend(f"| {i} | {m} | {st} | {w} | `{cell(a, 120)}` |" for i, m, st, w, a in rows)
    report.append("")


# ======================================================================
# TASK 2b - 4 bad prompts and their rewrites
# ======================================================================
TICKET = "My order #45120 was delivered to the wrong address in Coimbatore and I need it before Friday."
REVIEW = "The phone looked nice but the battery died after two hours and support never replied. Not buying again."

BAD_PROMPTS = [
    ("Vague instruction",
     f"Tell me about this ticket: {TICKET}",
     "Classify this support ticket into exactly one category from [Billing, Technical, Account, Shipping, General]. "
     f"Reply with the category name only.\n\nTicket: {TICKET}",
     lambda a: a.strip().strip(".").lower() == "shipping",
     "answer is exactly 'Shipping'"),
    ("No output format",
     f"Find the order number and city in this ticket: {TICKET}",
     'Extract the order number and city from the ticket. Return only JSON like {"order_id": "...", "city": "..."}.'
     f"\n\nTicket: {TICKET}",
     lambda a: _json_has(a, {"order_id": "45120", "city": "coimbatore"}),
     "valid JSON with order_id 45120 and city Coimbatore"),
    ("No examples",
     "Convert the date 'March 5th, 2026' to our standard date format.",
     "Convert dates to our standard format.\nExamples:\nJanuary 12th, 2025 -> 2025-01-12\nJuly 3rd, 2024 -> 2024-07-03\n"
     "Now convert: March 5th, 2026 ->\nReply with the converted date only.",
     lambda a: a.strip().strip(".") == "2026-03-05",
     "answer is exactly 2026-03-05"),
    ("Conflicting instructions",
     f"Summarise this review in one word. Give a detailed explanation with at least three reasons.\n\nReview: {REVIEW}",
     f"What is the sentiment of this review? Reply with one word: positive, negative or neutral.\n\nReview: {REVIEW}",
     lambda a: a.strip().strip(".").lower() == "negative",
     "answer is exactly 'negative'"),
]


def _json_has(text, expected):
    try:
        data = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except json.JSONDecodeError:
        return False
    return all(str(data.get(k, "")).strip("#").lower() == v for k, v in expected.items())


def task2b():
    print("\n=== TASK 2b: bad prompts vs rewritten prompts ===")
    report.extend(["## Task 2b - Fixing weak prompts", "",
                   "| Mistake | Version | Prompt | Model answer | Passes check? |", "|---|---|---|---|---|"])
    passed = {"bad": 0, "rewritten": 0}
    for name, bad, good, test, rule in BAD_PROMPTS:
        for version, prompt in [("bad", bad), ("rewritten", good)]:
            ans, n, _ = generate(prompt, max_new_tokens=150)
            ok = test(ans)
            passed[version] += ok
            print(f"  {name:25} {version:10} -> {'PASS' if ok else 'fail'} | {cell(ans, 90)}")
            report.append(f"| {name} | {version} | {cell(prompt, 120)} | {cell(ans, 120)} | "
                          f"{'yes' if ok else 'no'} ({rule}) |")
    report.extend(["", f"**Bad prompts passed: {passed['bad']}/4 - rewritten prompts passed: {passed['rewritten']}/4**", ""])


if __name__ == "__main__":
    task1a()
    task1bc()
    task2a()
    task2b()
    OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved all tables to {OUT.name}")