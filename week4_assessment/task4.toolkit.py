"""Task 4: prompt engineering toolkit (golden set, 3 prompt versions, CoT, sampling, JSON mode, bad prompts).
All answers are cached in results/cache.json; results tables -> results/task4_results.md"""
import json
import re
import sys
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from llm_local import BASE, MODEL, ask, cell

PROMPTS = BASE / "prompts"
PROMPTS.mkdir(exist_ok=True)
CATS, PRIOS = ["Billing", "Technical", "Account"], ["High", "Medium", "Low"]
report = [f"# Task 4 results - {MODEL} (local CPU)", ""]

# ---------------------------------------------------------------- golden set (my own labels)
TICKETS = [
    ("I was charged twice for my Pro subscription this month, please refund the extra payment.", "Billing", "High"),
    ("Can I get a copy of last month's invoice with our GST number on it?", "Billing", "Low"),
    ("The discount coupon SAVE20 was not applied to my renewal bill.", "Billing", "Medium"),
    ("My card was charged but the order shows payment failed. Money is gone from my bank!", "Billing", "High"),
    ("Please change the billing address on future invoices to our new office.", "Billing", "Low"),
    ("The app crashes every time I open the reports page since yesterday's update.", "Technical", "Medium"),
    ("Our whole team cannot load the dashboard - blank screen for everyone, work has stopped.", "Technical", "High"),
    ("The export to PDF button does nothing on Firefox, Chrome works fine.", "Technical", "Low"),
    ("Checkout shows error E-102 every time I try to pay, I cannot complete my purchase.", "Technical", "High"),
    ("The website is a bit slow in the evenings, not a big problem, just letting you know.", "Technical", "Low"),
    ("I can't log in - the password reset email never arrives and I must send invoices to clients today.", "Account", "High"),
    ("Someone changed my login email without my permission and I see orders I did not place.", "Account", "High"),
    ("How do I add a second team member to my account?", "Account", "Low"),
    ("My account was locked after too many attempts, please unlock it this week.", "Account", "Medium"),
    ("I want to update the phone number on my profile.", "Account", "Low"),
]
EXAMPLES = [   # few-shot examples - NOT in the golden set, chosen for edge cases
    ("The payment page shows error 500 when I try to pay with UPI.", "Technical", "High"),     # payment word, but a bug
    ("I was billed for 5 users but we only have 3, please correct the next invoice.", "Billing", "Medium"),
    ("I forgot my password but I can still use the mobile app, no hurry.", "Account", "Low"),  # login, low impact
]

RESUMES = [
    ("Ravi Kumar is a Python developer from Chennai with 4 years of experience. He works with Django, PostgreSQL "
     "and Docker, and recently started using AWS.",
     {"name": "Ravi Kumar", "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS"], "years_experience": 4}),
    ("Meena Iyer has spent the last seven years as a data analyst. Her toolkit: SQL, Excel, Power BI and some Python.",
     {"name": "Meena Iyer", "skills": ["SQL", "Excel", "Power BI", "Python"], "years_experience": 7}),
    ("Arjun S - fresher, graduated 2025. Internship (6 months) in React and Node.js. Knows Git. Total experience: 0.5 years.",
     {"name": "Arjun S", "skills": ["React", "Node.js", "Git"], "years_experience": 0.5}),
    ("Dr. Fatima Khan, ML engineer, 9 yrs. Skills include PyTorch, NLP, computer vision and MLOps (Kubeflow).",
     {"name": "Fatima Khan", "skills": ["PyTorch", "NLP", "Computer Vision", "MLOps", "Kubeflow"], "years_experience": 9}),
    ("Karthik worked 2 years in QA (Selenium, JIRA) and then 3 years in DevOps with Jenkins, Kubernetes and Terraform.",
     {"name": "Karthik", "skills": ["Selenium", "JIRA", "Jenkins", "Kubernetes", "Terraform"], "years_experience": 5}),
]

# ---------------------------------------------------------------- 3 classification prompt versions
V1 = "Classify this support ticket and give its priority."
V2 = """ROLE
You are a customer-support ticket classifier.

TASK
Choose exactly one CATEGORY and one PRIORITY for the ticket.
Categories:
- Billing: charges, refunds, invoices, coupons, payment amounts
- Technical: app or website errors, crashes, bugs, slow pages, error codes (even on the payment page)
- Account: login, password, email or phone change, locked account, team members, security
Priorities:
- High: money lost, customer completely blocked, security risk, or a deadline today
- Medium: a real problem that needs fixing soon but has a workaround or no deadline
- Low: questions, small requests, minor issues with no real impact

CONSTRAINTS
- Use only the category and priority names above, spelled exactly.
- Do not explain your answer.

OUTPUT FORMAT
Category | Priority"""
V3 = V2 + "\n\nEXAMPLES\n" + "\n".join(f'Ticket: "{t}"\nAnswer: {c} | {p}\n' for t, c, p in EXAMPLES)


def save_inputs():
    for name, text in [("classify_v1.txt", V1), ("classify_v2.txt", V2), ("classify_v3.txt", V3)]:
        (PROMPTS / name).write_text(text, encoding="utf-8")
    (BASE / "golden_tickets.json").write_text(json.dumps(
        [{"text": t, "category": c, "priority": p} for t, c, p in TICKETS], indent=2), encoding="utf-8")
    (BASE / "golden_resumes.json").write_text(json.dumps(
        [{"text": t, "expected": e} for t, e in RESUMES], indent=2), encoding="utf-8")


def first_word(text, words):
    hits = [(m.start(), w) for w in words for m in [re.search(rf"\b{w}\b", text, re.I)] if m]
    return min(hits)[1] if hits else None


# ---------------------------------------------------------------- A. classification v1 / v2 / v3
def part_a():
    print("\n=== A. Classification prompts v1 / v2 / v3 (temperature 0) ===")
    versions = {"v1 basic": (None, "{t}\n\n" + V1), "v2 Role/Task/Constraints/Output": (V2, "Ticket: {t}"),
                "v3 + 3 few-shot examples": (V3, "Ticket: {t}")}
    report.extend(["## A. Classification prompt versions (15 golden tickets, temperature 0)", "",
                   "| Version | Category correct | Priority correct | Both correct (accuracy) | Avg input tokens |",
                   "|---|---|---|---|---|"])
    mistakes = {}
    for vi, (name, (system, template)) in enumerate(versions.items(), 1):
        cat_ok = pri_ok = both = tokens = 0
        mistakes[name] = []
        for i, (text, cat, pri) in enumerate(TICKETS, 1):
            r = ask(f"t4a_v{vi}_{i}", template.format(t=text), system=system, max_new_tokens=60)
            pc, pp = first_word(r["text"], CATS), first_word(r["text"], PRIOS)
            cat_ok += pc == cat
            pri_ok += pp == pri
            both += (pc, pp) == (cat, pri)
            tokens += r["in_tokens"]
            if (pc, pp) != (cat, pri):
                mistakes[name].append(f"T{i}: {cat}/{pri} -> {pc}/{pp}")
        print(f"  {name:32} category {cat_ok}/15 | priority {pri_ok}/15 | both {both}/15 ({both / 15:.0%})")
        report.append(f"| {name} | {cat_ok}/15 | {pri_ok}/15 | **{both}/15 ({both / 15:.0%})** | {tokens // 15} |")
    report.append("")
    for name, m in mistakes.items():
        report.append(f"- **{name}** mistakes: " + ("none" if not m else "; ".join(m)))
        print(f"  {name} mistakes: {m}")
    report.append("")


# ---------------------------------------------------------------- B. chain-of-thought
PROBLEMS = [
    ("A shop has 48 apples. It sells 3 boxes of 12 apples each and then receives 20 more apples. How many apples now?", 32),
    ("Priya earns Rs 600 per day. She works 5 days a week for 4 weeks and spends Rs 4500. How many rupees does she save?", 7500),
    ("A tank holds 500 litres and is 40% full. 120 litres are added and then 50 litres are used. How many litres now?", 270),
    ("A class has 30 students and 2/5 are girls. Each boy brings 3 notebooks and each girl 2. How many notebooks in total?", 78),
    ("I have 3 times as many marbles as Arun. Together we have 48 marbles. How many marbles do I have?", 36),
]


def last_number(text):
    m = re.search(r"answer\s*[:=]\s*\**\s*(?:rs\.?\s*)?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return float(nums[-1].replace(",", "")) if nums else None


def part_b():
    print("\n=== B. Chain-of-thought ===")
    modes = {"Direct (final number only)": ("Answer with only the final number. No explanation.", 20),
             "Think step by step": ("Think step by step. Then write the final answer on the last line as "
                                    "'Answer: <number>'.", 400)}
    report.extend(["## B. Chain-of-thought (5 word problems, temperature 0)", "",
                   "| Mode | Correct | Output tokens (total) | Answers (expected 32, 7500, 270, 78, 36) |", "|---|---|---|---|"])
    for mi, (mode, (instruction, max_tok)) in enumerate(modes.items(), 1):
        correct = out_tok = 0
        answers = []
        for i, (q, truth) in enumerate(PROBLEMS, 1):
            r = ask(f"t4b_m{mi}_{i}", f"{q}\n\n{instruction}", max_new_tokens=max_tok)
            got = last_number(r["text"])
            correct += got is not None and abs(got - truth) < 0.01
            out_tok += r["out_tokens"]
            answers.append(str(got))
        print(f"  {mode:28} {correct}/5 correct | {out_tok} output tokens | {answers}")
        report.append(f"| {mode} | {correct}/5 | {out_tok} | {', '.join(answers)} |")
    report.append("")


# ---------------------------------------------------------------- C. temperature and top-p
CREATIVE = "Write a catchy one-sentence slogan for a small filter-coffee shop in Chennai."


def part_c():
    print("\n=== C. Temperature and top-p ===")
    report.extend(["## C. Sampling - same creative prompt, 5 runs each", f"Prompt: *{CREATIVE}*", "",
                   "| Setting | Different answers (of 5) | Answers |", "|---|---|---|"])
    settings = [("temperature 0", 0.0, 1.0), ("temperature 0.7", 0.7, 1.0), ("temperature 1.2", 1.2, 1.0),
                ("temperature 1.0, top_p 0.3", 1.0, 0.3), ("temperature 1.0, top_p 0.95", 1.0, 0.95)]
    for si, (name, t, p) in enumerate(settings, 1):
        outs = [ask(f"t4c_s{si}_{seed}", CREATIVE, temperature=t, top_p=p, max_new_tokens=40, seed=seed)["text"]
                .split("\n")[0].strip('" ') for seed in range(5)]
        print(f"  {name:28} {len(set(outs))}/5 different")
        for o in outs:
            print(f"      - {o[:110]}")
        report.append(f"| {name} | {len(set(outs))} | " + "<br>".join(cell(o, 110) for o in outs) + " |")
    report.append("")


# ---------------------------------------------------------------- D. structured output
class Resume(BaseModel):
    name: str
    skills: list[str] = Field(min_length=1)
    years_experience: Optional[float] = Field(default=None, ge=0, le=50)


def first_json_object(text):
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            esc, in_str = (False, in_str) if esc else (ch == "\\", ch != '"')
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def validate(text):
    try:
        return Resume.model_validate_json(text), "valid"
    except ValidationError as e:
        err = e.errors()[0]
        return None, f"invalid ({'.'.join(map(str, err['loc'])) or 'json'}: {err['msg'][:50]})"


def part_d():
    print("\n=== D. Structured output: plain vs JSON mode ===")
    plain = "Extract the person's name, skills and years of experience from this resume and return JSON.\n\n{t}"
    json_system = ('You are a JSON API. Reply with ONE JSON object and nothing else: '
                   '{"name": string, "skills": [string], "years_experience": number or null}')
    prefill = '{"name": "'
    report.extend(["## D. Structured output - {name, skills[], years_experience} from 5 resumes", "",
                   "Every output is parsed with Pydantic `model_validate_json` (name str, skills non-empty list, "
                   "years 0-50).", "",
                   "| Resume | Plain prompting | JSON mode | JSON mode name / years correct? |", "|---|---|---|---|"])
    invalid = {"plain": 0, "json": 0}
    for i, (text, expected) in enumerate(RESUMES, 1):
        rp = ask(f"t4d_plain_{i}", plain.format(t=text), max_new_tokens=150)
        _, sp = validate(rp["text"])
        rj = ask(f"t4d_json_{i}", f"Resume:\n{text}", system=json_system, prefill=prefill, max_new_tokens=150)
        obj, sj = validate(first_json_object(rj["text"]) or rj["text"])
        invalid["plain"] += sp != "valid"
        invalid["json"] += sj != "valid"
        correct = "-"
        if obj:
            name_ok = expected["name"].lower() in obj.name.lower()
            yrs_ok = obj.years_experience is not None and abs(obj.years_experience - expected["years_experience"]) < 0.01
            correct = f"name {'yes' if name_ok else 'NO'}, years {'yes' if yrs_ok else 'NO'} ({obj.years_experience})"
        print(f"  Resume {i}: plain {sp} | JSON mode {sj} | {correct}")
        report.append(f"| {i} | {sp} | {sj} | {correct} |")
    print(f"  Plain: {invalid['plain']}/5 invalid | JSON mode: {invalid['json']}/5 invalid")
    report.extend(["", f"**Plain: {invalid['plain']}/5 invalid JSON | JSON mode: {invalid['json']}/5 invalid**", ""])


# ---------------------------------------------------------------- E. 3 bad prompts and rewrites
TICKET = "My order #45120 was delivered to the wrong address in Coimbatore and I need it before Friday."
REVIEW = "The phone looked nice but the battery died after two hours and support never replied. Not buying again."
BAD = [
    ("Vague instruction", f"Tell me about this ticket: {TICKET}",
     "Classify this support ticket into exactly one category from [Billing, Technical, Account, Shipping]. "
     f"Reply with the category name only.\n\nTicket: {TICKET}",
     lambda a: a.strip().strip(".").lower() == "shipping"),
    ("No output format", f"Find the order number and city in this ticket: {TICKET}",
     'Extract the order number and city. Return ONLY this JSON, nothing else: {"order_id": "...", "city": "..."}'
     f"\n\nTicket: {TICKET}",
     lambda a: (lambda d: d is not None and "45120" in str(d.get("order_id", "")) and
                str(d.get("city", "")).lower() == "coimbatore")(_load(a))),
    ("Conflicting instructions",
     f"Summarise this review in one word. Give a detailed explanation with at least three reasons.\n\nReview: {REVIEW}",
     f"What is the sentiment of this review? Reply with one word: positive, negative or neutral.\n\nReview: {REVIEW}",
     lambda a: a.strip().strip(".").lower() == "negative"),
]


def _load(text):
    try:
        return json.loads(first_json_object(text) or "")
    except json.JSONDecodeError:
        return None


def part_e():
    print("\n=== E. Bad prompts vs rewritten ===")
    report.extend(["## E. Three bad prompts and their rewrites", "",
                   "| Mistake | Bad prompt -> answer | Rewritten prompt -> answer | Bad passes? | Rewrite passes? |",
                   "|---|---|---|---|---|"])
    passed = [0, 0]
    for i, (name, bad, good, check) in enumerate(BAD, 1):
        rb = ask(f"t4e_bad_{i}", bad, max_new_tokens=120)
        rg = ask(f"t4e_good_{i}", good, max_new_tokens=60)
        ok_b, ok_g = check(rb["text"]), check(rg["text"])
        passed[0] += ok_b
        passed[1] += ok_g
        print(f"  {name:26} bad -> {'PASS' if ok_b else 'fail'} | rewritten -> {'PASS' if ok_g else 'fail'}: {cell(rg['text'], 80)}")
        report.append(f"| {name} | {cell(bad, 70)} -> {cell(rb['text'], 90)} | {cell(good, 70)} -> {cell(rg['text'], 90)} | "
                      f"{'yes' if ok_b else 'no'} | {'yes' if ok_g else 'no'} |")
    print(f"  Bad prompts passed {passed[0]}/3 | rewritten prompts passed {passed[1]}/3")
    report.extend(["", f"**Bad prompts passed {passed[0]}/3 - rewritten prompts passed {passed[1]}/3**", ""])


if __name__ == "__main__":
    save_inputs()
    parts = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}
    for key in (sys.argv[1:] or parts):
        parts[key.upper()]()
    (BASE / "results" / "task4_results.md").write_text("\n".join(report), encoding="utf-8")
    print("\nSaved results/task4_results.md")