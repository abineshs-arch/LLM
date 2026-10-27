"""Week 4 Day 3 - zero-shot vs few-shot prompting lab (10 tickets, 8 prompt variants)."""
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
PROMPTS = BASE / "prompts"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

CATEGORIES = ["Billing", "Technical", "Account", "Shipping", "General"]
PRIORITIES = ["High", "Medium", "Low"]

# 10 test tickets with MY labels (category, priority). Several are deliberately ambiguous.
TICKETS = [
    ("I was charged twice for my Pro subscription this month, please refund the extra payment.", "Billing", "High"),
    ("The app crashes every time I open the reports page since yesterday's update.", "Technical", "Medium"),
    ("I can't log in - the password reset email never arrives and I must send invoices to clients today.", "Account", "High"),
    ("My order #45120 was marked as delivered but nothing arrived at my door.", "Shipping", "High"),
    ("Do you have a student discount on the yearly plan?", "General", "Low"),
    ("Someone changed my login email without my permission and I see orders I did not place.", "Account", "High"),
    ("The website is a bit slow in the evenings, not a big problem, just letting you know.", "Technical", "Low"),
    ("Can you add a dark mode to the mobile app? It would be great.", "General", "Low"),
    ("The laptop I received has a cracked screen, it was damaged during delivery.", "Shipping", "Medium"),
    ("My card payment failed three times, the checkout page shows error E-102.", "Technical", "High"),
]

# Few-shot examples - NOT the same as the test tickets, chosen to cover edge cases
EXAMPLES = [
    ("The checkout page shows error 500 when I try to pay with UPI.", "Technical", "High"),        # payment word, but a bug
    ("Do you offer a discount for NGOs?", "General", "Low"),                                       # discount question, not Billing
    ("Please change the delivery address on my order before it ships tomorrow.", "Shipping", "Medium"),  # address, but for an order
    ("I forgot my password but I can still use the mobile app, no hurry.", "Account", "Low"),      # login problem, but no impact
    ("The headphones I received stopped working after two days.", "Technical", "Medium"),         # received item, but a device fault
]


def examples_block(examples):
    lines = ["", "EXAMPLES"]
    for text, cat, pri in examples:
        lines += [f'Ticket: "{text}"', f"Answer: {cat} | {pri}", ""]
    return "\n".join(lines)


def build_variants():
    v1 = (PROMPTS / "v1.txt").read_text(encoding="utf-8")
    wrong = [(EXAMPLES[0][0], "Billing", "Low")] + EXAMPLES[1:3]          # first example labelled WRONG
    variants = {
        "v1_zero_shot_full": v1,
        "v2_zero_shot_no_format": (PROMPTS / "v2.txt").read_text(encoding="utf-8"),
        "v3_zero_shot_no_constraints": (PROMPTS / "v3.txt").read_text(encoding="utf-8"),
        "v4_few_shot_2": v1 + examples_block(EXAMPLES[:2]),
        "v5_few_shot_3": v1 + examples_block(EXAMPLES[:3]),
        "v6_few_shot_5": v1 + examples_block(EXAMPLES),
        "v7_few_shot_5_reversed": v1 + examples_block(EXAMPLES[::-1]),
        "v8_few_shot_3_one_wrong": v1 + examples_block(wrong),
    }
    for name, text in variants.items():                                    # save every prompt version
        if name.startswith(("v4", "v5", "v6", "v7", "v8")):
            (PROMPTS / f"{name}.txt").write_text(text, encoding="utf-8")
    return variants


USER_MESSAGE = "Classify these support tickets:\n\n" + "\n".join(
    f"Ticket {i}: {text}" for i, (text, _, _) in enumerate(TICKETS, 1))


def parse(answer, n=len(TICKETS)):
    """Find each ticket number in the answer, then the first category and priority word after it."""
    starts, pos = [], 0
    for i in range(1, n + 1):
        m = re.compile(rf"(?im)^[\s#>*\-|]*(?:ticket\s*#?\s*)?{i}\b").search(answer, pos)
        starts.append(m.start() if m else None)
        if m:
            pos = m.end()
    preds = []
    for i, s in enumerate(starts):
        if s is None:
            preds.append((None, None))
            continue
        nxt = [x for x in starts[i + 1:] if x is not None]
        seg = answer[s: nxt[0] if nxt else len(answer)]

        def first(words):
            hits = [(m.start(), w) for w in words for m in [re.search(rf"\b{w}\b", seg, re.I)] if m]
            return min(hits)[1] if hits else None
        preds.append((first(CATEGORIES), first(PRIORITIES)))
    return preds


def score(preds):
    cat = sum(p[0] == t[1] for p, t in zip(preds, TICKETS))
    pri = sum(p[1] == t[2] for p, t in zip(preds, TICKETS))
    both = sum(p[0] == t[1] and p[1] == t[2] for p, t in zip(preds, TICKETS))
    return cat, pri, both


def main():
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    variants = build_variants()
    only = sys.argv[1:]                                   # optional: run only some variants
    rows = []
    for name, system in variants.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        out_file = RESULTS / f"{name}.txt"
        if out_file.exists():                             # already done -> reuse (lets you resume)
            answer = out_file.read_text(encoding="utf-8")
            in_tok, secs = None, None
            print(f"{name}: using saved answer")
        else:
            from llm import ask
            print(f"{name}: calling the model...")
            answer, in_tok, _, secs = ask(USER_MESSAGE, system=system)
            out_file.write_text(answer, encoding="utf-8")
        est = len(enc.encode(system)) + len(enc.encode(USER_MESSAGE))
        tokens = in_tok or est
        cat, pri, both = score(parse(answer))
        rows.append((name, cat, pri, both, tokens))
        print(f"  category {cat}/10 | priority {pri}/10 | both {both}/10 | input tokens {tokens}"
              + (f" | {secs:.1f} s" if secs else ""))

    print("\n" + "=" * 78)
    print(f"{'Prompt version':30} {'Category':>9} {'Priority':>9} {'Both':>6} {'Input tokens/call':>18}")
    for name, cat, pri, both, tokens in rows:
        print(f"{name:30} {cat:>7}/10 {pri:>7}/10 {both:>4}/10 {tokens:>18}")

    print("\nPer-ticket predictions (label -> prediction):")
    for name in [r[0] for r in rows]:
        preds = parse((RESULTS / f"{name}.txt").read_text(encoding="utf-8"))
        wrong = [f"T{i}: {t[1]}/{t[2]} -> {p[0]}/{p[1]}" for i, (p, t) in enumerate(zip(preds, TICKETS), 1)
                 if (p[0], p[1]) != (t[1], t[2])]
        print(f"  {name}: " + ("all correct" if not wrong else "; ".join(wrong)))


if __name__ == "__main__":
    main()