"""Week 4 Day 1 (part 2) - Task 1: Capabilities lab."""
from pathlib import Path

from llm import MODEL, ask, md

BASE = Path(__file__).parent

SAMPLE = (
    "Hi team, my name is Priya Sharma and I run a small bakery in Coimbatore. On 14 October I paid "
    "Rs. 2,499 for the yearly Pro plan of your billing software, but the invoice still shows the Free "
    "plan and the reports page crashes every time I open it. I have 3 staff who need access before "
    "Diwali. Please fix this urgently or refund my money. You can call me on 98400 12345."
)

TASKS = [
    ("Summarisation",
     f"Summarise this customer message in exactly 2 short sentences:\n\n{SAMPLE}"),
    ("Classification",
     "Classify this support ticket into ONE category from [Billing, Technical, Account, Shipping, General] "
     f"and ONE priority from [High, Medium, Low]. Answer as 'Category: X, Priority: Y' only.\n\n{SAMPLE}"),
    ("Extraction",
     "Extract these fields from the message and return valid JSON only, with keys: "
     f"name, city, amount_inr, payment_date, plan, phone, issues (list).\n\n{SAMPLE}"),
    ("Translation",
     "Translate this reply into Tamil: 'Thank you for your patience. We have activated your Pro plan "
     "and our technical team will fix the reports page within 24 hours.'"),
    ("Code generation",
     "Write a Python function is_valid_indian_mobile(number: str) -> bool that returns True for a "
     "10-digit Indian mobile number starting with 6, 7, 8 or 9 (spaces and an optional +91 prefix allowed). "
     "Include 4 example assert tests. Code only."),
]


def main():
    rows = []
    for task, prompt in TASKS:
        print(f"Running {task}...")
        answer, in_tok, out_tok, secs = ask(prompt)
        rows.append((task, prompt, answer, in_tok, out_tok, secs))
        print(f"  {in_tok} input + {out_tok} output tokens | {secs:.1f} s")
        print("  " + answer.strip().replace("\n", "\n  ")[:600] + "\n")

    lines = [f"# Capabilities lab results ({MODEL})", "",
             "| Task | Prompt | Answer | Rating (1-5) | Tokens (in + out) | Time (s) |",
             "|---|---|---|---|---|---|"]
    for task, prompt, answer, in_tok, out_tok, secs in rows:
        lines.append(f"| {task} | {md(prompt)} | {md(answer)} | ? | {in_tok} + {out_tok} | {secs:.1f} |")
    (BASE / "capabilities_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved capabilities_results.md")


if __name__ == "__main__":
    main()