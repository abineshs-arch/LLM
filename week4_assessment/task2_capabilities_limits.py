"""Task 2: capabilities and limitations lab with the local model (results -> results/task2_results.md)."""
from llm_local import BASE, MODEL, ask, cell

SAMPLE = ("Hi team, my name is Priya Sharma and I run a small bakery in Coimbatore. On 14 October I paid "
          "Rs. 2,499 for the yearly Pro plan of your billing software, but the invoice still shows the Free "
          "plan and the reports page crashes every time I open it. Please fix this urgently or refund my money. "
          "You can call me on 98400 12345.")

FAKE_PAPER = ("Summarise the 2019 NeurIPS paper 'Quantum Gradient Folding for Tamil Sentiment Analysis' "
              "by R. Venkataraman and L. Ortiz. Give its main method and the accuracy it reported.")
FILLER = ("The quarterly operations report covers stock levels, delivery schedules, staff rosters and "
          "customer feedback for all branches. Most metrics stayed within the normal range this quarter. ")
LONG_TEXT = FILLER * 120 + "IMPORTANT: The emergency access code for the Coimbatore warehouse is 7341. " + FILLER * 120

TESTS = [
    # (test, prompt, max_tokens, correct answer / what to check)
    ("Summarisation", f"Summarise this customer message in exactly 2 short sentences:\n\n{SAMPLE}", 120,
     "2 sentences: paid for Pro, still Free plan, reports crash, wants fix or refund"),
    ("Classification", "Classify this ticket into ONE category from [Billing, Technical, Account] and ONE priority "
     f"from [High, Medium, Low]. Answer as 'Category: X, Priority: Y' only.\n\n{SAMPLE}", 30,
     "Category: Billing, Priority: High"),
    ("Extraction", "Extract name, city, amount_inr, payment_date, plan and phone from the message. "
     f"Return valid JSON only.\n\n{SAMPLE}", 150, "Priya Sharma, Coimbatore, 2499, 14 October, Pro, 98400 12345"),
    ("Translation", "Translate into Tamil, return only the Tamil text: 'Thank you for your patience. "
     "We have activated your Pro plan.'", 80, "Correct Tamil, nothing extra"),
    ("Code generation", "Write a Python function is_valid_indian_mobile(number: str) -> bool that returns True for "
     "a 10-digit number starting with 6, 7, 8 or 9 (optional +91 prefix). Code only.", 200,
     "Correct regex / logic"),
    ("Knowledge cutoff", "What is today's date, and what were the 3 biggest technology news stories this week?", 150,
     "Model cannot know - should say so"),
    ("Hallucination (plain)", FAKE_PAPER, 200, "Paper does NOT exist"),
    ("Hallucination + 'say I don't know'", FAKE_PAPER + " If you are not sure this paper exists, say 'I don't know'.",
     200, "Paper does NOT exist -> I don't know"),
    ("Context length (fact in the middle)",
     LONG_TEXT + "\n\nQuestion: What is the emergency access code for the Coimbatore warehouse?", 40, "7341"),
    ("Multi-step arithmetic", "A shop sells pens at Rs 17 each. Ravi buys 23 pens and gets a 15% discount on the total. "
     "Then 18% GST is added to the discounted amount. How much does Ravi pay? Show the steps and give the final "
     "amount rounded to 2 decimals.", 300, "391 -> 332.35 -> 392.17"),
    ("Logic puzzle", "Five friends sit in a row of 5 seats. Esha sits at the left end. Chitra sits at the right end. "
     "Anu sits somewhere to the left of Babu. Dev sits between Anu and Babu. Who sits in the middle seat?", 200,
     "Esha, Anu, Dev, Babu, Chitra -> Dev"),
]


def main():
    print(f"Model: {MODEL} (local, temperature 0) | {len(TESTS)} tests\n")
    lines = [f"# Task 2 results - {MODEL} (local CPU, temperature 0)", "",
             "| Test | Prompt | Result | Expected | Rating (1-5) | Tokens (in + out) | Time (s) |",
             "|---|---|---|---|---|---|---|"]
    for i, (test, prompt, max_tokens, expected) in enumerate(TESTS, 1):
        print(f"{i:2}. {test}")
        r = ask(f"t2_{i}", prompt, max_new_tokens=max_tokens)
        print("    " + r["text"].replace("\n", "\n    ")[:500] + "\n")
        short_prompt = prompt if len(prompt) < 300 else f"[{len(prompt.split()):,} words of report text, fact in the middle] + question"
        lines.append(f"| {test} | {cell(short_prompt, 120)} | {cell(r['text'], 220)} | {expected} | ? | "
                     f"{r['in_tokens']} + {r['out_tokens']} | {r['seconds']} |")
    (BASE / "results" / "task2_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved results/task2_results.md")


if __name__ == "__main__":
    main()