"""Week 4 Day 1 (part 2) - Task 2: Limitations lab."""
from pathlib import Path

from llm import MODEL, ask

BASE = Path(__file__).parent
OUT = BASE / "limitations_results.md"

FAKE_PAPER = ("Summarise the 2019 NeurIPS paper 'Quantum Gradient Folding for Tamil Sentiment Analysis' "
              "by R. Venkataraman and L. Ortiz. Give its main method and the accuracy it reported.")

# A long text (~52,000 words, ~60k tokens) with one fact hidden in the middle
FILLER = ("The quarterly operations report covers stock levels, delivery schedules, staff rosters and "
          "customer feedback for all branches. Most metrics stayed within the normal range this quarter. ")
NEEDLE = "IMPORTANT: The emergency access code for the Coimbatore warehouse is 7341. "
LONG_TEXT = FILLER * 1000 + NEEDLE + FILLER * 1000

ARITHMETIC = ("A shop sells pens at Rs 17 each. Ravi buys 23 pens and gets a 15% discount on the total. "
              "Then 18% GST is added to the discounted amount. How much does Ravi pay? "
              "Show the steps and give the final amount rounded to 2 decimals.")
LOGIC = ("Five friends sit in a row of 5 seats. Esha sits at the left end. Chitra sits at the right end. "
         "Anu sits somewhere to the left of Babu. Dev sits between Anu and Babu. Who sits in the middle seat?")

EXPERIMENTS = [
    ("1. Knowledge cutoff", "What is today's date, and what were the 3 biggest technology news stories this week?",
     "Model does not know today's date or this week's news (no internet access)"),
    ("2a. Hallucination (plain)", FAKE_PAPER, "The paper does NOT exist"),
    ("2b. Hallucination ('say I don't know')", FAKE_PAPER + " If you are not sure this paper exists, say 'I don't know'.",
     "The paper does NOT exist"),
    ("3. Context length (fact in the middle)", LONG_TEXT + "\n\nQuestion: What is the emergency access code for the Coimbatore warehouse?",
     "7341"),
    ("4a. Reasoning: multi-step arithmetic", ARITHMETIC, "17 x 23 = 391 -> x 0.85 = 332.35 -> x 1.18 = 392.17"),
    ("4b. Reasoning: logic puzzle", LOGIC, "Esha 1, Anu 2, Dev 3, Babu 4, Chitra 5 -> Dev"),
]


def main():
    lines = [f"# Limitations lab results ({MODEL})", ""]
    for name, prompt, correct in EXPERIMENTS:
        print(f"Running {name}...")
        try:
            answer, in_tok, out_tok, secs = ask(prompt)
        except Exception as e:
            print(f"  FAILED after retries: {e}\n")
            lines += [f"## {name}", f"**Failed:** {type(e).__name__}", "", "---", ""]
            continue
        short_prompt = prompt if len(prompt) < 400 else f"[{len(prompt.split()):,} words of report text with one fact in the middle] + question"
        print(f"  {in_tok:,} input + {out_tok:,} output tokens | {secs:.1f} s")
        print("  " + answer.strip().replace("\n", "\n  ")[:700] + "\n")
        lines += [f"## {name}", f"**Prompt:** {short_prompt}", "",
                  f"**Correct answer (checked by hand):** {correct}", "",
                  f"**Model answer** ({in_tok:,} input + {out_tok:,} output tokens, {secs:.1f} s):", "",
                  answer.strip(), "", "---", ""]
        OUT.write_text("\n".join(lines), encoding="utf-8")      # save after EVERY experiment
    print("Saved limitations_results.md")


if __name__ == "__main__":
    main()