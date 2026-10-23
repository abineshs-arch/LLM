"""Creates long_text.txt: ~52,000 words of report text with one fact hidden in the middle."""
FILLER = ("The quarterly operations report covers stock levels, delivery schedules, staff rosters and "
          "customer feedback for all branches. Most metrics stayed within the normal range this quarter. ")
NEEDLE = "IMPORTANT: The emergency access code for the Coimbatore warehouse is 7341. "
text = (FILLER * 1000 + NEEDLE + FILLER * 1000 +
        "\n\nQuestion: What is the emergency access code for the Coimbatore warehouse?")
with open("long_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"Saved long_text.txt with {len(text.split()):,} words")