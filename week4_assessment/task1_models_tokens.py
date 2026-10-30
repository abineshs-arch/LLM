"""Task 1: token counts (tiktoken), words per token, context windows and monthly cost."""
import json

import tiktoken

enc = tiktoken.get_encoding("o200k_base")

SENTENCE = "Our support team will reply to your billing question within two working days."
PARAGRAPH = (
    "Customer support teams receive hundreds of messages every day, and many of them ask the "
    "same questions about passwords, refunds, delivery times and invoices. A chatbot built on a "
    "large language model can answer these common questions instantly, at any time of the day, "
    "and in several languages. It reads the customer's message, looks up the relevant policy or "
    "order details, and writes a clear, polite reply. When the question is unusual, sensitive or "
    "the customer is unhappy, the bot hands the conversation to a human agent together with a short "
    "summary, so the customer does not have to repeat everything. This reduces waiting time, lowers "
    "cost per ticket and lets human agents focus on the difficult cases that really need them."
)
PAGE = " ".join([PARAGRAPH] * 4)
JSON_OBJ = json.dumps({
    "ticket_id": "T10234",
    "customer": {"name": "Ravi Kumar", "email": "ravi@example.com", "city": "Chennai"},
    "category": "Billing", "priority": "High",
    "message": "I was charged twice for my Pro subscription this month.",
    "amount_inr": 1499, "created_at": "2026-10-23T10:15:00+05:30",
    "tags": ["refund", "duplicate-charge"],
}, indent=2)
TAMIL = "உங்கள் பில்லிங் கேள்விக்கு எங்கள் உதவிக் குழு இரண்டு வேலை நாட்களுக்குள் பதிலளிக்கும்."
REPORT = " ".join([PAGE] * 20)            # extra: a 20-page report to test the 8k window

TEXTS = {"Sentence": SENTENCE, "Paragraph": PARAGRAPH, "Page": PAGE, "JSON object": JSON_OBJ,
         "Tamil sentence": TAMIL, "20-page report": REPORT}

# USD per 1M tokens, from the providers' official pricing pages (checked 23 Oct 2026)
PRICES = {"OpenAI GPT-6 Luna": (0.10, 0.50), "Google Gemini 3.8 Flash": (0.75, 3.75),
          "OpenAI GPT-6.1 Sol": (2.00, 10.00), "Anthropic Claude Sonnet 5.5": (2.00, 10.00)}


def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    """Prices are USD per 1 million tokens."""
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


def main():
    print(f"{'Text':15} {'Words':>6} {'Tokens':>7} {'Words/token':>12}  Fits 8k  Fits 128k")
    for name, text in TEXTS.items():
        words, tokens = len(text.split()), len(enc.encode(text))
        print(f"{name:15} {words:6,} {tokens:7,} {words / tokens:12.2f}  {'yes' if tokens <= 8000 else 'NO':7}  "
              f"{'yes' if tokens <= 128000 else 'NO'}")
    print("Rule of thumb: ~0.75 words per token")

    chats = 1000 * 30
    in_tok, out_tok = chats * 1500, chats * 400
    print(f"\nMonthly: 1000 chats/day x 30 days = {in_tok:,} input + {out_tok:,} output tokens")
    for model, (p_in, p_out) in PRICES.items():
        print(f"  {model:28} ${p_in:5.2f} / ${p_out:5.2f}  ->  ${estimate_cost(in_tok, out_tok, p_in, p_out):9,.2f} per month")


if __name__ == "__main__":
    main()