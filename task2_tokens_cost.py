"""Week 4 Day 1 - Task 2: count tokens with tiktoken and estimate API cost."""
import json

import tiktoken

enc = tiktoken.get_encoding("o200k_base")     # tokenizer used by current OpenAI GPT models

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

PAGE = " ".join([PARAGRAPH] * 4)    # about one page of text (~480 words)

JSON_OBJ = json.dumps({
    "ticket_id": "T10234",
    "customer": {"name": "Ravi Kumar", "email": "ravi@example.com", "city": "Chennai"},
    "category": "Billing",
    "priority": "High",
    "message": "I was charged twice for my Pro subscription this month.",
    "amount_inr": 1499,
    "created_at": "2026-10-23T10:15:00+05:30",
    "tags": ["refund", "duplicate-charge"],
}, indent=2)

TAMIL = "உங்கள் பில்லிங் கேள்விக்கு எங்கள் உதவிக் குழு இரண்டு வேலை நாட்களுக்குள் பதிலளிக்கும்."

REPORT = " ".join([PAGE] * 20)      # extra: a 20-page report, to test the 8k limit

TEXTS = {"Sentence": SENTENCE, "Paragraph": PARAGRAPH, "Page": PAGE,
         "JSON object": JSON_OBJ, "Tamil sentence": TAMIL, "20-page report": REPORT}


def count_tokens(text):
    return len(enc.encode(text))


def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    """Prices are in US dollars per 1 million tokens."""
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


def main():
    print("=" * 72 + "\nTOKEN COUNTS (tiktoken o200k_base)\n" + "=" * 72)
    print(f"{'Text':15} {'Words':>6} {'Tokens':>7} {'Words/token':>12}   Fits 8k?  Fits 128k?")
    for name, text in TEXTS.items():
        words, tokens = len(text.split()), count_tokens(text)
        print(f"{name:15} {words:6} {tokens:7} {words / tokens:12.2f}   "
              f"{'yes' if tokens <= 8_000 else 'NO':8}  {'yes' if tokens <= 128_000 else 'NO'}")
    print("Rule of thumb: about 0.75 words per token (1 token ~ 4 characters of English)")

    print("\nExample split of the sentence into tokens:")
    print([enc.decode([t]) for t in enc.encode(SENTENCE)])

    print("\n" + "=" * 72 + "\nMONTHLY COST: 1000 chats/day x 30 days, 1500 input + 400 output tokens/chat\n"
          + "=" * 72)
    chats = 1000 * 30
    in_tok, out_tok = chats * 1500, chats * 400
    print(f"Monthly tokens: {in_tok:,} input + {out_tok:,} output")
    prices = {                                  # USD per 1M tokens, official pages, 23 Oct 2026
        "OpenAI GPT-6 Luna": (0.10, 0.50),
        "OpenAI GPT-6.1 Sol": (2.00, 10.00),
        "Google Gemini 3.8 Flash": (0.75, 3.75),
        "Anthropic Claude Sonnet 5.5": (2.00, 10.00),
    }
    for model, (p_in, p_out) in prices.items():
        month = estimate_cost(in_tok, out_tok, p_in, p_out)
        print(f"  {model:28} ${p_in:>5.2f} in / ${p_out:>5.2f} out  ->  ${month:8,.2f} per month")


if __name__ == "__main__":
    main()