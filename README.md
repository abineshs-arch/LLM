# Week 4 - Day 1: LLM Fundamentals

| Part | Task | File |
|---|---|---|
| 1 | Foundation models comparison | `model_comparison.md` |
| 1 | Tokens and cost | `task2_tokens_cost.py` |
| 2 | Capabilities lab | `task1_capabilities.py`, `capabilities_results.md` |
| 2 | Limitations lab | `task2_limitations.py`, `make_long_text.py`, `limitations_results.md` |
| 2 | Gemini API helper | `llm.py` (API key in `.env`, which is in `.gitignore`) |

---

# Part 1

## Task 2 - Tokens and cost (tiktoken `o200k_base`)

| Text | Words | Tokens | Words / token | Fits 8k? | Fits 128k? |
|---|---|---|---|---|---|
| Sentence | 13 | 14 | 0.93 | yes | yes |
| Paragraph | 122 | 139 | 0.88 | yes | yes |
| Page | 488 | 556 | 0.88 | yes | yes |
| JSON object | 38 | 125 | 0.30 | yes | yes |
| Tamil sentence | 10 | 26 | 0.38 | yes | yes |
| 20-page report | 9,760 | 11,120 | 0.88 | **no** | yes |

- **English text: 0.88-0.93 words/token** - better than the 0.75 rule of thumb because the modern `o200k_base` tokenizer has a ~200k vocabulary.
- **JSON: 0.30** - braces, quotes and keys each become tokens, so structured data costs ~3x more per word.
- **Tamil: 0.38** - non-English script is split into smaller pieces, so it costs more than 2x the tokens.

`estimate_cost(input_tokens, output_tokens, price_in, price_out)` with prices in USD per 1 million tokens.

**Monthly cost of a support bot** (1000 chats/day x 30 days, 1500 input + 400 output tokens per chat
= 45,000,000 input + 12,000,000 output tokens):

| Model | Input $/1M | Output $/1M | Monthly cost |
|---|---|---|---|
| OpenAI GPT-6 Luna | 0.10 | 0.50 | **$10.50** |
| Google Gemini 3.8 Flash | 0.75 | 3.75 | $78.75 |
| OpenAI GPT-6.1 Sol | 2.00 | 10.00 | $210.00 |
| Anthropic Claude Sonnet 5.5 | 2.00 | 10.00 | $210.00 |

Prices from the providers' official pricing pages on 23 October 2026.

---

# Part 2: LLM capabilities and limitations

Model: **Gemini 3.8 Flash** (`gemini-3.8-flash`, free tier, thinking level Low).

## Task 1 - Capabilities lab
Sample text: a customer message from Priya Sharma (Coimbatore bakery) who paid Rs. 2,499 for the Pro plan,
still sees the Free plan, and the reports page crashes.

| Task | Prompt | Answer (short) | Rating (1-5) | Tokens (in + out) | Time |
|---|---|---|---|---|---|
| Summarisation | Summarise in exactly 2 short sentences | "Priya Sharma paid for the yearly Pro plan, but her account still shows the Free plan and the reports page crashes. She urgently requests a fix before Diwali or a full refund." | 4 | 113 + 474 | 149.4 s |
| Classification | One category + one priority, format 'Category: X, Priority: Y' | `Category: Billing, Priority: High` | 5 | 145 + 155 | 172.5 s |
| Extraction | JSON with name, city, amount_inr, payment_date, plan, phone, issues | Correct JSON: Priya Sharma, Coimbatore, 2499, "14 October", "yearly Pro", "98400 12345", 2 issues | 4 | 137 + 111 | 159.2 s |
| Translation | Translate a reply into Tamil | Tamil translation + extra transliteration and a note | 3 | 35 + 162 | 128.3 s |
| Code generation | `is_valid_indian_mobile()` + 4 asserts | Correct regex `^(?:\+91)?[6-9]\d{9}$` and 4 correct asserts | 5 | 66 + 169 | 154.1 s |

Output tokens include the model's internal "thinking" tokens.

**What I would improve**
- Summarisation: keep amounts and deadlines; it said "account" instead of "invoice".
- Classification: already perfect - keep the fixed format and label list.
- Extraction: ask for raw JSON (no ``` fences) and ISO dates (YYYY-MM-DD), then check with `json.loads`.
- Translation: say "return only the Tamil text" and have a Tamil speaker check it.
- Code generation: also handle hyphens and a leading 0, and run the asserts automatically.

**Best and worst task**
1. Best: classification - a short answer from a fixed list with an exact format leaves little room for mistakes.
2. Code generation was equally strong because it was precise and could be tested.
3. Worst: translation - it ignored "only translate" and added extra text; Tamil also has less training data.
4. Constrained tasks with a clear format work best; open tasks need tighter instructions and checking.

**Speed:** each API call took ~130-170 s from my network (normally a few seconds).

## Task 2 - Limitations lab
Experiments 1 and 2a ran with `task2_limitations.py`. The Python connection then became too slow, so 2b, 3 and 4
were run with the same model in the **Google AI Studio chat**, with **Google Search grounding turned OFF**.

| # | Experiment | Correct answer | Model answer | Tokens | Result |
|---|---|---|---|---|---|
| 1 | Knowledge cutoff: "today's date and this week's tech news" | Model cannot know | "I do not have access to real-time information... cannot provide today's date or this week's news" | 21 + 46 | Admitted the limit |
| 2a | Hallucination: summarise a made-up 2019 NeurIPS paper | Paper does not exist | "**No such paper exists**" + explanation | 43 + 227 | Did not invent details |
| 2b | Same + "say I don't know if unsure" | Paper does not exist | "**I don't know.**" | ~66 | Short, honest answer |
| 3 | Context length: 52,000 words with a code hidden in the middle | 7341 | "The emergency access code... is **7341**" | ~60,060 input | Found the fact |
| 4a | Arithmetic: 23 pens x Rs 17, 15% discount, 18% GST | 391 -> 332.35 -> **392.17** | Same steps -> **Rs 392.17** | ~432 | Correct |
| 4b | Logic: 5 friends in a row, who is in the middle? | **Dev** | Esha, Anu, **Dev**, Babu, Chitra | ~254 | Correct |

**What this shows**
- Knowledge cutoff: no live data, so the app must pass the date and fresh information.
- Hallucination: this model refused to invent the paper, but models can still hallucinate, so we cannot rely on it.
- Context: 60k tokens would not fit in an 8k window at all, and long inputs cost more.
- Reasoning: correct here, but longer calculations can go wrong, so numbers must be checked by code.

**5 safeguards for a business application**
1. Ground answers in real data (RAG) and pass today's date in the prompt.
2. Allow "I don't know" and hand unclear or sensitive cases to a human.
3. Calculate prices, tax and refunds with code, not the LLM.
4. Validate every output (JSON parses, labels allowed, values in range); retry or reject otherwise.
5. Send only relevant context, set timeouts and retries, log everything, and require human approval for refunds or account changes.

## Concepts
- **LLM capabilities:** one model can summarise, classify, extract, translate and write code from a prompt.
- **Qualitative evaluation:** rate each answer 1-5 and record tokens and time.
- **Knowledge cutoff:** the model only knows its training data.
- **Hallucination:** confident but false facts or references.
- **Context limits:** only a limited number of tokens fit; long inputs cost more.
- **Reasoning errors:** multi-step maths and logic can go wrong.
- **Safeguards:** rules that keep a business app safe despite these limits.