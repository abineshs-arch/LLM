# Week 4 - Day 1: LLM Fundamentals

| Task | File |
|---|---|
| 1. Foundation models comparison | `model_comparison.md` |
| 2. Tokens and cost | `task2_tokens_cost.py` |

## Task 2 - Tokens and cost (tiktoken `o200k_base`)

| Text | Words | Tokens | Words / token | Fits 8k? | Fits 128k? |
|---|---|---|---|---|---|
| Sentence | 13 | 14 | 0.93 | yes | yes |
| Paragraph | 122 | 139 | 0.88 | yes | yes |
| Page | 488 | 556 | 0.88 | yes | yes |
| JSON object | 38 | 125 | 0.30 | yes | yes |
| Tamil sentence | 10 | 26 | 0.38 | yes | yes |
| 20-page report | 9,760 | 11,120 | 0.88 | **no** | yes |

Token split of the sentence:
`['Our', ' support', ' team', ' will', ' reply', ' to', ' your', ' billing', ' question', ' within', ' two', ' working', ' days', '.']`

### Comparison with the rule of thumb (~0.75 words per token)
- **English text: 0.88-0.93 words/token** - better than 0.75 because the modern `o200k_base` tokenizer has a
  ~200k vocabulary, so most common English words are a single token.
- **JSON: 0.30** - braces, quotes, colons and keys each become separate tokens, so structured data costs ~3x more per word.
- **Tamil: 0.38** - non-English scripts are split into smaller pieces, so the same message costs more than 2x the tokens.
- A 20-page report (11,120 tokens) does not fit an 8k context window but fits easily in 128k.

### Cost function
`estimate_cost(input_tokens, output_tokens, price_in, price_out)` with prices in USD per 1 million tokens:
`cost = input_tokens / 1M * price_in + output_tokens / 1M * price_out`

### Monthly cost of a support bot
1000 chats/day x 30 days, 1500 input + 400 output tokens per chat
= **45,000,000 input + 12,000,000 output tokens per month**

| Model | Input $/1M | Output $/1M | Monthly cost |
|---|---|---|---|
| OpenAI GPT-6 Luna | 0.10 | 0.50 | **$10.50** |
| Google Gemini 3.8 Flash | 0.75 | 3.75 | $78.75 |
| OpenAI GPT-6.1 Sol | 2.00 | 10.00 | $210.00 |
| Anthropic Claude Sonnet 5.5 | 2.00 | 10.00 | $210.00 |

Prices from the providers' official pricing pages on 23 October 2026 (Gemini 3.8 Flash price rises to $1.50 / $7.50 from January 2027).
A small model is ~20x cheaper for the same traffic; input tokens are 79% of the volume, so keeping prompts short saves the most money.

Sources:
- https://developers.openai.com/api/docs/models/compare
- https://platform.claude.com/docs/en/about-claude/models/overview
- https://ai.google.dev/gemini-api/docs/pricing