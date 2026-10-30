# Week 4 Assessment – LLM Fundamentals

All experiments ran locally on CPU with **Qwen/Qwen2.5-1.5B-Instruct** (temperature 0 unless stated).
Every model answer is cached in `results/cache.json`; full result tables are in `results/`.

| File | Task |
|---|---|
| `llm_local.py` | Shared helper: loads the local model, `ask()` with caching, token counts and timing |
| `task1_models_tokens.py` | Task 1 – token counts, `estimate_cost`, monthly cost, context limits |
| `task2_capabilities_limits.py` | Task 2 – 11 capability / limitation tests → `results/task2_results.md` |
| `task3_transformer.py` | Task 3 – attention, multi-head, positional encoding, BERT vs GPT-2 |
| `task4.toolkit.py` | Task 4 – golden set, prompt versions, CoT, sampling, JSON mode, bad prompts → `results/task4_results.md` |
| `prompts/classify_v1/v2/v3.txt` | The 3 classification prompt versions |
| `golden_tickets.json`, `golden_resumes.json` | Golden sets with my own labels |

---

## Task 1 – Models, tokens and cost

### Model comparison (checked 23 Oct 2026 on official sites)

| Family | Owner | Weights | Context window | Strengths | Typical use |
|---|---|---|---|---|---|
| GPT (GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna) | OpenAI | Closed | ~1.05M | Reasoning, coding, tools, very cheap small model | Chatbots, coding, agents |
| Claude (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5) | Anthropic | Closed | 1M | Long documents, instruction following, coding, safety | Document analysis, support, agents |
| Llama (Llama 4 Scout, Maverick) | Meta | **Open weights** | Up to 10M (Scout) | Self-hosting, fine-tuning, no per-token fee | On-premise / private apps |
| Gemini (Gemini 4 Argon, 3.8 Flash) | Google | Closed | 1M | Native multimodal, long context, cheap Flash | Multimodal and Google Cloud apps |

Sources:
- https://developers.openai.com/api/docs/models/compare
- https://platform.claude.com/docs/en/about-claude/models/overview
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.meta.com/blog/llama-4-multimodal-intelligence/

### Token counts (tiktoken `o200k_base`)

| Text | Words | Tokens | Words/token | Fits 8k | Fits 128k |
|---|---|---|---|---|---|
| Sentence | 13 | 14 | 0.93 | yes | yes |
| Paragraph | 122 | 139 | 0.88 | yes | yes |
| Page | 488 | 556 | 0.88 | yes | yes |
| JSON object | 38 | 125 | 0.30 | yes | yes |
| Tamil sentence | 10 | 26 | 0.38 | yes | yes |
| 20-page report | 9,760 | 11,120 | 0.88 | **NO** | yes |

English is about 0.9 words per token; JSON (punctuation) and Tamil (non-Latin script) use **about 3× more tokens per word**, so they cost more and fill the context faster.

### Monthly cost – 1000 chats/day × 30 days × (1500 input + 400 output tokens) = 45M input + 12M output

`estimate_cost(in, out, price_in, price_out) = in/1M × price_in + out/1M × price_out`

| Model | $ per 1M in / out | Monthly cost |
|---|---|---|
| OpenAI GPT-6 Luna | 0.10 / 0.50 | **$10.50** |
| Google Gemini 3.8 Flash | 0.75 / 3.75 | $78.75 |
| OpenAI GPT-6.1 Sol | 2.00 / 10.00 | $210.00 |
| Anthropic Claude Sonnet 5.5 | 2.00 / 10.00 | $210.00 |

### Recommendation
1. **Public customer-support bot:** GPT-6 Luna (or Claude Haiku 5.5 / Gemini 3.8 Flash) – a small, fast hosted model.
2. High volume and short chats mean cost and speed matter most; Luna costs only ~$10.50/month for this load.
3. A hosted API needs no GPU servers; difficult chats can be escalated to a bigger model or a human.
4. **On-premise internal tool:** Llama 4 (open weights) – runs on company servers, so confidential data never leaves the network.
5. No per-token fee (only hardware cost), it can be fine-tuned on internal documents, and Scout supports very long context.

---

## Task 2 – Capabilities and limitations (Qwen2.5-1.5B, temperature 0)

| Test | Prompt | Result | Rating (1-5) |
|---|---|---|---|
| Summarisation | Summarise the Priya Sharma bakery complaint in 2 sentences | First summary correct, but it added a second summary that wrongly says she paid for the Free plan | 3 |
| Classification | Category [Billing/Technical/Account] + priority | `Technical, High` (expected Billing, High – the ticket mixes a payment and a crash) | 3 |
| Extraction | name, city, amount_inr, payment_date, plan, phone as JSON | All 6 values correct, but wrapped in a ```json block, added extra fields and "+91" | 4 |
| Translation | English sentence → Tamil | Output looks garbled / not a correct Tamil sentence | 2 |
| Code generation | `is_valid_indian_mobile()` | Long docstring, ran out of 200 tokens before finishing; `len != 10` check rejects +91 numbers | 2 |
| Knowledge cutoff | Today's date and this week's tech news | Correctly said it has no real-time access and suggested news sites | 5 |
| Hallucination (plain) | Summarise a **fake** 2019 NeurIPS paper | Did not invent the paper, but drifted into general quantum-computing text | 4 |
| Hallucination + "say I don't know" | Same + "If not sure, say I don't know" | `I don't know.` | 5 |
| Long context (needle) | 7,506-token report with a code in the middle | `The emergency access code ... is 7341.` – correct | 5 |
| Multi-step arithmetic | 23 pens × Rs 17, −15%, +18% GST | Correct first step (391) but hit the 300-token limit before the final answer (392.17) | 2 |
| Logic puzzle | 5 friends in a row – who is in the middle? | Wrong reasoning ("Anu cannot be in second or third seat"), no answer (correct: **Dev**) | 1 |

**Observations:** the small model is good at extraction, refusing unknown facts and finding a fact in long context, but weak at multi-step maths, logic, Tamil and long code. Adding "say I don't know" turned a risky answer into a safe one.

### 5 safeguards for production
1. **Ground answers in our own data (RAG)** and tell the model to say "I don't know" when the answer is not in the context.
2. **Validate structured output** with JSON mode + Pydantic, and retry or fall back when validation fails.
3. **Never trust LLM maths** – do calculations (prices, GST, discounts) in Python code / tools, not in the model.
4. **Human in the loop** for refunds, legal, or low-confidence answers, and log every prompt/response for review.
5. **Evaluate before every change** with a golden test set (accuracy, hallucination checks) and set `max_tokens`, timeouts and cost limits.

---

## Task 3 – Transformer from scratch (NumPy)

**A. Single-head self-attention (4 tokens, size 3)** – `softmax(QKᵀ/√d) · V`
```
[[0.001 0.999 0.    0.   ]
 [0.257 0.    0.    0.743]
 [0.265 0.335 0.099 0.3  ]
 [0.    1.    0.    0.   ]]   each row sums to 1.0: True
```

**B. Two-head attention** (d_model 4 = 2 heads × 2): each head's weights sum to 1 per row; the head outputs are concatenated and multiplied by the output projection `Wo` → a 4×4 output.

**C. Sinusoidal positional encoding (10 × 8)** – saved as `positional_encoding.png`. Early columns change fast (sin/cos of high frequency), later columns change slowly, so every position gets a unique pattern.

**D. Encoder vs decoder**
- BERT fill-mask: *"The customer asked for a [MASK] because the product arrived broken."* → check 0.043, fee 0.043, discount 0.026, price 0.025, copy 0.016
- GPT-2: *"Thank you for contacting support. Your refund is not processed until we receive the results of the refund or exchange. Sorry, the payment type chosen is invalid for this event."*

### Encoder vs decoder, and why transformers replaced RNNs
1. An **encoder** (BERT) sees the whole sentence in both directions, so it is best for understanding: classification, NER, filling a masked word.
2. A **decoder** (GPT) uses causal masking – each token sees only earlier tokens – so it is built for generating text one token at a time.
3. RNNs read one token after another, so training cannot be parallelised and is slow on long texts.
4. RNNs also forget information from far back (vanishing gradients), even with LSTMs.
5. Self-attention connects every token to every other token directly in one step, so long-range links are easy and training runs in parallel on GPUs.
6. Positional encoding adds the word order back, and this parallel design is what made scaling to today's large language models possible.

---

## Task 4 – Prompt engineering toolkit

Golden sets: 15 support tickets (Billing / Technical / Account + High / Medium / Low) and 5 resumes with expected JSON – labelled by me.

### A. Classification prompt versions (15 tickets, temperature 0)

| Version | Category correct | Priority correct | Both correct | Avg input tokens |
|---|---|---|---|---|
| v1 basic (one line) | 7/15 | 7/15 | **3/15 (20%)** | ~55 |
| v2 Role / Task / Constraints / Output | 13/15 | 10/15 | **8/15 (53%)** | ~200 |
| v3 v2 + 3 few-shot examples | **15/15** | 9/15 | **9/15 (60%)** | ~284 |

- v1 often gave no category name at all (None) or confused Billing with Technical.
- v2 fixed most categories; its 2 category mistakes were login tickets classed as Technical.
- v3 got **every category right**; the remaining mistakes are all **priority** (it rates too many tickets High) – the priority definitions need more examples.
- Trade-off: better accuracy costs ~5× more input tokens per call.

### B. Chain-of-thought (5 word problems)

| Mode | Correct | Output tokens | Answers (expected 32, 7500, 270, 78, 36) |
|---|---|---|---|
| Direct – final number only | 1/5 | 17 | 76, 150, 310, 48, 36 |
| "Think step by step" | **5/5** | 1,094 | 32, 7500, 270, 78, 36 |

Step-by-step reasoning fixed every answer but used ~64× more output tokens (slower and more expensive).

### C. Temperature and top-p (slogan for a Chennai filter-coffee shop, 5 runs each)

| Setting | Different answers (of 5) | What happened |
|---|---|---|
| temperature 0 | 1 | Same slogan every time – repeatable |
| temperature 0.7 | 5 | Varied and still sensible – best for creative text |
| temperature 1.2 | 5 | Very random – invented names ("JCharlotte") and stray foreign characters |
| T 1.0, top_p 0.3 | 5 | Small variations of the same idea (only top words allowed) |
| T 1.0, top_p 0.95 | 5 | Wider variety, mostly sensible |

Use temperature 0 for classification/extraction, about 0.7 for creative writing.

### D. Structured output – {name, skills[], years_experience}, validated with Pydantic

| Method | Invalid outputs (of 5) | Name and years correct |
|---|---|---|
| Plain prompting ("return JSON") | **5/5 invalid** – wrapped in ```json code fences | – |
| JSON mode (system rule + prefill `{"name": "` + extract first JSON object) | **0/5 invalid** | 5/5 |

### E. Three bad prompts and the rewrites

| Mistake | Bad prompt | Rewritten prompt | Result |
|---|---|---|---|
| Vague instruction | "Tell me about this ticket" → long free text | "Classify into one of [Billing, Technical, Account, Shipping], reply with the name only" | fail → **PASS** (`Shipping`) |
| No output format | "Find the order number and city" → free sentence | "Return ONLY this JSON: {order_id, city}" | fail → **PASS** (`45120`, `Coimbatore`) |
| Conflicting instructions | "Summarise in one word. Give a detailed explanation with 3 reasons." | "Reply with one word: positive, negative or neutral" | fail → **PASS** (`Negative`) |

**Bad prompts passed 0/3 – rewritten prompts passed 3/3.**

## How to run
```
pip install tiktoken numpy matplotlib torch transformers pydantic
python week4_assessment/task1_models_tokens.py
python week4_assessment/task2_capabilities_limits.py
python week4_assessment/task3_transformer.py
python week4_assessment/task4.toolkit.py
```
No API key is needed – everything runs locally. (`.env` is never committed.)