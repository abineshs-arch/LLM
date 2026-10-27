---

# Week 4 - Day 3 (Part 2): Chain-of-thought, sampling, structured output, fixing weak prompts

Script: `prompting_lab2.py` - all tables are generated automatically in `day3_part2_results.md`.
Model: **Qwen2.5-0.5B-Instruct**, run locally on the CPU with Hugging Face transformers.
Why a local model: the Python connection to the Gemini API was blocked on my network, and Gemini 3.8 Flash
ignores temperature/top_p, so a local model was the only way to really change those settings.

## Task 1 - Chain-of-thought and sampling

### Chain-of-thought (5 word problems, greedy decoding)
| Mode | Correct | Output tokens (all 5) |
|---|---|---|
| Direct answer ("only the final number") | **1/5** | 308 |
| "Think step by step" | **3/5** | 1,356 |

Step-by-step reasoning tripled the accuracy but used ~4x more tokens. Two CoT answers were cut off at the
300-token limit before the final line - long reasoning also needs a bigger output budget.

### Temperature (same slogan prompt, 5 runs each, top_p = 1.0)
| Temperature | Different answers | What the answers looked like |
|---|---|---|
| 0 | 1/5 | Exactly the same slogan 5 times (deterministic) |
| 0.7 | 5/5 | Varied and mostly good; 2 drifted into Chinese |
| 1.2 | 5/5 | Random, often nonsense ("Brew deep leather planes over chocolate ships") |

### Top-p (temperature 1.0, 5 runs each)
| top_p | Different answers | What the answers looked like |
|---|---|---|
| 0.3 | 5/5 | Different but safe and similar ("Chennai's Filter-Cafe: Fresh Coffee, ...") - all on topic |
| 0.95 | 5/5 | Wild and mixed-language - many unlikely words allowed |

Temperature changes how random every choice is; top_p cuts off the unlikely tail of words.
Low top_p keeps creativity "safe" even at a high temperature.

### Recommended settings
| Task | Temperature | top_p | Other |
|---|---|---|---|
| **Extraction** (JSON fields, labels) | **0** | 1.0 | JSON mode / structured output, no CoT, small max tokens |
| **Summarisation** | **0.2 - 0.3** | 0.9 | Limit length ("3 bullet points"), no CoT |
| **Brainstorming** | **0.8 - 1.0** | 0.9 - 0.95 | Ask for several ideas in one call; avoid > 1.2 (nonsense) |

### When chain-of-thought is a waste of tokens
- Simple lookups, extraction, classification and formatting - there is nothing to reason about.
- With reasoning models that already "think" internally (Gemini 3.8 Flash got all arithmetic right on Day 1 without being asked).
- When only the final answer is used and every token costs money or latency (CoT here was ~4x the tokens).
- Use CoT for multi-step maths, logic and planning, especially with small models.

## Task 2 - Structured output and common mistakes

### Plain prompting vs JSON mode (5 resumes, schema `{name, skills[], years_experience}`)
Every output was checked with `json.loads` and a Pydantic model (`name: str`, `skills: list[str]`, `years_experience: float`).

| Method | Invalid outputs |
|---|---|
| Plain prompting ("...return JSON") | **5/5 invalid** - JSON wrapped in prose or ```json fences |
| JSON mode | **0/5 invalid** - all passed json.loads and Pydantic |

JSON mode locally = strict "You are a JSON API" system prompt + the answer is forced to start with `{"name": "`
+ the text is cut at the matching closing brace. Hosted APIs do this even more strictly with `response_format`
/ Structured outputs, which constrain every generated token to the schema.

### 4 bad prompts and their rewrites
| Mistake | Bad prompt | Rewritten prompt | Bad result | Rewritten result |
|---|---|---|---|---|
| Vague instruction | "Tell me about this ticket: ..." | "Classify into exactly one of [Billing, Technical, Account, Shipping, General]. Reply with the category only." | Long apology paragraph | "Technical" - short, but wrong (should be Shipping) |
| No output format | "Find the order number and city in this ticket" | "Return only JSON like {"order_id": "...", "city": "..."}" | A sentence | Explained steps instead of JSON |
| No examples | "Convert 'March 5th, 2026' to our standard date format" | Same + 2 examples (January 12th, 2025 -> 2025-01-12, ...) | Long explanation | **"2026-03-05" - correct** |
| Conflicting instructions | "Summarise in one word. Give a detailed explanation with three reasons." | "Sentiment? Reply with one word: positive, negative or neutral." | Long summary | "Neutral" - one word, but wrong (should be negative) |

**Bad prompts passed 0/4, rewritten prompts passed 1/4.** The rewrites fixed the *shape* of the answer
(3 of 4 became one short answer instead of a paragraph), and the example-based rewrite was fully correct.
The tiny 0.5B model still made wrong judgments - a better prompt fixes format and focus but cannot fully fix a
weak model. Next steps: add category definitions to the classifier prompt, use JSON mode for the extraction
prompt (as in Task 2a), or use a larger model.

## Concepts
- **Chain-of-thought:** asking the model to reason step by step before answering; helps multi-step problems, costs tokens.
- **Temperature:** how random the next-token choice is (0 = always the same, high = creative or chaotic).
- **Top-p:** only sample from the smallest set of words whose probabilities add up to p.
- **Task-specific settings:** extraction = deterministic, summarisation = low, brainstorming = high.
- **Structured output / JSON mode:** force the answer into a fixed JSON schema so code can read it.
- **Validation:** always check outputs with `json.loads` and a schema (Pydantic) before using them.
- **Fixing weak prompts:** be specific, give an output format, add examples, remove conflicting instructions.