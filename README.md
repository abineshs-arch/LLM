---

# Week 4 - Day 3: Prompt engineering - system prompts, zero-shot and few-shot

Model: Gemini 3.8 Flash (thinking level Low). The Python API connection was blocked on my network
(WinError 10053), so every prompt was run in Google AI Studio and the answers were saved in `results/`.
`prompt_lab.py` builds the prompt versions, scores the saved answers against my labels and counts input
tokens with tiktoken (`o200k_base`). All 10 tickets are sent in one call per prompt version.

## Task 1 - System prompt and zero-shot
- `prompts/v1.txt` - full system prompt with **Role, Task, Tone, Constraints, Output format**
- `prompts/v2.txt` - same prompt **without the output format**
- `prompts/v3.txt` - same prompt **without the constraints**
- 10 test tickets with my own labels (several are deliberately ambiguous), stored in `prompt_lab.py`

| Version | Category | Priority | Both correct | What changed in the answer |
|---|---|---|---|---|
| v1 full prompt | 8/10 | 9/10 | **7/10** | Clean `1 \| Billing \| High` lines, easy to parse |
| v2 no output format | 8/10 | 9/10 | **7/10** | Same labels, but a long "Ticket 1: / Category: / Priority:" layout - harder to parse automatically |
| v3 no constraints | 8/10 | 9/10 | **7/10** | Same as v1 |

Zero-shot mistakes (same in all three): T5 student discount -> Billing (should be General),
T10 payment error E-102 -> Billing (should be Technical), T9 cracked screen -> High priority (should be Medium).
For this strong model, removing the format or the constraints did not change the labels; removing the
**output format** changed the **shape** of the answer, which would break a program that reads it.

## Task 2 - Few-shot prompting
Examples were chosen to cover the edge cases (payment word but a bug, discount question, address change for an order,
login problem with no impact, received item that is a device fault). Prompts: `prompts/v4`-`v8`.

| Examples | Category | Priority | Accuracy (both) | Input tokens per call |
|---|---|---|---|---|
| 0 (v1 zero-shot) | 8/10 | 9/10 | 7/10 | 485 |
| **2 (v4)** | 10/10 | 9/10 | **9/10** | **531** |
| 3 (v5) | 10/10 | 9/10 | 9/10 | 553 |
| 5 (v6) | 10/10 | 9/10 | 9/10 | 597 |
| 5, reversed order (v7) | 10/10 | 9/10 | 9/10 | 597 |
| 3, one example labelled wrong (v8) | 9/10 | 9/10 | 9/10 | 553 |

**Example order and a wrong example**
- 2 examples fixed T5 (-> General) and T10 (-> Technical) - exactly the cases the examples covered.
- The "delivery address -> Shipping | Medium" example fixed T9 but pushed T4 (order never arrived) down to Medium.
- **Order matters:** v6 and v7 use the same 5 examples; v6 got T4 wrong, the reversed v7 got T9 wrong instead.
- **A wrong example is copied:** in v8 the checkout error was labelled "Billing | Low", and T10 (payment error E-102)
  was predicted Billing / Medium again - the model followed the bad example.

**Best number of examples: 2.** It reached the best accuracy (9/10) for only +46 tokens (+9%) over zero-shot.
3 and 5 examples cost up to +112 tokens (+23%) per call with no accuracy gain, so they only add cost.

**When few-shot is better than zero-shot**
1. When categories overlap and the model needs to see where the boundary is (e.g. "payment error" = Technical, not Billing).
2. When you need an exact output format or labelling style that is hard to describe in words.
3. When there are known edge cases - one good example fixes them better than a longer rule.
4. Zero-shot is enough for clear tasks; few-shot costs tokens on every call, and bad or badly ordered examples can make results worse.

## Concepts
- **System prompt:** fixed instructions (role, task, tone, constraints, format) sent before the user message.
- **Zero-shot:** the model gets only instructions, no examples.
- **Few-shot:** a few labelled examples are added to the prompt to show the pattern.
- **Prompt versioning:** each prompt version is a separate file in `prompts/`, committed to git, so results can be compared and repeated.
- **Example selection:** examples should cover edge cases, be correctly labelled, and be checked for order effects.
- **Token cost vs accuracy:** more examples = more input tokens on every call; stop adding them when accuracy stops improving.