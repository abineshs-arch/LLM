# Task 4 results - Qwen/Qwen2.5-1.5B-Instruct (local CPU)

## A. Classification prompt versions (15 golden tickets, temperature 0)

| Version | Category correct | Priority correct | Both correct (accuracy) | Avg input tokens |
|---|---|---|---|---|
| v1 basic | 7/15 | 7/15 | **3/15 (20%)** | 55 |
| v2 Role/Task/Constraints/Output | 13/15 | 10/15 | **8/15 (53%)** | 199 |
| v3 + 3 few-shot examples | 15/15 | 9/15 | **9/15 (60%)** | 284 |

- **v1 basic** mistakes: T1: Billing/High -> Technical/High; T2: Billing/Low -> Account/High; T4: Billing/High -> Account/High; T5: Billing/Low -> Billing/High; T6: Technical/Medium -> Technical/High; T8: Technical/Low -> None/Medium; T10: Technical/Low -> None/High; T11: Account/High -> None/High; T12: Account/High -> None/High; T13: Account/Low -> Technical/High; T14: Account/Medium -> Account/High; T15: Account/Low -> Account/None
- **v2 Role/Task/Constraints/Output** mistakes: T2: Billing/Low -> Billing/Medium; T3: Billing/Medium -> Billing/Low; T5: Billing/Low -> Billing/Medium; T6: Technical/Medium -> Technical/High; T11: Account/High -> Technical/High; T12: Account/High -> Technical/High; T14: Account/Medium -> Account/High
- **v3 + 3 few-shot examples** mistakes: T2: Billing/Low -> Billing/High; T3: Billing/Medium -> Billing/High; T5: Billing/Low -> Billing/Medium; T6: Technical/Medium -> Technical/High; T8: Technical/Low -> Technical/High; T14: Account/Medium -> Account/High

## B. Chain-of-thought (5 word problems, temperature 0)

| Mode | Correct | Output tokens (total) | Answers (expected 32, 7500, 270, 78, 36) |
|---|---|---|---|
| Direct (final number only) | 1/5 | 17 | 76.0, 150.0, 310.0, 48.0, 36.0 |
| Think step by step | 5/5 | 1094 | 32.0, 7500.0, 270.0, 78.0, 36.0 |

## C. Sampling - same creative prompt, 5 runs each
Prompt: *Write a catchy one-sentence slogan for a small filter-coffee shop in Chennai.*

| Setting | Different answers (of 5) | Answers |
|---|---|---|
| temperature 0 | 1 | Every sip, crafted with care, from the heart of Chennai.<br>Every sip, crafted with care, from the heart of Chennai.<br>Every sip, crafted with care, from the heart of Chennai.<br>Every sip, crafted with care, from the heart of Chennai.<br>Every sip, crafted with care, from the heart of Chennai. |
| temperature 0.7 | 5 | Chennai's Brew Treasure: Freshly Filtered Coffee at Your Doorstep!<br>Fresh brews and cozy vibes at your doorstep—where every cup is art!<br>Chill out with us and let the beans talk while you sip on our crisp, freshly brewed coffee.<br>Every sip, crafted with love from our cozy, aromatic coffee beans.<br>Chill out with our fresh brew at the quaint coffee spot in Chennai! |
| temperature 1.2 | 5 | Brewing Excellence in Every Precious Cup - Tap Into the Smell of Quality at JCharlotte Coffee!<br>Espresso Beyond Borders: Local Brewsourcing from Our Chocolate-Covered Grind<br>Chill out, sit bean-to-cup artisanally, forever fresh coffeeﺍﬃックス.<br>Every Grind Delivers: Refreshing Coffee Small Batch / Chennai Espresso Corner<br>Coffee Bean Bliss: Refine Your Brew onshore! |
| temperature 1.0, top_p 0.3 | 5 | Chill out with us and brew up the perfect cup of coffee in Chennai!<br>Fresh Brews for Every Journey - Where Coffee Flavors Connect.<br>Chill out with us and savor the perfect cup of coffee from our tiny but cozy filter-shop!<br>Every sip, crafted with care, from the heart of Chennai.<br>Chill out with us and savor the perfect cup of coffee from our tiny but cozy filter-shop in Chennai. |
| temperature 1.0, top_p 0.95 | 5 | Chennai's Art of Coffee – Freshly Filtered Every Morning.<br>Espresso Elegance / Freshness Everlasting - Where the spirit of coffee meets South Indian charm.<br>Chill out, brew up; enjoy the finest coffee from our cozy Chennaitte spot.<br>Every sip sips at our cozy Chennai coffee spot!<br>Coffee Bean Bliss: A Smaller Story of Brewed Excellence |

## D. Structured output - {name, skills[], years_experience} from 5 resumes

Every output is parsed with Pydantic `model_validate_json` (name str, skills non-empty list, years 0-50).

| Resume | Plain prompting | JSON mode | JSON mode name / years correct? |
|---|---|---|---|
| 1 | invalid (json: Invalid JSON: expected value at line 1 column 1) | valid | name yes, years yes (4.0) |
| 2 | invalid (json: Invalid JSON: expected value at line 1 column 1) | valid | name yes, years yes (7.0) |
| 3 | invalid (json: Invalid JSON: expected value at line 1 column 1) | valid | name yes, years yes (0.5) |
| 4 | invalid (json: Invalid JSON: expected value at line 1 column 1) | valid | name yes, years yes (9.0) |
| 5 | invalid (json: Invalid JSON: expected value at line 1 column 1) | valid | name yes, years yes (5.0) |

**Plain: 5/5 invalid JSON | JSON mode: 0/5 invalid**

## E. Three bad prompts and their rewrites

| Mistake | Bad prompt -> answer | Rewritten prompt -> answer | Bad passes? | Rewrite passes? |
|---|---|---|---|---|
| Vague instruction | Tell me about this ticket: My order #45120 was delivered to the wrong ... -> I'm sorry to hear that you received your order at the wrong address. This is clearly an is... | Classify this support ticket into exactly one category from [Billing, ... -> Shipping | no | yes |
| No output format | Find the order number and city in this ticket: My order #45120 was del... -> The order number is 45120 and the city mentioned is Coimbatore. | Extract the order number and city. Return ONLY this JSON, nothing else... -> ```json {"order_id": "45120", "city": "Coimbatore"} ``` | no | yes |
| Conflicting instructions | Summarise this review in one word. Give a detailed explanation with at... -> The summary of this review is "Disappointed". Three reasons for this summary: 1. **Battery... | What is the sentiment of this review? Reply with one word: positive, n... -> Negative | no | yes |

**Bad prompts passed 0/3 - rewritten prompts passed 3/3**
