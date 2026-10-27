# Week 4 Day 3 (part 2) results - model Qwen/Qwen2.5-0.5B-Instruct (local, CPU)

## Task 1a - Chain-of-thought (5 word problems, greedy)

| Mode | Correct | Output tokens (total) | Time (s) |
|---|---|---|---|
| Direct answer | 1/5 | 308 | 56.3 |
| Think step by step | 3/5 | 1356 | 237.7 |

| Mode | Problem | Expected | Model answer | Correct | Tokens |
|---|---|---|---|---|---|
| Direct answer | 1 | 32 | 56.0 | NO | 3 |
| Direct answer | 2 | 7500 | 18000.0 | NO | 6 |
| Direct answer | 3 | 270 | 760.0 | NO | 4 |
| Direct answer | 4 | 78 | 180.0 | NO | 4 |
| Direct answer | 5 | 36 | 36.0 | yes | 291 |
| Think step by step | 1 | 32 | 32.0 | yes | 232 |
| Think step by step | 2 | 7500 | 3.0 | NO | 300 |
| Think step by step | 3 | 270 | 270.0 | yes | 265 |
| Think step by step | 4 | 78 | 60.0 | NO | 300 |
| Think step by step | 5 | 36 | 36.0 | yes | 259 |

## Task 1b - Temperature (same creative prompt, 5 runs each, top_p = 1.0)
Prompt: *Write a catchy one-sentence slogan for a small filter-coffee shop in Chennai.*

| Temperature | Different answers (of 5) | Answers |
|---|---|---|
| 0.0 | 1 | Chennai's Filter-Cafe: Where Every Cup is Pure and Flavorful!<br>Chennai's Filter-Cafe: Where Every Cup is Pure and Flavorful!<br>Chennai's Filter-Cafe: Where Every Cup is Pure and Flavorful!<br>Chennai's Filter-Cafe: Where Every Cup is Pure and Flavorful!<br>Chennai's Filter-Cafe: Where Every Cup is Pure and Flavorful! |
| 0.7 | 5 | Chennai's Filter-Coffee Haven: Where Coffee Culture Meets Sustainability!<br>Fresh coffee, aromatic filters, and cozy vibes—truly the perfect spot for your morning brew!<br>不妨一杯平凡的香醇，邂逅一场味蕾上的盛宴——你身边的小小咖啡馆吧！<br>Chennai's No. 1 Filter-Coffee Shop:醇香飘逸，味蕾无限畅饮!<br>Chennai's Filter-Coffee Haven: Savor the Flavor with Us! |
| 1.2 | 5 | Brew deep leather planes over chocolate ships—a Tiny Coffee Loft.<br>breeze South India and sip with confidence - fi".<br>不妨尝试—一汤接一抹，品味中国经济便捷新风情!<br>fresher sips, warmer treats: cozy coffee magic greeted from your decadent bagel café.<br>Coffee+Customizer+Quality Approved at CCTV Filter Aficionados |

## Task 1c - Top-p (temperature 1.0, 5 runs each)

| top_p | Different answers (of 5) | Answers |
|---|---|---|
| 0.3 | 5 | Chennai's Filter-Cafe: Fresh Coffee, Unique Sips, and Exceptional Service.<br>Freshly Filtered Coffee, Freshly Brewed Life.<br>Chennai's finest coffee, crafted with care and love at our small filter-coffee shop.<br>Chennai's Filter-Cafe: Fresh Coffee, Local Flavor, and Exceptional Service.<br>Chennai's Filter-Cafe: Fresh Coffee, Delicate Moments. |
| 0.95 | 5 | Brewing daily moments like coffee – Filter for you at the heart of Chennai's unique niche.<br>Trift Beyond the Strain: Filter-Driven Comfort at the Heart of Indestructible Melodies.<br>不妨尝试，一汤接一抹，品味对味新体验——在 Chennai的小小咖啡馆，一瓶一杯，一口一口，品出不一样的味蕾之旅。<br>Find your way through the coffee capital with our daily mix and you'll never know how good we can be!<br>Coffee Filter Co., Chennai: Introducing Your Homemade Filters for an Extra Stave of Drunk Travel |

## Task 2a - Structured output: plain prompting vs JSON mode

Schema: `{"name": string, "skills": [string], "years_experience": number}`, every output checked with `json.loads` and a Pydantic model.

| Method | Invalid outputs (json.loads or Pydantic failed) |
|---|---|
| Plain prompting | 5/5 |
| JSON mode (strict system prompt + forced '{' start + cut at closing brace) | 0/5 |

| Resume | Method | Result | Reason | Output |
|---|---|---|---|---|
| 1 | Plain prompting | invalid JSON | json.loads: Expecting value | ````json { "name": "Ravi Kumar", "skills": [ "Python", "Django", "PostgreSQL", "Docker" ], "years_of_experience": 4 } ```` |
| 1 | JSON mode | valid |  | `{"name": "Ravi Kumar", "skills": ["Django", "PostgreSQL", "Docker", "AWS"], "years_experience": 4}` |
| 2 | Plain prompting | invalid JSON | json.loads: Expecting value | ````json { "name": "Meena Iyer", "skills": [ "SQL", "Excel", "Power BI", "Python" ], "years_of_experience": 7 } ```` |
| 2 | JSON mode | valid |  | `{"name": "data analyst", "skills": ["SQL", "Excel", "Power BI", "Python"], "years_experience": 7}` |
| 3 | Plain prompting | invalid JSON | json.loads: Expecting value | ````json { "name": "Arjun S.", "skills": ["React", "Node.js"], "years_of_experience": 0.5 } ```` |
| 3 | JSON mode | valid |  | `{"name": "Freshman", "skills": ["React", "Node.js"], "years_experience": 0.5}` |
| 4 | Plain prompting | invalid JSON | json.loads: Expecting value | ````json { "name": "Fatima Khan", "skills": [ "PyTorch", "NLP", "computer vision", "MLOps (Kubeflow)", "team leadership" ...` |
| 4 | JSON mode | valid |  | `{"name": "Dr. Fatima Khan", "skills": ["PyTorch", "NLP", "computer vision", "MLOps (Kubeflow)", "team leadership"], "yea...` |
| 5 | Plain prompting | invalid JSON | json.loads: Expecting value | ````json { "name": "Karthik", "skills": [ "Selenium", "JIRA", "DevOps" ], "years_of_experience": { "QA": 2, "DevOps": 3 }...` |
| 5 | JSON mode | valid |  | `{"name": "QA Engineer", "skills": ["Selenium", "JIRA", "DevOps"], "years_experience": 5}` |

## Task 2b - Fixing weak prompts

| Mistake | Version | Prompt | Model answer | Passes check? |
|---|---|---|---|---|
| Vague instruction | bad | Tell me about this ticket: My order #45120 was delivered to the wrong address in Coimbatore and I need it before Friday. | I'm sorry to hear that you're experiencing issues with your order. It sounds like you've received an incorrect delivery ... | no (answer is exactly 'Shipping') |
| Vague instruction | rewritten | Classify this support ticket into exactly one category from [Billing, Technical, Account, Shipping, General]. Reply with... | Technical | no (answer is exactly 'Shipping') |
| No output format | bad | Find the order number and city in this ticket: My order #45120 was delivered to the wrong address in Coimbatore and I ne... | The order number is 45120, which corresponds to your ticket for delivery to Coimbatore. To clarify, if you received the ... | no (valid JSON with order_id 45120 and city Coimbatore) |
| No output format | rewritten | Extract the order number and city from the ticket. Return only JSON like {"order_id": "...", "city": "..."}. Ticket: My ... | To extract the order ID and city from the given ticket, I'll follow these steps: 1. Identify the order ID. 2. Identify t... | no (valid JSON with order_id 45120 and city Coimbatore) |
| No examples | bad | Convert the date 'March 5th, 2026' to our standard date format. | To convert the date "March 5th, 2026" to our standard date format (YYYY-MM-DD), you can follow these steps: 1. Identify ... | no (answer is exactly 2026-03-05) |
| No examples | rewritten | Convert dates to our standard format. Examples: January 12th, 2025 -> 2025-01-12 July 3rd, 2024 -> 2024-07-03 Now conver... | 2026-03-05 | yes (answer is exactly 2026-03-05) |
| Conflicting instructions | bad | Summarise this review in one word. Give a detailed explanation with at least three reasons. Review: The phone looked nic... | The review summarizes that the product (likely a smartphone) had a good appearance but experienced issues with its batte... | no (answer is exactly 'negative') |
| Conflicting instructions | rewritten | What is the sentiment of this review? Reply with one word: positive, negative or neutral. Review: The phone looked nice ... | Neutral | no (answer is exactly 'negative') |

**Bad prompts passed: 0/4 - rewritten prompts passed: 1/4**
