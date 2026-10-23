# Foundation Models Comparison (Week 4 - Day 1)

Information checked on 23 October 2026 from each provider's official website (sources at the end).

## Comparison table

| Model family | Owner | Open or closed weights | Context window | Strengths | Typical use |
|---|---|---|---|---|---|
| **GPT** (GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna) | OpenAI | Closed (API / ChatGPT only) | ~1.05M tokens | Strong reasoning and coding, large tool ecosystem, a cheap small model (Luna) | Chatbots, coding assistants, agents, general business apps |
| **Claude** (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5) | Anthropic | Closed (API / Claude apps) | 1M tokens | Long documents, careful instruction following, coding and agentic work, safety focus | Document analysis, coding, customer support, agents |
| **Llama** (Llama 4 Scout, Llama 4 Maverick) | Meta | **Open weights** (download from llama.com / Hugging Face, Llama licence) | Up to 10M tokens (Scout) | Can run on your own servers, can be fine-tuned, no per-token API fee | Private / on-premise apps, custom fine-tuned models, research |
| **Gemini** (Gemini 4 Argon, Gemini 3.8 Flash, Flash-Lite) | Google | Closed (API / Gemini app) | 1M tokens | Natively multimodal (text, image, audio, video), long context, cheap Flash models | Multimodal apps, long-document and video understanding, Google Cloud apps |

## What "pre-trained on massive text" and "foundation model" mean
1. **Pre-training** means the model first learns from a huge amount of text (web pages, books, code) by repeatedly predicting the next token.
2. Doing this billions of times teaches it grammar, facts, reasoning patterns and how code works - without anyone labelling the data.
3. A **foundation model** is the general-purpose model that results from this pre-training (plus some instruction tuning).
4. It is a "foundation" because many different applications are built on top of the same model instead of training a new model for each task.
5. We adapt it with prompts, retrieval (RAG), tools or light fine-tuning, rather than training from scratch.

## 5 tasks a foundation model can do without being retrained
1. Summarise a long document or meeting transcript
2. Classify customer tickets or feedback (zero-shot, just describe the labels in the prompt)
3. Translate text between languages (for example English to Tamil)
4. Extract structured data (names, amounts, dates) from text into JSON
5. Write and explain code, or draft email replies

## Model choices

**Customer-support bot -> a small, fast hosted model (GPT-6 Luna; Claude Haiku 5.5 or Gemini 3.8 Flash are equal alternatives)**
- Support bots handle thousands of short chats a day, so cost and speed matter more than maximum intelligence.
- GPT-6 Luna costs $0.10 per 1M input and $0.50 per 1M output tokens and has a ~1M token context, enough for product FAQs.
- A hosted API means no GPU servers to manage; harder cases can be escalated to a bigger model or a human.

**Private on-premise document tool -> Llama 4 (open weights)**
- The weights can be downloaded and run on the company's own servers, so confidential documents never leave the network.
- No per-token API cost; the main cost is the hardware.
- It can be fine-tuned on internal documents, and Scout's very long context (up to 10M tokens) suits large document sets.

## Sources
- OpenAI - Compare models: https://developers.openai.com/api/docs/models/compare
- Anthropic - Claude models overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Google - Gemini API models: https://ai.google.dev/gemini-api/docs/models
- Google - Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Meta - The Llama 4 herd: https://ai.meta.com/blog/llama-4-multimodal-intelligence/