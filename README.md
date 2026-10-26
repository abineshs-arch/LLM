---

# Week 4 - Day 2 (Part 2): Encoder vs decoder, positional encoding, Transformers vs RNNs

## Task 1 - Encoder vs decoder and positional encoding (`task1_encoder_decoder.py`)

| | Encoder-only (BERT-style) | Decoder-only (GPT-style) |
|---|---|---|
| How it reads | Whole sentence at once, in **both directions** (every token sees every token) | **Left to right** only; a causal mask hides future tokens |
| What it outputs | A context vector for every token -> a label, a filled-in word, an embedding | The **next token**, again and again -> new text |
| Best tasks | Classification, NER, sentiment, search / embeddings, fill-mask | Chat, drafting emails and replies, summaries, code generation |
| Example models | BERT, DistilBERT, RoBERTa | GPT-2, GPT-6, Claude, Llama, Gemini |

**Positional encoding:** sinusoidal, 10 positions x 8 dimensions
(`PE[pos, 2i] = sin(pos / 10000^(2i/8))`, `PE[pos, 2i+1] = cos(...)`) -> heatmap `positional_encoding.png`.
Row 0 = [0, 1, 0, 1, ...]; every position gets a different pattern; the left dimensions change fast, the right ones slowly.

**Why word order is lost without positional encoding**
1. Self-attention compares every token with every other token using only their vectors.
2. It has no built-in notion of "first", "second" or "next to".
3. So "dog bites man" and "man bites dog" contain the same vectors and would get the same attention scores.
4. Adding a unique position pattern to each token vector lets the model know where each word is.

**Example - encoder (BERT fill-mask):** "The customer was angry because the [MASK] arrived broken."
-> glass 0.174, window 0.039, ice 0.024, clock 0.015, bottle 0.014.
BERT used the words on **both sides** ("arrived broken") and guessed things that break -> good at understanding.

**Example - decoder (GPT-2 generation):** "Dear customer, we are sorry that your order arrived broken. We will"
-> "...be contacting you to resolve the issue. Thank you for your patience."
GPT-2 wrote a fluent continuation one token at a time -> good at writing.

## Task 2 - Transformers vs RNNs (`task2_lstm_vs_transformer.py`)
Toy task: is a sequence of 8 digits sorted? 8,000 train / 2,000 test sequences (49% sorted; hard negatives
are "almost sorted" with one neighbour swap). Same data, Adam lr 1e-3, batch 64, 15 epochs, CPU.

| Model | Parameters | Test accuracy | Training time |
|---|---|---|---|
| LSTM (1 layer, hidden 64) | 25,473 | **0.999** | **6.5 s** |
| TransformerEncoder (2 layers, 4 heads, d_model 32) | 17,697 | 0.994 | 22.9 s |

**Result:** on this tiny task the LSTM was slightly more accurate and 3.5x faster. The sequences are short (8 digits)
and sortedness only needs neighbour comparisons, which an RNN reads naturally; on a CPU with tiny models the
attention overhead costs more than the parallelism saves. Transformers win on long texts, large models and GPUs.

**4 reasons Transformers replaced RNNs**
1. **Parallelism:** an RNN must read token 1 before token 2; attention processes all tokens at once, so GPUs train it much faster on big data.
2. **Long-range attention:** any token can look directly at any other token, even thousands of tokens away; an RNN must pass information step by step and forgets it.
3. **Scaling:** Transformers keep improving as you add data, layers and GPUs, which made billion-parameter LLMs possible.
4. **Transfer learning:** one pre-trained Transformer (BERT, GPT) can be fine-tuned or prompted for many tasks instead of training a new RNN each time.

**Vanishing gradient problem**
1. An RNN is trained by sending the error backwards through every time step (backpropagation through time).
2. At each step the gradient is multiplied by small numbers (weights and activation slopes below 1).
3. After many steps the gradient shrinks towards zero, so the early words get almost no learning signal.
4. The network therefore "forgets" the beginning of long sentences; LSTM gates reduce this, and attention avoids it with direct connections.

**How ChatGPT-style models generate text - for a non-technical manager (150 words)**
ChatGPT-style models write one small piece of text at a time. First, your question is split into tokens - short
word pieces. The model has learned, from reading a huge amount of text, which token is most likely to come next
after any sequence of tokens. It looks at your whole question, scores every possible next token, picks a likely
one, and adds it to the answer. Then it repeats the process: it reads your question plus everything it has written
so far and predicts the next token again, hundreds of times, until the answer is complete. This is why answers
appear word by word on the screen. The model does not look facts up in a database; it predicts what text should
come next based on patterns. That makes it fluent and fast, but it can sound confident while being wrong, so
important answers should still be checked by a person.

## Concepts
- **Encoder vs decoder:** encoders understand (read both ways), decoders generate (read left to right).
- **Positional encoding:** adds position information so attention knows word order.
- **RNN / LSTM vs Transformer:** step-by-step reading vs all tokens at once with attention.
- **Parallelism:** Transformers train on all tokens at the same time on GPUs.
- **Explaining AI to non-experts:** describe next-token prediction simply and be honest about limits.