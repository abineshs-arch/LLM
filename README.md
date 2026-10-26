---

# Week 4 - Day 2: Transformer attention

## Task 1 - Self-attention from scratch (`task1_self_attention.py`)
- 4 token vectors of size 3 (`np.random.seed(0)`), random Wq, Wk, Wv (3x3)
- Q = X·Wq, K = X·Wk, V = X·Wv -> scores = Q·Kᵀ / √3 -> softmax per row -> output = weights·V

**Attention weight matrix (4x4):**

| | token 1 | token 2 | token 3 | token 4 |
|---|---|---|---|---|
| token 1 | 0.001 | 0.999 | 0.000 | 0.000 |
| token 2 | 0.257 | 0.000 | 0.000 | 0.743 |
| token 3 | 0.265 | 0.335 | 0.099 | 0.300 |
| token 4 | 0.000 | 1.000 | 0.000 | 0.000 |

Row sums: [1, 1, 1, 1] -> verified with `np.allclose` = True.

**Diagram:** `self_attention_diagram.png` (X -> Q/K/V -> scores -> softmax -> weights·V -> output)

**Query, Key and Value - library analogy**
1. **Query** is the question you bring to the library: "I need books about rivers".
2. **Key** is the label on each book's spine that says what the book is about.
3. Comparing your query with every key (dot product) gives a score: how well each book matches.
4. Softmax turns the scores into percentages that add up to 100%, so you know how much to read from each book.
5. **Value** is the actual content of each book; the output is a weighted mix of the contents of the best-matching books.

## Task 2 - Multi-head attention (`task2_multi_head.py`)
- 4-dimensional model split into **2 heads of 2 dimensions**; each head runs attention on its own columns of Q, K, V
- Head outputs are **concatenated** (4x2 + 4x2 -> 4x4) and multiplied by an **output projection** Wo
- Head 1 focused almost entirely on token 3; Head 2 spread its attention over all tokens -> different patterns

**Hugging Face** (`bert-base-uncased`, `output_attentions=True`, sentence "The bank by the river was steep"):
- 12 layers, each attention tensor has shape (1, 12, 9, 9) = (batch, heads, tokens, tokens)
- Layer 9, head 1, word "bank": [SEP] 0.811, [CLS] 0.099, the 0.027, the 0.020, by 0.020, ...
- This head "parks" most attention on [SEP] (a known BERT pattern when a head has nothing useful to look at);
  other heads are the ones that link "bank" to "river" - which is exactly why we need several heads.

**Why several heads are better than one**
1. Each head has its own Wq, Wk, Wv, so it can learn a different kind of relationship (grammar, meaning, position).
2. One head can link "bank" to "river" while another links "bank" to "steep" or to the next word.
3. With one head, softmax forces a single attention pattern; several heads look at several things at the same time.
4. The outputs are concatenated and projected, so the model combines all these views into one richer representation.

**What the mask does in decoder attention**
A decoder (like GPT) writes text one token at a time, so a token must not see the words that come after it.
The causal mask sets the scores of all future positions to -infinity before softmax, so their weights become 0
(upper triangle of the matrix is 0). Our masked Head 1 shows this: row 1 only sees token 1, row 2 sees tokens 1-2, and so on.

## Concepts
- **Self-attention:** every token looks at every other token and builds a new vector as a weighted mix of them.
- **Q/K/V:** Query = what I am looking for, Key = what I contain, Value = what I give if selected.
- **Scaled dot-product:** Q·Kᵀ divided by √d keeps scores small so softmax does not become too sharp.
- **Softmax:** turns scores into weights between 0 and 1 that sum to 1 per row.
- **Multi-head attention:** several attention heads in parallel, concatenated and projected.
- **Causal mask:** blocks future tokens in decoder (GPT-style) models.