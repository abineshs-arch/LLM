"""Week 4 Day 2 - Task 2: multi-head attention, Hugging Face attention inspection, causal mask."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)       # blocked positions get -infinity -> weight 0
    weights = softmax(scores)
    return weights @ V, weights


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    d_model = X.shape[1]
    d_head = d_model // n_heads                         # 4 / 2 = 2
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    heads, all_weights = [], []
    for h in range(n_heads):
        cols = slice(h * d_head, (h + 1) * d_head)      # head 0 -> columns 0-1, head 1 -> columns 2-3
        out, w = attention(Q[:, cols], K[:, cols], V[:, cols], mask)
        heads.append(out)
        all_weights.append(w)
    concat = np.concatenate(heads, axis=1)              # (4, 2) + (4, 2) -> (4, 4)
    return concat @ Wo, all_weights                     # output projection


def part_a_numpy():
    print("=" * 60 + "\nPART A: MULTI-HEAD ATTENTION IN NUMPY (2 heads x 2 dims = 4)\n" + "=" * 60)
    np.random.seed(0)
    X = np.random.randn(4, 4)                           # 4 tokens, 4-dimensional model
    Wq, Wk, Wv, Wo = (np.random.randn(4, 4) for _ in range(4))
    output, weights = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=2)
    for h, w in enumerate(weights):
        print(f"\nHead {h + 1} attention weights (4x4), row sums {w.sum(axis=1)}:\n{w}")
    print("\nFinal output after concat + output projection (4 x 4):\n", output)
    print("The two heads give different weight patterns -> they look at different relationships.")

    print("\n" + "=" * 60 + "\nPART C: CAUSAL MASK (decoder attention)\n" + "=" * 60)
    causal = np.tril(np.ones((4, 4), dtype=bool))       # token i may only see tokens 0..i
    print("Mask (True = allowed):\n", causal.astype(int))
    _, masked = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=2, mask=causal)
    print("\nHead 1 weights WITH causal mask (upper triangle is 0 = no looking ahead):\n", masked[0])


def part_b_huggingface():
    print("\n" + "=" * 60 + "\nPART B: HUGGING FACE ATTENTION (bert-base-uncased)\n" + "=" * 60)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
    inputs = tok("The bank by the river was steep", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Tokens: {tokens}")
    print(f"{len(out.attentions)} layers, each with shape {tuple(out.attentions[0].shape)} "
          "(batch, heads, tokens, tokens)")

    layer, head = 8, 0
    bank = tokens.index("bank")
    row = out.attentions[layer][0, head, bank]          # how much 'bank' attends to every token
    print(f"\nAttention of 'bank' in layer {layer + 1}, head {head + 1} (row sums to {row.sum():.2f}):")
    for t, w in sorted(zip(tokens, row.tolist()), key=lambda p: -p[1]):
        print(f"  {t:8} {w:.3f}  {'#' * int(w * 50)}")


if __name__ == "__main__":
    part_a_numpy()
    part_b_huggingface()