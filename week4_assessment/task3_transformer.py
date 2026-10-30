"""Task 3: self-attention, 2-head attention, positional encoding, BERT fill-mask and GPT-2 generation."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
np.set_printoptions(precision=3, suppress=True)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])          # scaled dot-product
    weights = softmax(scores)                        # each row sums to 1
    return weights @ V, weights


def part_a_single_head():
    print("=" * 60 + "\nA. SINGLE-HEAD SELF-ATTENTION (4 tokens, size 3)\n" + "=" * 60)
    np.random.seed(0)
    X = np.random.randn(4, 3)
    Wq, Wk, Wv = np.random.randn(3, 3), np.random.randn(3, 3), np.random.randn(3, 3)
    output, weights = attention(X @ Wq, X @ Wk, X @ Wv)
    print("Attention weights (4x4):\n", weights)
    print("Row sums:", weights.sum(axis=1), "-> each row sums to 1.0:", np.allclose(weights.sum(axis=1), 1))
    print("Output (4x3):\n", output)


def part_b_two_heads():
    print("\n" + "=" * 60 + "\nB. TWO-HEAD ATTENTION (d_model 4 = 2 heads x 2)\n" + "=" * 60)
    np.random.seed(0)
    X = np.random.randn(4, 4)
    Wq, Wk, Wv, Wo = (np.random.randn(4, 4) for _ in range(4))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    heads = []
    for h in range(2):
        cols = slice(2 * h, 2 * h + 2)
        out, w = attention(Q[:, cols], K[:, cols], V[:, cols])
        heads.append(out)
        print(f"Head {h + 1} weights (rows sum to 1: {np.allclose(w.sum(axis=1), 1)}):\n{w}")
    concat = np.concatenate(heads, axis=1)            # (4, 4)
    print("Output after concat + output projection Wo (4x4):\n", concat @ Wo)


def part_c_positional_encoding():
    print("\n" + "=" * 60 + "\nC. SINUSOIDAL POSITIONAL ENCODING (10 x 8)\n" + "=" * 60)
    pos = np.arange(10)[:, None]
    i = np.arange(0, 8, 2)[None, :]
    angles = pos / np.power(10000, i / 8)
    pe = np.zeros((10, 8))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
    print(pe)
    plt.figure(figsize=(8, 5))
    plt.imshow(pe, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="value")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal positional encoding (10 positions x 8 dimensions)")
    plt.savefig(BASE / "positional_encoding.png", dpi=150, bbox_inches="tight")
    print("Saved positional_encoding.png")


def part_d_bert_gpt2():
    from transformers import pipeline, set_seed
    print("\n" + "=" * 60 + "\nD. ENCODER (BERT fill-mask) vs DECODER (GPT-2 generation)\n" + "=" * 60)
    fill = pipeline("fill-mask", model="bert-base-uncased")
    sentence = "The customer asked for a [MASK] because the product arrived broken."
    print(sentence)
    for p in fill(sentence, top_k=5):
        print(f"  {p['token_str']:10} {p['score']:.3f}")
    gen = pipeline("text-generation", model="gpt2")
    set_seed(42)
    prompt = "Thank you for contacting support. Your refund"
    out = gen(prompt, max_new_tokens=30, do_sample=True, temperature=0.7, pad_token_id=gen.tokenizer.eos_token_id)
    print("\nGPT-2:", out[0]["generated_text"])


if __name__ == "__main__":
    part_a_single_head()
    part_b_two_heads()
    part_c_positional_encoding()
    part_d_bert_gpt2()