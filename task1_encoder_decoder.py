"""Week 4 Day 2 (part 2) - Task 1: positional encoding heatmap + BERT fill-mask + GPT-2 generation."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
np.set_printoptions(precision=3, suppress=True)


def positional_encoding(n_positions, d_model):
    """Sinusoidal encoding from 'Attention Is All You Need':
    PE[pos, 2i] = sin(pos / 10000^(2i/d)),  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))"""
    pos = np.arange(n_positions)[:, None]                 # (10, 1)
    i = np.arange(0, d_model, 2)[None, :]                 # (1, 4) -> 0, 2, 4, 6
    angles = pos / np.power(10000, i / d_model)           # (10, 4)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angles)                          # even dimensions
    pe[:, 1::2] = np.cos(angles)                          # odd dimensions
    return pe


def part_a_positional_encoding():
    print("=" * 60 + "\nPART A: SINUSOIDAL POSITIONAL ENCODING (10 positions x 8 dims)\n" + "=" * 60)
    pe = positional_encoding(10, 8)
    print(pe)
    plt.figure(figsize=(8, 5))
    plt.imshow(pe, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="value")
    plt.xlabel("Dimension")
    plt.ylabel("Position in the sentence")
    plt.xticks(range(8))
    plt.yticks(range(10))
    plt.title("Sinusoidal positional encoding (10 positions x 8 dimensions)")
    plt.savefig(BASE / "positional_encoding.png", dpi=150, bbox_inches="tight")
    print("Saved positional_encoding.png")
    print("Every row (position) is a different pattern, so the model can tell positions apart.")


def part_b_bert_fill_mask():
    print("\n" + "=" * 60 + "\nPART B: ENCODER (BERT) - fill-mask = understanding\n" + "=" * 60)
    from transformers import pipeline
    fill = pipeline("fill-mask", model="bert-base-uncased")
    sentence = "The customer was angry because the [MASK] arrived broken."
    print(sentence)
    for p in fill(sentence, top_k=5):
        print(f"  {p['token_str']:10} {p['score']:.3f}")
    print("BERT reads the words on BOTH sides of [MASK] ('customer', 'arrived broken') to guess it.")


def part_c_gpt2_generation():
    print("\n" + "=" * 60 + "\nPART C: DECODER (GPT-2) - text generation = writing\n" + "=" * 60)
    from transformers import pipeline, set_seed
    gen = pipeline("text-generation", model="gpt2")
    set_seed(42)
    prompt = "Dear customer, we are sorry that your order arrived broken. We will"
    out = gen(prompt, max_new_tokens=30, do_sample=True, temperature=0.7,
              pad_token_id=gen.tokenizer.eos_token_id)
    print(out[0]["generated_text"])
    print("GPT-2 only reads LEFT to right and keeps predicting the next word.")


if __name__ == "__main__":
    part_a_positional_encoding()
    part_b_bert_fill_mask()
    part_c_gpt2_generation()