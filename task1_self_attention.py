"""Week 4 Day 2 - Task 1: self-attention from scratch with NumPy."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).parent
np.set_printoptions(precision=3, suppress=True)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv               # (4,3) each
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)                  # (4,4) how much each token matches each other token
    weights = softmax(scores)                      # each row sums to 1
    output = weights @ V                           # (4,3) weighted mix of the values
    return Q, K, V, scores, weights, output


def draw_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    boxes = {
        "X\n(4 tokens x 3)": (0.02, 0.42),
        "Q = X·Wq": (0.22, 0.75), "K = X·Wk": (0.22, 0.42), "V = X·Wv": (0.22, 0.09),
        "Scores\nQ·Kᵀ / √d\n(4x4)": (0.43, 0.58),
        "Softmax\n(each row\nsums to 1)": (0.62, 0.58),
        "Output\nweights·V\n(4 x 3)": (0.82, 0.33),
    }
    for text, (x, y) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), 0.15, 0.17, boxstyle="round,pad=0.01",
                                    fc="#dbe9f6", ec="#2b6cb0", transform=ax.transAxes))
        ax.text(x + 0.075, y + 0.085, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    arrows = [((0.17, 0.50), (0.22, 0.83)), ((0.17, 0.50), (0.22, 0.50)), ((0.17, 0.50), (0.22, 0.17)),
              ((0.37, 0.83), (0.43, 0.68)), ((0.37, 0.50), (0.43, 0.62)),
              ((0.58, 0.66), (0.62, 0.66)), ((0.77, 0.62), (0.82, 0.46)), ((0.37, 0.17), (0.82, 0.38))]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=1.5))
    ax.set_title("Self-attention data flow (scaled dot-product attention)", fontsize=13)
    plt.savefig(path, dpi=150, bbox_inches="tight")


def main():
    np.random.seed(0)
    X = np.random.randn(4, 3)                       # 4 token vectors of size 3
    Wq, Wk, Wv = np.random.randn(3, 3), np.random.randn(3, 3), np.random.randn(3, 3)

    Q, K, V, scores, weights, output = self_attention(X, Wq, Wk, Wv)
    print("X (4 tokens x 3):\n", X)
    print("\nQ:\n", Q, "\nK:\n", K, "\nV:\n", V)
    print("\nScores = Q @ K.T / sqrt(3):\n", scores)
    print("\nATTENTION WEIGHTS (4x4):\n", weights)
    print("\nRow sums:", weights.sum(axis=1), "-> all equal 1:", np.allclose(weights.sum(axis=1), 1))
    print("\nOutput = weights @ V (4 x 3):\n", output)

    draw_diagram(BASE / "self_attention_diagram.png")
    print("\nSaved self_attention_diagram.png")


if __name__ == "__main__":
    main()