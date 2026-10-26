"""Week 4 Day 2 (part 2) - Task 2: tiny LSTM vs tiny Transformer on 'is this digit sequence sorted?'."""
import time

import numpy as np
import torch
from torch import nn

SEQ_LEN, EPOCHS, BATCH = 8, 15, 64
torch.manual_seed(42)
np.random.seed(42)


def make_data(n):
    """50% sorted sequences, 25% random, 25% 'almost sorted' (one neighbour swap) -> hard negatives."""
    X, y = [], []
    for _ in range(n):
        r = np.random.rand()
        if r < 0.5:
            s = np.sort(np.random.randint(0, 10, SEQ_LEN))
        elif r < 0.75:
            s = np.random.randint(0, 10, SEQ_LEN)
        else:
            while True:
                s = np.sort(np.random.randint(0, 10, SEQ_LEN))
                diff = np.where(s[:-1] != s[1:])[0]
                if len(diff):
                    break
            i = np.random.choice(diff)
            s[i], s[i + 1] = s[i + 1], s[i]          # swap two neighbours -> not sorted any more
        X.append(s)
        y.append(int(np.all(np.diff(s) >= 0)))       # 1 = sorted, 0 = not sorted
    return torch.tensor(np.array(X), dtype=torch.long), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)          # reads the digits ONE BY ONE
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.out(h[-1]).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 32)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, 32) * 0.1)   # learned positional encoding
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        h = self.encoder(self.emb(x) + self.pos)                # all digits AT ONCE
        return self.out(h.mean(dim=1)).squeeze(-1)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)) > 0.5).float()
    return (pred == y).float().mean().item()


def train(model, name, X_tr, y_tr, X_te, y_te):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}  ({n_params:,} parameters)")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        for b in range(0, len(X_tr), BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        if epoch in (1, 5, 10, 15):
            print(f"  epoch {epoch:2}: loss {loss.item():.3f} | test accuracy {accuracy(model, X_te, y_te):.3f}")
    seconds = time.time() - start
    acc = accuracy(model, X_te, y_te)
    print(f"  -> final test accuracy {acc:.3f} | training time {seconds:.1f} s")
    return n_params, acc, seconds


def main():
    X_tr, y_tr = make_data(8000)
    X_te, y_te = make_data(2000)
    print(f"Task: is a sequence of {SEQ_LEN} digits sorted? Train {len(X_tr)} | test {len(X_te)} "
          f"| {y_tr.mean():.0%} sorted")
    print("Examples:", X_tr[0].tolist(), "->", int(y_tr[0]), "|", X_tr[1].tolist(), "->", int(y_tr[1]))

    results = {
        "LSTM (1 layer)": train(LSTMModel(), "LSTM (1 layer, hidden 64)", X_tr, y_tr, X_te, y_te),
        "Transformer (2 layers)": train(TransformerModel(), "TransformerEncoder (2 layers, 4 heads)",
                                        X_tr, y_tr, X_te, y_te),
    }
    print("\n" + "=" * 60 + "\nCOMPARISON (same data, 15 epochs, CPU)\n" + "=" * 60)
    print(f"{'Model':24} {'Params':>8} {'Test acc':>9} {'Time (s)':>9}")
    for name, (p, acc, s) in results.items():
        print(f"{name:24} {p:8,} {acc:9.3f} {s:9.1f}")


if __name__ == "__main__":
    main()