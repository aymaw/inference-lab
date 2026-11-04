import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def architecture():
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))


def train(out="runs/model"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    x = torch.randn(1800, 8)
    y = torch.stack((x[:, 0] + x[:, 1], x[:, 2] - x[:, 3], -x[:, 0] + x[:, 4]), 1).argmax(1)
    model = architecture()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(100):
        loss = nn.functional.cross_entropy(model(x[:1200]), y[:1200])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        accuracy = (model(x[1200:]).argmax(1) == y[1200:]).float().mean().item()
    path = out / "weights.pt"
    torch.save(model.state_dict(), path)
    manifest = {
        "version": "synthetic-mlp-v1",
        "features": 8,
        "classes": 3,
        "seed": 7,
        "architecture": "8-32-3-relu",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "synthetic_only": True,
        "torch": torch.__version__,
        "test_accuracy": accuracy,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/model")
    train(p.parse_args().out)
