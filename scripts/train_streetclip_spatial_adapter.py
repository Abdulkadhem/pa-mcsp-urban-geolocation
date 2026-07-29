"""Train a lightweight residual spatial adapter on frozen StreetCLIP features.

Training is train-only and uses soft geodesic positives inside each batch. The
adapter starts as identity and can be applied to the cached gallery later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--positive-km", type=float, default=100.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260725)
    return p.parse_args()


class ResidualAdapter(torch.nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.down = torch.nn.Linear(dimension, hidden)
        self.up = torch.nn.Linear(hidden, dimension)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.up(F.gelu(self.down(x))), dim=1)


def distance_km(coords: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(coords[:, 0]); lon = torch.deg2rad(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = torch.sin(dlat / 2).square() + torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlon / 2).square()
    return 6371.0088 * 2.0 * torch.arcsin(torch.sqrt(a.clamp(0, 1)))


def main() -> None:
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter training")
    device = torch.device("cuda")
    index = pd.read_parquet(a.index)
    train_rows = np.flatnonzero(index["split"].astype(str).eq("train").to_numpy())
    if a.sample_size > len(train_rows):
        a.sample_size = len(train_rows)
    rng = np.random.default_rng(a.seed)
    selected = np.sort(rng.choice(train_rows, size=a.sample_size, replace=False))
    mmap = np.load(a.features, mmap_mode="r")
    features = np.asarray(mmap[selected], dtype=np.float32)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-6)
    coords = index.iloc[selected][["latitude", "longitude"]].to_numpy(dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(coords)), batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    model = ResidualAdapter(features.shape[1], a.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    records = []
    for epoch in range(a.epochs):
        model.train(); losses = []
        for xb, cb in loader:
            xb = xb.to(device, non_blocking=True); cb = cb.to(device, non_blocking=True)
            z = model(xb)
            logits = (z @ z.T) / a.temperature
            dist = distance_km(cb)
            positive = (dist <= a.positive_km) & ~torch.eye(len(cb), dtype=torch.bool, device=device)
            valid = positive.any(dim=1)
            if not bool(valid.any()):
                continue
            logits = logits.masked_fill(torch.eye(len(cb), dtype=torch.bool, device=device), -1e4)
            numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
            denominator = torch.logsumexp(logits, dim=1)
            loss = (denominator[valid] - numerator[valid]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        records.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "batches": len(losses)})
        print(json.dumps(records[-1]), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "method": "streetclip_spatial_residual_adapter", "dimension": int(features.shape[1]), "hidden": a.hidden, "sample_size": int(a.sample_size), "positive_km": a.positive_km, "seed": a.seed, "model_state": model.state_dict(), "records": records}, a.output)
    print(json.dumps({"output": str(a.output), "records": records}, indent=2))


if __name__ == "__main__":
    main()
