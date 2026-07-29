"""Apply a trained residual adapter to existing feature arrays in streaming batches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_streetclip_spatial_adapter import ResidualAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=65536)
    return p.parse_args()


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    src = np.load(a.input, mmap_mode="r")
    ckpt = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    model = ResidualAdapter(int(src.shape[1]), int(ckpt["hidden"])).cuda().eval()
    model.load_state_dict(ckpt["model_state"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(a.output, mode="w+", dtype=np.float16, shape=src.shape)
    with torch.inference_mode():
        for start in range(0, len(src), a.batch_size):
            end = min(len(src), start + a.batch_size)
            x = torch.from_numpy(np.asarray(src[start:end], dtype=np.float32)).cuda(non_blocking=True)
            dst[start:end] = model(x).cpu().numpy().astype(np.float16)
            if start % (a.batch_size * 8) == 0: print(json.dumps({"rows": end, "total": len(src)}), flush=True)
    dst.flush()
    meta = {"schema_version": 1, "input": str(a.input), "output": str(a.output), "shape": list(src.shape), "dtype": "float16", "input_sha256": sha256_file(a.input), "output_sha256": sha256_file(a.output), "checkpoint": str(a.checkpoint)}
    a.output.with_suffix(a.output.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
