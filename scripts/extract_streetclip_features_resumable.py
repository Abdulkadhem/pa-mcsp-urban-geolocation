#!/usr/bin/env python3
"""Resumable, manifest-ordered StreetCLIP feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPModel


class IndexedImageDataset(Dataset[tuple[int, Image.Image]]):
    def __init__(self, paths: list[str], offset: int) -> None:
        self.paths = paths
        self.offset = offset

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[int, Image.Image]:
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        return self.offset + index, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", default="geolocal/StreetCLIP")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(14, os.cpu_count() or 1),
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_precision() -> tuple[torch.dtype, str]:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def probe_batch_size(
    paths: list[str],
    *,
    processor: Any,
    model: CLIPModel,
    dtype: torch.dtype,
) -> int:
    for candidate in (256, 224, 192, 160, 128, 96, 64, 48, 32, 16):
        images: list[Image.Image] = []
        for path in paths[: min(candidate, len(paths))]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        try:
            pixel_values = processor(
                images=images,
                return_tensors="pt",
            )["pixel_values"].to("cuda", non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=dtype,
            ):
                _ = model.get_image_features(pixel_values=pixel_values)
            torch.cuda.synchronize()
            del pixel_values, images
            torch.cuda.empty_cache()
            return candidate
        except torch.OutOfMemoryError:
            del images
            torch.cuda.empty_cache()
    raise RuntimeError("StreetCLIP does not fit even with batch size 16.")


def feature_index_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "id",
            "split",
            "official_split",
            "sequence",
            "latitude",
            "longitude",
            "country",
            "urban_class",
            "spatial_block_id",
            "image_path",
        )
        if column in manifest.columns
    ]
    output = manifest[columns].copy()
    output.insert(0, "feature_row", np.arange(len(output)))
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StreetCLIP extraction.")
    manifest_hash = sha256_file(args.manifest)
    manifest = pd.read_parquet(args.manifest)
    manifest = manifest.loc[manifest["decode_ok"].astype(bool)].reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("The decoded manifest is empty.")
    paths = manifest["image_path"].astype(str).tolist()

    model_info = HfApi().model_info(
        repo_id=args.model_id,
        revision=args.model_revision,
        files_metadata=True,
    )
    locked_revision = str(model_info.sha)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=locked_revision,
    )
    model = CLIPModel.from_pretrained(
        args.model_id,
        revision=locked_revision,
        torch_dtype="auto",
    ).eval().to("cuda")
    model.requires_grad_(False)
    feature_dimension = int(model.config.projection_dim)

    dtype, precision_name = resolve_precision()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    batch_size = args.batch_size or probe_batch_size(
        paths,
        processor=processor,
        model=model,
        dtype=dtype,
    )

    processed_rows = 0
    cumulative_active_seconds = 0.0
    if args.state_output.is_file():
        state = json.loads(args.state_output.read_text(encoding="utf-8"))
        compatible = (
            state.get("manifest_sha256") == manifest_hash
            and state.get("model_revision") == locked_revision
            and int(state.get("total_rows", -1)) == len(manifest)
            and int(state.get("feature_dimension", -1)) == feature_dimension
            and args.features_output.is_file()
        )
        if compatible:
            processed_rows = int(state.get("processed_rows", 0))
            cumulative_active_seconds = float(
                state.get("active_seconds", 0.0)
            )
        else:
            raise RuntimeError(
                "Existing feature state is incompatible; use a new output path."
            )

    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    if processed_rows == 0:
        matrix = np.lib.format.open_memmap(
            args.features_output,
            mode="w+",
            dtype=np.float16,
            shape=(len(manifest), feature_dimension),
        )
    else:
        matrix = np.load(args.features_output, mmap_mode="r+")
        if matrix.shape != (len(manifest), feature_dimension):
            raise RuntimeError("Feature matrix shape differs from resume state.")

    def collate(
        samples: list[tuple[int, Image.Image]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, images = zip(*samples, strict=True)
        pixel_values = processor(
            images=list(images),
            return_tensors="pt",
        )["pixel_values"]
        return torch.tensor(indices, dtype=torch.int64), pixel_values

    loader = DataLoader(
        IndexedImageDataset(paths[processed_rows:], processed_rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3 if args.num_workers > 0 else None,
        collate_fn=collate,
    )
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    last_processed = processed_rows
    with torch.inference_mode():
        for batch_number, (indices, pixel_values) in enumerate(loader, start=1):
            expected_start = last_processed
            expected = torch.arange(
                expected_start,
                expected_start + len(indices),
                dtype=torch.int64,
            )
            if not torch.equal(indices, expected):
                raise RuntimeError("DataLoader changed manifest row order.")
            pixel_values = pixel_values.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                features = model.get_image_features(pixel_values=pixel_values)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            batch_features = features.cpu().numpy().astype(np.float16)
            matrix[expected_start : expected_start + len(indices)] = batch_features
            last_processed += len(indices)
            if (
                batch_number % max(1, args.checkpoint_every_batches) == 0
                or last_processed == len(manifest)
            ):
                matrix.flush()
                active_seconds = (
                    cumulative_active_seconds
                    + time.perf_counter()
                    - run_started
                )
                atomic_json(
                    args.state_output,
                    {
                        "schema_version": 1,
                        "status": (
                            "complete"
                            if last_processed == len(manifest)
                            else "running"
                        ),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "manifest_sha256": manifest_hash,
                        "model_id": args.model_id,
                        "model_revision": locked_revision,
                        "total_rows": len(manifest),
                        "processed_rows": last_processed,
                        "feature_dimension": feature_dimension,
                        "feature_dtype": "float16",
                        "batch_size": batch_size,
                        "num_workers": args.num_workers,
                        "active_seconds": active_seconds,
                        "features_output": str(args.features_output),
                    },
                )
    torch.cuda.synchronize()
    elapsed = cumulative_active_seconds + time.perf_counter() - run_started
    matrix.flush()
    if last_processed != len(manifest):
        raise RuntimeError("Feature extraction stopped before all rows were written.")
    if not np.isfinite(np.asarray(matrix)).all():
        raise RuntimeError("Feature matrix contains non-finite values.")

    index = feature_index_frame(manifest)
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = args.index_output.with_suffix(".parquet.partial")
    index.to_parquet(temporary_index, index=False)
    temporary_index.replace(args.index_output)
    stats = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "model_id": args.model_id,
        "model_revision": locked_revision,
        "precision": precision_name,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "n_images": len(manifest),
        "feature_dimension": feature_dimension,
        "feature_dtype": "float16",
        "active_seconds": elapsed,
        "images_per_second": len(manifest) / elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_name": torch.cuda.get_device_name(0),
        "features_output": str(args.features_output),
        "features_sha256": sha256_file(args.features_output),
        "index_output": str(args.index_output),
        "index_sha256": sha256_file(args.index_output),
    }
    atomic_json(args.stats_output, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
