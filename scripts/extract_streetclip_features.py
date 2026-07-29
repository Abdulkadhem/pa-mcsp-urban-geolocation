from __future__ import annotations

import argparse
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


class ImageDataset(Dataset[tuple[int, Image.Image]]):
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[int, Image.Image]:
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        return index, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen StreetCLIP features for the urban pilot."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", default="geolocal/StreetCLIP")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=min(24, os.cpu_count() or 1))
    return parser.parse_args()


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
    for candidate in (160, 128, 96, 64, 48, 32, 16):
        images = []
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StreetCLIP feature extraction.")

    manifest = pd.read_parquet(args.manifest)
    manifest = manifest.loc[manifest["decode_ok"].astype(bool)].copy()
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("The decoded pilot manifest is empty.")
    paths = manifest["image_path"].astype(str).tolist()

    model_info = HfApi().model_info(
        repo_id=args.model_id,
        revision=args.model_revision,
        files_metadata=True,
    )
    locked_revision = model_info.sha
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

    dtype, precision_name = resolve_precision()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    batch_size = args.batch_size or probe_batch_size(
        paths,
        processor=processor,
        model=model,
        dtype=dtype,
    )

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
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate,
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    batches: list[np.ndarray] = []
    observed_indices: list[np.ndarray] = []
    with torch.inference_mode():
        for indices, pixel_values in loader:
            pixel_values = pixel_values.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                features = model.get_image_features(pixel_values=pixel_values)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            batches.append(features.cpu().numpy().astype(np.float16))
            observed_indices.append(indices.numpy())
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    features_array = np.concatenate(batches, axis=0)
    order = np.concatenate(observed_indices, axis=0)
    if not np.array_equal(order, np.arange(len(manifest))):
        raise RuntimeError("Feature extraction changed manifest row order.")
    if not np.isfinite(features_array).all():
        raise RuntimeError("Feature array contains non-finite values.")

    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_features = args.features_output.with_suffix(
        args.features_output.suffix + ".partial"
    )
    with temporary_features.open("wb") as handle:
        np.save(handle, features_array, allow_pickle=False)
    temporary_features.replace(args.features_output)

    index_columns = [
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
            "image_path",
        )
        if column in manifest.columns
    ]
    feature_index = manifest[index_columns].copy()
    feature_index.insert(0, "feature_row", np.arange(len(feature_index)))
    temporary_index = args.index_output.with_suffix(
        args.index_output.suffix + ".partial"
    )
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    feature_index.to_parquet(temporary_index, index=False)
    temporary_index.replace(args.index_output)

    stats = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_revision": locked_revision,
        "precision": precision_name,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "n_images": len(manifest),
        "feature_dimension": features_array.shape[1],
        "feature_dtype": str(features_array.dtype),
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": len(manifest) / elapsed_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu_name": torch.cuda.get_device_name(0),
        "features_output": str(args.features_output),
        "index_output": str(args.index_output),
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

