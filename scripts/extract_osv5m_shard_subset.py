#!/usr/bin/env python3
"""Download, verify, and extract a fixed OSV-5M train-shard subset.

This stage is intentionally independent from an experiment manifest.  It
downloads only predeclared archives, records every image member, and therefore
avoids scanning all 103 OSV-5M archives for a small pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--member-index", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=5)
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


def canonical_shard(value: str) -> str:
    token = value.strip()
    if token.isdigit():
        token = f"{int(token):02d}"
    if token.startswith("images/train/") and token.endswith(".zip"):
        return token
    if token.endswith(".zip"):
        token = token[:-4]
    return f"images/train/{token}.zip"


def selection_fingerprint(
    repo_id: str,
    revision: str,
    shard_paths: list[str],
) -> str:
    rendered = json.dumps(
        {
            "repo_id": repo_id,
            "revision": revision,
            "shards": sorted(shard_paths),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def process_shard(
    *,
    repo_id: str,
    revision: str,
    shard: dict[str, Any],
    dataset_root: Path,
    images_root: Path,
    state_root: Path,
    fingerprint: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shard_path = str(shard["path"])
    shard_token = Path(shard_path).stem
    state_path = state_root / f"train_{shard_token}.json"
    shard_index_path = state_root / "indices" / f"train_{shard_token}.parquet"

    if state_path.is_file() and shard_index_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("source_lfs_sha256") == shard.get("lfs_sha256")
            and existing.get("shard_path") == shard_path
        ):
            frame = pd.read_parquet(shard_index_path)
            if frame["image_path"].map(lambda value: Path(value).is_file()).all():
                existing["resumed_from_state"] = True
                existing["selection_fingerprint"] = fingerprint
                return frame, existing

    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=shard_path,
            cache_dir=dataset_root / "hf_cache",
        )
    )
    actual_size = downloaded_path.stat().st_size
    actual_sha256 = sha256_file(downloaded_path)
    if actual_size != int(shard["size_bytes"]):
        raise RuntimeError(
            f"Size mismatch for {shard_path}: "
            f"{actual_size} != {shard['size_bytes']}"
        )
    expected_sha256 = shard.get("lfs_sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {shard_path}: "
            f"{actual_sha256} != {expected_sha256}"
        )

    output_directory = images_root / "train"
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    extracted_bytes = 0
    with zipfile.ZipFile(downloaded_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            image_id = Path(member.filename).stem
            destination = output_directory / f"{image_id}{suffix}"
            if not destination.is_file() or destination.stat().st_size == 0:
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with archive.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                temporary.replace(destination)
            extracted_bytes += destination.stat().st_size
            rows.append(
                {
                    "id": str(image_id),
                    "official_split": "train",
                    "source_shard": shard_path,
                    "source_member": member.filename,
                    "source_crc32": int(member.CRC),
                    "image_path": str(destination),
                    "image_bytes": int(destination.stat().st_size),
                }
            )

    frame = pd.DataFrame(rows).sort_values("id", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No image members were found in {shard_path}.")
    if frame["id"].duplicated().any():
        raise RuntimeError(f"Duplicate image IDs inside {shard_path}.")
    shard_index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = shard_index_path.with_suffix(".parquet.partial")
    frame.to_parquet(temporary_index, index=False)
    temporary_index.replace(shard_index_path)

    payload = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_fingerprint": fingerprint,
        "shard_path": shard_path,
        "source_lfs_sha256": expected_sha256,
        "archive_sha256": actual_sha256,
        "archive_size_bytes": actual_size,
        "member_count": len(frame),
        "extracted_bytes": extracted_bytes,
        "shard_index": str(shard_index_path),
        "resumed_from_state": False,
    }
    atomic_json(state_path, payload)
    downloaded_path.unlink(missing_ok=True)
    return frame, payload


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    repo_id = str(inventory["repo_id"])
    revision = str(inventory["resolved_revision"])
    requested = sorted({canonical_shard(value) for value in args.train_shards})
    files = {str(item["path"]): item for item in inventory["files"]}
    missing = sorted(set(requested) - set(files))
    if missing:
        raise RuntimeError(f"Requested shards are absent from inventory: {missing}")
    selected = [files[path] for path in requested]
    fingerprint = selection_fingerprint(repo_id, revision, requested)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    args.images_root.mkdir(parents=True, exist_ok=True)
    args.state_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(int(args.workers), len(selected)))
    results: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_shard,
                repo_id=repo_id,
                revision=revision,
                shard=shard,
                dataset_root=args.dataset_root,
                images_root=args.images_root,
                state_root=args.state_root,
                fingerprint=fingerprint,
            )
            for shard in selected
        ]
        for future in as_completed(futures):
            results.append(future.result())

    member_index = pd.concat([item[0] for item in results], ignore_index=True)
    member_index = member_index.sort_values(
        ["source_shard", "id"], kind="stable"
    ).reset_index(drop=True)
    if member_index["id"].duplicated().any():
        raise RuntimeError("The selected train shards contain duplicate image IDs.")
    args.member_index.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = args.member_index.with_suffix(".parquet.partial")
    member_index.to_parquet(temporary_index, index=False)
    temporary_index.replace(args.member_index)

    states = [item[1] for item in results]
    payload = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "resolved_revision": revision,
        "selection_fingerprint": fingerprint,
        "train_shards": requested,
        "workers": worker_count,
        "archive_bytes": sum(int(item["archive_size_bytes"]) for item in states),
        "extracted_bytes": int(member_index["image_bytes"].sum()),
        "image_count": len(member_index),
        "member_index": str(args.member_index),
        "member_index_sha256": sha256_file(args.member_index),
        "shard_states": states,
    }
    atomic_json(args.stats_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
