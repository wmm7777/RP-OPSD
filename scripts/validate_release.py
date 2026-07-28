#!/usr/bin/env python3
"""Validate a portable Dataset4.0 release and optional extracted assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from release_safety import assert_public_text, assert_public_value

EXPECTED_ROWS = 5295
EXPECTED_MCQ_ROWS = 5205
EXPECTED_OPENQA_ROWS = 90
EXPECTED_ASSETS = 10590
IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-subset", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    release_dir = args.release_dir.resolve()
    required_files = {
        "DATA_CARD.md",
        "SHA256SUMS",
        "asset_manifest.jsonl",
        "selection_manifest.jsonl",
        "summary.json",
        "train.parquet",
    }
    optional_files = {"dataset4_0_assets.tar.zst"}
    discovered_files = set()
    for path in release_dir.rglob("*"):
        relative = path.relative_to(release_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"release contains a symlink: {relative}")
        if path.is_file():
            discovered_files.add(relative)
        elif not path.is_dir():
            raise ValueError(f"release contains a special file: {relative}")
    missing_files = required_files - discovered_files
    unexpected_files = discovered_files - required_files - optional_files
    if missing_files:
        raise ValueError(f"release files are missing: {sorted(missing_files)}")
    if unexpected_files:
        raise ValueError(f"unexpected release files: {sorted(unexpected_files)}")

    summary = json.loads((release_dir / "summary.json").read_text(encoding="utf-8"))
    assert_public_value(summary, "summary.json")
    assert_public_text(release_dir / "DATA_CARD.md")
    declared_assets_included = bool(summary.get("assets_included"))
    bundled_images = [
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if bundled_images:
        raise ValueError(
            "metadata release contains image files: " + ", ".join(bundled_images[:8])
        )

    checksum_file = release_dir / "SHA256SUMS"
    checksums = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            name = name.strip()
            if name in checksums:
                raise ValueError(f"duplicate checksum entry: {name}")
            checksums[name] = expected
    expected_checksum_files = discovered_files - {"SHA256SUMS"}
    if set(checksums) != expected_checksum_files:
        raise ValueError(
            "checksum entries do not exactly match release files: "
            f"expected={sorted(expected_checksum_files)}, "
            f"found={sorted(checksums)}"
        )
    for name, expected in checksums.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checksum path: {name}")
        actual = sha256_file(release_dir / relative)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {name}: {actual} != {expected}")

    rows = pq.read_table(release_dir / "train.parquet").to_pylist()
    if not rows:
        raise ValueError("training parquet is empty")
    if not args.allow_subset and len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    task_counts: Counter[str] = Counter()
    parquet_paths = []
    for index, row in enumerate(rows):
        task_counts[str((row.get("extra_info") or {}).get("task_family") or "")] += 1
        assert_public_value(row, f"train.parquet[{index}]")
        for key in ("images", "teacher_images"):
            items = row.get(key)
            if not isinstance(items, list) or len(items) != 1:
                raise ValueError(f"row {index} has invalid {key}")
            relative = Path(items[0]["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"row {index} has non-portable path: {relative}")
            parquet_paths.append(relative.as_posix())
    if not args.allow_subset and task_counts != Counter(
        {"mcq": EXPECTED_MCQ_ROWS, "open_qa": EXPECTED_OPENQA_ROWS}
    ):
        raise ValueError(f"unexpected task counts: {task_counts}")

    asset_records = [
        json.loads(line)
        for line in (release_dir / "asset_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_assets = len(rows) * 2 if args.allow_subset else EXPECTED_ASSETS
    if len(asset_records) != expected_assets:
        raise ValueError(
            f"expected {expected_assets} assets, found {len(asset_records)}"
        )
    assert_public_value(asset_records, "asset_manifest.jsonl")
    manifest_paths = [record["path"] for record in asset_records]
    for value in manifest_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"asset manifest has unsafe path: {value}")
    if sorted(manifest_paths) != sorted(parquet_paths):
        raise ValueError("asset manifest paths do not match parquet image paths")

    selection_records = [
        json.loads(line)
        for line in (release_dir / "selection_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_selection_rows = len(rows) if args.allow_subset else EXPECTED_ROWS
    if len(selection_records) != expected_selection_rows:
        raise ValueError(
            f"expected {expected_selection_rows} selection rows, "
            f"found {len(selection_records)}"
        )
    assert_public_value(selection_records, "selection_manifest.jsonl")

    archive = release_dir / "dataset4_0_assets.tar.zst"
    if declared_assets_included != archive.is_file():
        raise ValueError(
            "summary assets_included does not match dataset4_0_assets.tar.zst presence"
        )
    archive_paths = []
    if archive.is_file():
        archive_paths = subprocess.check_output(
            ["tar", "--use-compress-program=unzstd", "-tf", str(archive)],
            text=True,
        ).splitlines()
        if sorted(archive_paths) != sorted(manifest_paths):
            raise ValueError("archive members do not match the asset manifest")

    deep_checked = 0
    if args.materialized_root:
        materialized_root = args.materialized_root.resolve()
        for record in asset_records:
            path = materialized_root / record["path"]
            if not path.is_file():
                raise FileNotFoundError(f"materialized asset is missing: {path}")
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"asset byte size mismatch: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"asset checksum mismatch: {path}")
            with Image.open(path) as image:
                image.load()
                if list(image.size) != [record["width"], record["height"]]:
                    raise ValueError(f"asset dimensions mismatch: {path}")
            deep_checked += 1

    result = {
        "status": "ok",
        "release_dir": release_dir.name,
        "rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "asset_references": len(asset_records),
        "assets_included": declared_assets_included,
        "bundled_image_files": len(bundled_images),
        "archive_members": len(archive_paths),
        "portable_paths": True,
        "sensitive_metadata": False,
        "deep_assets_checked": deep_checked,
        "release_checksums_checked": len(checksums),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
