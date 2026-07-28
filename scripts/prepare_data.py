#!/usr/bin/env python3
"""Build a relocatable Dataset4.0 release without changing the source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from release_safety import assert_public_text, assert_public_value

EXPECTED_ROWS = 5295
EXPECTED_MCQ_ROWS = 5205
EXPECTED_OPENQA_ROWS = 90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(item: dict) -> Path:
    value = item.get("image") or item.get("path")
    if not value:
        raise ValueError(f"image item has no path: {item!r}")
    return Path(value)


def add_file_to_tar(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    stat = source.stat()
    info = tarfile.TarInfo(arcname)
    info.size = stat.st_size
    info.mode = 0o644
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def write_asset_archive(
    archive_path: Path,
    assets: list[tuple[Path, str]],
    compression_level: int,
) -> None:
    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    partial.unlink(missing_ok=True)
    command = [
        "zstd",
        "-T0",
        f"-{compression_level}",
        "--quiet",
        "--force",
        "-o",
        str(partial),
        "-",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("failed to create zstd input pipe")
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            for source, arcname in assets:
                add_file_to_tar(archive, source, arcname)
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        partial.unlink(missing_ok=True)
        raise
    if return_code != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"zstd exited with status {return_code}")
    os.replace(partial, archive_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--data-card-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    for path in (
        args.source_parquet,
        args.selection_manifest,
        args.source_summary,
        args.data_card_template,
    ):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_table = pq.read_table(args.source_parquet)
    source_rows = source_table.to_pylist()
    if not args.allow_subset and len(source_rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(source_rows)}")

    raw_selection_rows = [
        json.loads(line)
        for line in args.selection_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_selection_rows) != len(source_rows):
        raise ValueError(
            f"selection manifest has {len(raw_selection_rows)} rows; "
            f"parquet has {len(source_rows)}"
        )
    selection_rows = []
    for index, selection in enumerate(raw_selection_rows):
        if not isinstance(selection, dict):
            raise ValueError(f"selection row {index} must be a mapping")
        assert_public_value(selection, f"selection[{index}]")
        selection_rows.append(
            {
                "source_dataset": selection.get("source_dataset"),
                "source_id": selection.get("source_id"),
                "task_family": selection.get("task_family"),
            }
        )

    assert_public_text(args.data_card_template, "data_card_template")

    portable_rows: list[dict] = []
    asset_records: list[dict] = []
    assets: list[tuple[Path, str]] = []
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for index, (row, selection) in enumerate(zip(source_rows, selection_rows, strict=True)):
        task_family = str((row.get("extra_info") or {}).get("task_family") or "")
        task_counts[task_family] += 1
        source_counts[str(selection.get("source_dataset") or "unknown")] += 1

        for role, key in (("student", "images"), ("teacher", "teacher_images")):
            items = row.get(key)
            if not isinstance(items, list) or len(items) != 1:
                raise ValueError(f"row {index} must contain exactly one {key} item")
            source = image_path(items[0])
            if not source.is_file():
                raise FileNotFoundError(f"row {index} missing {role} image: {source}")
            suffix = source.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError(f"row {index} has unsupported image extension: {source}")
            relative = f"assets/{role}/{index:06d}{suffix}"
            with Image.open(source) as image:
                width, height = image.size
                image.verify()
            record = {
                "row": index,
                "role": role,
                "path": relative,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "width": width,
                "height": height,
                "source_dataset": selection.get("source_dataset"),
                "source_id": selection.get("source_id"),
            }
            asset_records.append(record)
            assets.append((source, relative))
            row[key] = [{"image": relative}]

        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict) and "source_extra_info_json" in extra_info:
            # The original blob includes source-machine paths. Dataset/source IDs
            # remain in structured fields and in selection_manifest.jsonl.
            extra_info["source_extra_info_json"] = ""
        assert_public_value(row, f"row[{index}]")
        portable_rows.append(row)

    if not args.allow_subset:
        if task_counts["mcq"] != EXPECTED_MCQ_ROWS:
            raise ValueError(f"expected {EXPECTED_MCQ_ROWS} MCQ rows, found {task_counts['mcq']}")
        if task_counts["open_qa"] != EXPECTED_OPENQA_ROWS:
            raise ValueError(
                f"expected {EXPECTED_OPENQA_ROWS} OpenQA rows, found {task_counts['open_qa']}"
            )

    portable_table = pa.Table.from_pylist(portable_rows, schema=source_table.schema)
    parquet_path = args.output_dir / "train.parquet"
    pq.write_table(portable_table, parquet_path, compression="zstd")

    selection_path = args.output_dir / "selection_manifest.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for selection in selection_rows:
            handle.write(json.dumps(selection, ensure_ascii=True, sort_keys=True) + "\n")
    shutil.copy2(args.data_card_template, args.output_dir / "DATA_CARD.md")

    manifest_path = args.output_dir / "asset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in asset_records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    source_summary_name = source_summary.get("name", "Dataset4.0")
    assert_public_value(source_summary_name, "source_summary.name")
    summary = {
        "name": "RP-OPSD Dataset4.0 portable release",
        "rows": len(portable_rows),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "student_view": "physical half width and height",
        "teacher_view": "original full resolution",
        "asset_references": len(asset_records),
        "assets_included": not args.no_archive,
        "source_asset_bytes": sum(record["bytes"] for record in asset_records),
        "source_parquet_sha256": sha256_file(args.source_parquet),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "source_summary_name": source_summary_name,
        "release_metadata_safety_checked": True,
    }
    assert_public_value(summary, "summary")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive_path = args.output_dir / "dataset4_0_assets.tar.zst"
    if not args.no_archive:
        write_asset_archive(archive_path, assets, args.compression_level)

    checksum_targets = [
        parquet_path,
        selection_path,
        manifest_path,
        summary_path,
        args.output_dir / "DATA_CARD.md",
    ]
    if archive_path.is_file():
        checksum_targets.append(archive_path)
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksum_targets:
            handle.write(f"{sha256_file(path)}  {path.name}\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
