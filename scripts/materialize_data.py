#!/usr/bin/env python3
"""Resolve Dataset4.0 assets and emit an absolute-path runtime parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_checksums(release_dir: Path) -> None:
    checksum_file = release_dir / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"missing release checksum file: {checksum_file}")
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = release_dir / name.strip()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="External root containing assets/student and assets/teacher.",
    )
    args = parser.parse_args()

    release_dir = args.release_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_parquet = release_dir / "train.parquet"
    archive = release_dir / "dataset4_0_assets.tar.zst"
    if not source_parquet.is_file():
        parser.error(f"required release file is missing: {source_parquet}")
    asset_root = args.asset_root.resolve() if args.asset_root else None
    if asset_root is not None and not asset_root.is_dir():
        parser.error(f"external asset root does not exist: {asset_root}")
    if asset_root is None and not archive.is_file():
        parser.error(
            "the code release does not bundle images; pass --asset-root or use "
            "a locally generated release containing dataset4_0_assets.tar.zst"
        )
    verify_release_checksums(release_dir)

    marker = output_dir / ".materialized.json"
    runtime_parquet = output_dir / "train.absolute.parquet"
    portable_parquet_sha256 = sha256_file(source_parquet)
    if asset_root is not None:
        asset_mode = "external"
        asset_fingerprint = hashlib.sha256(str(asset_root).encode("utf-8")).hexdigest()
    else:
        asset_mode = "archive"
        asset_fingerprint = sha256_file(archive)
    if marker.is_file() and runtime_parquet.is_file():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if (
            state.get("portable_parquet_sha256") == portable_parquet_sha256
            and state.get("asset_mode") == asset_mode
            and state.get("asset_fingerprint") == asset_fingerprint
        ):
            print(runtime_parquet)
            return
        raise RuntimeError(f"materialized data is stale: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"materialization output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if asset_root is None:
        listing = subprocess.check_output(
            ["tar", "--use-compress-program=unzstd", "-tf", str(archive)],
            text=True,
        ).splitlines()
        if not listing or any(
            name.startswith("/")
            or ".." in Path(name).parts
            or not name.startswith("assets/")
            for name in listing
        ):
            raise ValueError("asset archive contains an unsafe or unexpected path")
        subprocess.run(
            [
                "tar",
                "--use-compress-program=unzstd",
                "-xf",
                str(archive),
                "-C",
                str(output_dir),
            ],
            check=True,
        )
        resolved_asset_root = output_dir
    else:
        resolved_asset_root = asset_root

    source_table = pq.read_table(source_parquet)
    rows = source_table.to_pylist()
    for row in rows:
        for key in ("images", "teacher_images"):
            relative = Path(row[key][0]["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"non-portable image path in parquet: {relative}")
            absolute = resolved_asset_root / relative
            if not absolute.is_file():
                raise FileNotFoundError(f"asset root did not provide {relative}")
            row[key] = [{"image": str(absolute)}]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=source_table.schema),
        runtime_parquet,
        compression="zstd",
    )
    shutil.copy2(release_dir / "summary.json", output_dir / "release_summary.json")
    state = {
        "asset_mode": asset_mode,
        "asset_fingerprint": asset_fingerprint,
        "portable_parquet_sha256": portable_parquet_sha256,
        "rows": len(rows),
    }
    marker.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(runtime_parquet)


if __name__ == "__main__":
    main()
