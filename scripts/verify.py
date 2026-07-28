#!/usr/bin/env python3
"""Validate the pinned RP-OPSD model, data, native source, and runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_ROWS = 5295
EXPECTED_MCQ_ROWS = 5205
EXPECTED_OPENQA_ROWS = 90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(row: dict[str, Any], key: str, data_root: Path) -> Path:
    items = row.get(key)
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError(f"{key} must contain exactly one image")
    item = items[0]
    if not isinstance(item, dict):
        raise ValueError(f"{key}[0] must be a mapping")
    value = item.get("image") or item.get("path")
    if not value:
        raise ValueError(f"{key}[0] has no image path")
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path


def has_portable_image_path(row: dict[str, Any], key: str) -> bool:
    items = row.get(key)
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        return False
    value = items[0].get("image") or items[0].get("path")
    return bool(value) and not Path(str(value)).is_absolute()


def read_hidden_size(model_path: Path) -> tuple[int, str]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = [
        config.get("hidden_size"),
        (config.get("text_config") or {}).get("hidden_size"),
        (config.get("llm_config") or {}).get("hidden_size"),
    ]
    hidden_size = next((value for value in candidates if isinstance(value, int)), None)
    if hidden_size is None:
        raise ValueError(f"cannot find hidden_size in {config_path}")
    model_type = str(
        config.get("model_type")
        or (config.get("text_config") or {}).get("model_type")
        or "unknown"
    )
    return hidden_size, model_type


def verify_source(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "provenance/source_files.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"native source manifest is missing: {manifest_path}")
    checked = 0
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"invalid source manifest line {line_number}: {raw_line}"
            ) from exc
        relative_path = relative.lstrip("* ")
        path = source_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"native source file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"native source checksum mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"native source manifest is empty: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    config_path = source_dir / "config/best.env"
    config_prefix = "RP_OPSD_SOURCE_MANIFEST_SHA256="
    configured_manifest = next(
        (
            line.removeprefix(config_prefix)
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(config_prefix)
        ),
        None,
    )
    if configured_manifest != manifest_sha256:
        raise ValueError(
            "configured source manifest hash does not match "
            f"{manifest_path}: expected {configured_manifest}, got {manifest_sha256}"
        )

    required_fragments = {
        source_dir / "verl/trainer/ppo/core_algos.py": (
            'distillation_objective == "mopd_topk_reverse_kl"',
            "bias_correction = teacher_probs - student_probs",
        ),
        source_dir / "verl/workers/actor/dp_actor.py": (
            'distillation_topk_source == "teacher"',
            "teacher_topk_indices",
        ),
        source_dir / "verl/workers/config/actor.py": (
            'distillation_topk_source: str = "student"',
            "mopd_topk_reverse_kl requires alpha=1.0",
        ),
        source_dir / "eval/infer.py": (
            'parser.add_argument("--top_p", default=1.0, type=float)',
            "top_p=args.top_p",
            "seed=args.seed",
        ),
    }
    for path, fragments in required_fragments.items():
        if not path.is_file():
            raise FileNotFoundError(f"native source file is missing: {path}")
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                raise ValueError(f"expected RP-OPSD fragment not found in {path}: {fragment}")
    return {
        "source_layout": "native",
        "manifest": manifest_path.relative_to(source_dir).as_posix(),
        "manifest_sha256": manifest_sha256,
        "files_checked": checked,
    }


def verify_data(
    parquet_path: Path,
    data_root: Path,
    quick: bool,
    allow_subset: bool,
) -> dict[str, Any]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"training parquet does not exist: {parquet_path}")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if not allow_subset and len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")

    task_counts: Counter[str] = Counter()
    missing_files: list[str] = []
    dimension_errors: list[str] = []
    prompt_errors: list[str] = []
    sample_indices = range(min(32, len(rows))) if quick else range(len(rows))

    for index, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        task_counts[str(extra_info.get("task_family") or "")] += 1
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1:
            prompt_errors.append(f"row {index}: prompt must have one user message")
        else:
            content = str(prompt[0].get("content") or "")
            if prompt[0].get("role") != "user" or content.count("<image>") != 1:
                prompt_errors.append(f"row {index}: prompt must contain one <image>")
            if "red bounding box" in content.lower():
                prompt_errors.append(f"row {index}: obsolete red-box instruction")

    for index in sample_indices:
        row = rows[index]
        student_path = image_path(row, "images", data_root)
        teacher_path = image_path(row, "teacher_images", data_root)
        if not student_path.is_file():
            missing_files.append(str(student_path))
            continue
        if not teacher_path.is_file():
            missing_files.append(str(teacher_path))
            continue
        with Image.open(student_path) as student_image:
            student_image.load()
            student_size = student_image.size
        with Image.open(teacher_path) as teacher_image:
            teacher_image.load()
            teacher_size = teacher_image.size
        expected_student = (
            max(1, teacher_size[0] // 2),
            max(1, teacher_size[1] // 2),
        )
        if any(
            abs(actual - expected) > 1
            for actual, expected in zip(student_size, expected_student, strict=True)
        ):
            dimension_errors.append(
                f"row {index}: student={student_size}, teacher={teacher_size}, "
                f"expected~={expected_student}"
            )

    if prompt_errors:
        raise ValueError("prompt validation failed: " + "; ".join(prompt_errors[:8]))
    if missing_files:
        raise FileNotFoundError("missing image files: " + "; ".join(missing_files[:8]))
    if dimension_errors:
        raise ValueError("resolution validation failed: " + "; ".join(dimension_errors[:8]))
    if not allow_subset:
        if task_counts["mcq"] != EXPECTED_MCQ_ROWS:
            raise ValueError(f"expected {EXPECTED_MCQ_ROWS} MCQ rows, found {task_counts['mcq']}")
        if task_counts["open_qa"] != EXPECTED_OPENQA_ROWS:
            raise ValueError(
                f"expected {EXPECTED_OPENQA_ROWS} OpenQA rows, found {task_counts['open_qa']}"
            )

    return {
        "parquet": parquet_path.name,
        "parquet_sha256": sha256_file(parquet_path),
        "data_root": data_root.name,
        "rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "images_checked": len(list(sample_indices)) * 2,
        "quick": quick,
        "portable_paths": all(
            has_portable_image_path(row, key)
            for row in rows
            for key in ("images", "teacher_images")
        ),
    }


def verify_runtime() -> dict[str, Any]:
    import ray
    import torch
    import transformers
    import vllm

    expected = {
        "torch": "2.10.0",
        "transformers": "5.5.0",
        "vllm": "0.18.0",
        "ray": "2.53.0",
    }
    actual = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "ray": ray.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
    }
    for key, expected_version in expected.items():
        if actual[key] != expected_version:
            raise ValueError(
                f"{key} version mismatch: expected {expected_version}, got {actual[key]}"
            )
    if sys.version_info[:2] != (3, 12):
        raise ValueError(f"Python 3.12 is required, got {actual['python']}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-parquet", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--require-8-gpus", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not any((args.model_path, args.data_parquet, args.check_source, args.check_runtime)):
        parser.error("at least one check must be requested")

    result: dict[str, Any] = {"status": "ok"}
    if args.model_path:
        model_path = args.model_path.resolve()
        hidden_size, model_type = read_hidden_size(model_path)
        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise ValueError(
                f"wrong model size: expected hidden_size={EXPECTED_HIDDEN_SIZE} "
                f"(Qwen3.5-9B), got {hidden_size}"
            )
        result["model"] = {
            "path": model_path.name,
            "hidden_size": hidden_size,
            "model_type": model_type,
        }
    if args.data_parquet:
        parquet_path = args.data_parquet.resolve()
        data_root = (
            args.data_root.resolve() if args.data_root else parquet_path.parent.resolve()
        )
        result["data"] = verify_data(
            parquet_path,
            data_root,
            quick=args.quick,
            allow_subset=args.allow_subset,
        )
    if args.check_source:
        result["source"] = verify_source(Path(__file__).resolve().parents[1])
    if args.check_runtime:
        result["runtime"] = verify_runtime()
        if args.require_8_gpus and result["runtime"]["cuda_devices"] != 8:
            raise ValueError(
                f"exactly 8 visible GPUs are required, got {result['runtime']['cuda_devices']}"
            )

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
