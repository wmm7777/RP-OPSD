#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
GENERATED_TEST_BYTECODE = (
    Path(__cached__).resolve() if globals().get("__cached__") else None
)


def is_generated_test_cache(path: Path) -> bool:
    if GENERATED_TEST_BYTECODE is None:
        return False
    resolved = path.resolve()
    if resolved == GENERATED_TEST_BYTECODE:
        return True
    if path.is_dir() and resolved == GENERATED_TEST_BYTECODE.parent:
        return all(
            child.resolve() == GENERATED_TEST_BYTECODE for child in path.iterdir()
        )
    return False


def import_script(name: str):
    path = PACKAGE_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReproductionPackageTests(unittest.TestCase):
    def test_public_package_contains_no_data_media_weights_or_archives(self):
        allowed_documentation_media = {"assets/rp_opsd_overview.png"}
        forbidden_endings = (
            ".arrow",
            ".avif",
            ".bin",
            ".bmp",
            ".ckpt",
            ".csv",
            ".gif",
            ".heic",
            ".jpeg",
            ".jpg",
            ".jsonl",
            ".npy",
            ".npz",
            ".parquet",
            ".pyc",
            ".pyo",
            ".pth",
            ".pt",
            ".png",
            ".safetensors",
            ".tif",
            ".tiff",
            ".webp",
            ".tar",
            ".tar.gz",
            ".tar.zst",
            ".tgz",
            ".zip",
            ".7z",
        )
        forbidden_magic = (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"BM",
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"PK\x07\x08",
            b"\x1f\x8b",
            b"\x28\xb5\x2f\xfd",
            b"7z\xbc\xaf\x27\x1c",
            b"Rar!\x1a\x07",
        )
        violations = []
        for path in PACKAGE_ROOT.rglob("*"):
            if is_generated_test_cache(path):
                continue
            if path.is_dir() and path.name in {
                "__pycache__",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
            }:
                violations.append(str(path.relative_to(PACKAGE_ROOT)))
                continue
            if path.is_symlink():
                violations.append(str(path.relative_to(PACKAGE_ROOT)))
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(PACKAGE_ROOT)
            if relative.as_posix() in allowed_documentation_media:
                continue
            if relative.parts[0] in {".runtime", "outputs"}:
                continue
            name = path.name.lower()
            with path.open("rb") as handle:
                head = handle.read(512)
            if name.endswith(forbidden_endings):
                violations.append(str(relative))
            elif any(head.startswith(magic) for magic in forbidden_magic):
                violations.append(str(relative))
            elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                violations.append(str(relative))
            elif len(head) >= 262 and head[257:262] == b"ustar":
                violations.append(str(relative))
        self.assertEqual(violations, [])

    def test_dataset_release_is_placeholder_only(self):
        release_dir = PACKAGE_ROOT / "release" / "dataset4_0"
        files = sorted(
            path.relative_to(release_dir).as_posix()
            for path in release_dir.rglob("*")
            if path.is_file()
        )
        self.assertEqual(files, ["README.md"])

    def test_public_package_contains_no_internal_identity_or_credentials(self):
        forbidden_markers = [
            "/" + "mnt" + "/" + "bn" + "/",
            "/" + "home" + "/" + "tiger" + "/",
            "/" + "Users" + "/" + "byted" + "ance" + "/",
            "wyc" + "." + "wyc",
            "wang" + "yuchen",
            "hub." + "byted" + ".org",
            "byted" + "ance",
            "byte" + "intl",
            "cloud" + "native",
            "ies_" + "content_algorithm",
            "mlx " + "worker",
            "mer" + "lin" + "-cli",
            "huggingface" + "-proxy",
        ]
        credential_patterns = [
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(r"\bASIA[0-9A-Z]{16}\b"),
            re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
            re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            re.compile(
                r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
            ),
        ]
        violations = []
        for path in PACKAGE_ROOT.rglob("*"):
            if is_generated_test_cache(path):
                continue
            if not path.is_file():
                continue
            relative_path = path.relative_to(PACKAGE_ROOT)
            relative = relative_path.as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            if relative_path.parts[0] != "verl":
                for marker in forbidden_markers:
                    if marker.lower() in lowered:
                        violations.append(f"{relative}: {marker}")
            for pattern in credential_patterns:
                if pattern.search(text):
                    violations.append(f"{relative}: {pattern.pattern}")
        self.assertEqual(violations, [])

    def test_expected_configuration(self):
        text = (PACKAGE_ROOT / "config" / "best.env").read_text(encoding="utf-8")
        expected = {
            "RP_OPSD_MODEL_HIDDEN_SIZE=4096",
            "RP_OPSD_DATASET_ROWS=5295",
            "RP_OPSD_ROLLOUT_N=8",
            "RP_OPSD_TRAIN_BATCH_SIZE=96",
            "RP_OPSD_TOTAL_STEPS=55",
            "RP_OPSD_TOPK=100",
            "RP_OPSD_ALPHA=1.0",
            "RP_OPSD_TEACHER_UPDATE_RATE=0.05",
        }
        for line in expected:
            self.assertIn(line, text)

    def test_native_source_is_embedded_without_upstream_staging(self):
        required = [
            "verl/trainer/main_ppo.py",
            "verl/trainer/ppo/core_algos.py",
            "eval/infer.py",
            "eval/judge_qwenlm.py",
            "chat_templates/perception_chat_template_qwen35.jinja",
            "scripts/run_rp_opsd.sh",
            "pyproject.toml",
        ]
        for relative in required:
            self.assertTrue((PACKAGE_ROOT / relative).is_file(), relative)
        self.assertFalse((PACKAGE_ROOT / "patches").exists())
        self.assertFalse((PACKAGE_ROOT / "scripts/stage_source.sh").exists())

        operational_paths = [
            PACKAGE_ROOT / "config",
            PACKAGE_ROOT / "scripts",
            PACKAGE_ROOT / "run.sh",
        ]
        forbidden = [
            "VisionOPD/Vision-OPD",
            "RP_OPSD_VISION_OPD_",
            "stage_source.sh",
            "--upstream-source",
            "--source-dir",
            "git clone",
        ]
        for root in operational_paths:
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in paths:
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for marker in forbidden:
                    self.assertNotIn(marker, text, f"{marker} in {path}")

    def test_bias_corrected_reverse_kl_is_zero_for_equal_distributions(self):
        student = [0.2, 0.3, 0.1]
        teacher = list(student)
        value = sum(
            ps * math.log(ps / pt) - ps + pt
            for ps, pt in zip(student, teacher, strict=True)
        )
        self.assertAlmostEqual(value, 0.0, places=12)

    def test_bias_corrected_reverse_kl_is_positive(self):
        student = [0.4, 0.1, 0.05]
        teacher = [0.2, 0.25, 0.1]
        value = sum(
            ps * math.log(ps / pt) - ps + pt
            for ps, pt in zip(student, teacher, strict=True)
        )
        self.assertGreater(value, 0.0)

    def test_prepare_data_rewrites_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            student = root / "student.png"
            teacher = root / "teacher.png"
            Image.new("RGB", (8, 6), color="white").save(student)
            Image.new("RGB", (16, 12), color="black").save(teacher)
            row = {
                "data_source": "test",
                "prompt": [{"role": "user", "content": "<image>\nQuestion?"}],
                "images": [{"image": str(student)}],
                "teacher_images": [{"image": str(teacher)}],
                "ability": "image_reasoning_mcq",
                "reward_model": {"style": "rule", "ground_truth": "A"},
                "extra_info": {
                    "task_family": "mcq",
                    "source_extra_info_json": json.dumps({"path": str(student)}),
                },
            }
            source_parquet = root / "source.parquet"
            pq.write_table(pa.Table.from_pylist([row]), source_parquet)
            selection = root / "selection.jsonl"
            selection.write_text(
                json.dumps(
                    {
                        "source_dataset": "synthetic",
                        "source_id": "0",
                        "task_family": "mcq",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text('{"name": "synthetic"}\n', encoding="utf-8")
            data_card = root / "DATA_CARD.md"
            data_card.write_text("# Test\n", encoding="utf-8")
            output = root / "release"
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "prepare_data.py"),
                    "--source-parquet",
                    str(source_parquet),
                    "--selection-manifest",
                    str(selection),
                    "--source-summary",
                    str(summary),
                    "--data-card-template",
                    str(data_card),
                    "--output-dir",
                    str(output),
                    "--no-archive",
                    "--allow-subset",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            portable = pq.read_table(output / "train.parquet").to_pylist()[0]
            self.assertEqual(portable["images"][0]["image"], "assets/student/000000.png")
            self.assertEqual(
                portable["teacher_images"][0]["image"],
                "assets/teacher/000000.png",
            )
            self.assertNotIn("/tmp/", json.dumps(portable))
            published_selection = json.loads(
                (output / "selection_manifest.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(published_selection),
                {"source_dataset", "source_id", "task_family"},
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "validate_release.py"),
                    "--release-dir",
                    str(output),
                    "--allow-subset",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            hidden_file = output / ".env"
            hidden_file.write_text(
                "API_" + "KEY=test-only\n"
                + "OUTPUT=/"
                + "home"
                + "/example/private\n",
                encoding="utf-8",
            )
            rejected_release = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "validate_release.py"),
                    "--release-dir",
                    str(output),
                    "--allow-subset",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected_release.returncode, 0)
            self.assertIn("unexpected release files", rejected_release.stderr)
            hidden_file.unlink()

            unsafe_selection = root / "unsafe_selection.jsonl"
            unsafe_selection.write_text(
                json.dumps(
                    {
                        "source_dataset": "synthetic",
                        "source_id": "0",
                        "task_family": "mcq",
                        "workspace": "/" + "mnt" + "/" + "private" + "/" + "project",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "prepare_data.py"),
                    "--source-parquet",
                    str(source_parquet),
                    "--selection-manifest",
                    str(unsafe_selection),
                    "--source-summary",
                    str(summary),
                    "--data-card-template",
                    str(data_card),
                    "--output-dir",
                    str(root / "unsafe_release"),
                    "--no-archive",
                    "--allow-subset",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("absolute user or workspace path", rejected.stderr)

    def test_release_safety_rejects_private_values(self):
        module = import_script("release_safety.py")
        module.assert_public_value("https://github.com/verl-project/verl")
        with self.assertRaisesRegex(ValueError, "absolute user or workspace path"):
            module.assert_public_value(
                {"path": "/" + "home" + "/" + "example" + "/" + "private"}
            )
        with self.assertRaisesRegex(ValueError, "absolute user or workspace path"):
            module.assert_public_value(
                "/" + "scratch" + "/" + "example" + "/" + "run/output.json"
            )
        with self.assertRaisesRegex(ValueError, "unsupported URL scheme"):
            module.assert_public_value("hdfs" + "://namenode.example/data")
        with self.assertRaisesRegex(ValueError, "private or non-public hostname"):
            module.assert_public_value("worker-17.cluster." + "internal")
        with self.assertRaisesRegex(ValueError, "URL host is not allowlisted"):
            module.assert_public_value("https://private.example.invalid/artifact")
        with self.assertRaisesRegex(ValueError, "credential-like value"):
            module.assert_public_value("Bearer " + "A" * 32)
        with self.assertRaisesRegex(ValueError, "sensitive field"):
            module.assert_public_value({"api_" + "key": "test-only-value"})

    def test_metric_collection_protocol(self):
        module = import_script("collect_metrics.py")
        records = [
            {"judge": "Yes", "category": "Easy"},
            {"judge": "No", "category": "Easy"},
            {"judge": "Yes", "category": "Medium"},
            {"judge": "Yes", "category": "Hard"},
        ]
        self.assertAlmostEqual(module.score(records, "visualprobe"), 83.3333333333)
        self.assertEqual(module.score([{"judge": "Yes"}, {"judge": "No"}], "mmstar"), 50.0)

    def test_materialize_data_accepts_external_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release = root / "release"
            asset_root = root / "external_assets"
            output = root / "runtime"
            student = asset_root / "assets/student/000000.png"
            teacher = asset_root / "assets/teacher/000000.png"
            student.parent.mkdir(parents=True)
            teacher.parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), color="white").save(student)
            Image.new("RGB", (16, 12), color="black").save(teacher)

            row = {
                "images": [{"image": "assets/student/000000.png"}],
                "teacher_images": [{"image": "assets/teacher/000000.png"}],
            }
            release.mkdir()
            parquet = release / "train.parquet"
            pq.write_table(pa.Table.from_pylist([row]), parquet)
            digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
            (release / "SHA256SUMS").write_text(
                f"{digest}  train.parquet\n",
                encoding="utf-8",
            )
            (release / "summary.json").write_text(
                '{"assets_included": false}\n',
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "materialize_data.py"),
                    "--release-dir",
                    str(release),
                    "--asset-root",
                    str(asset_root),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            runtime = Path(completed.stdout.strip())
            materialized = pq.read_table(runtime).to_pylist()[0]
            self.assertEqual(
                materialized["images"][0]["image"],
                str(student.resolve()),
            )
            self.assertEqual(
                materialized["teacher_images"][0]["image"],
                str(teacher.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
