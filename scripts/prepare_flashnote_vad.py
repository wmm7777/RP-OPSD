#!/usr/bin/env python3
"""flash_note → VAD 三视图 parquet 转换。

输入: RP-OPSD/.runtime/flashnote_summary/train.parquet
  列: prompt, images, teacher_images, extra_info

输出: train_vad.parquet（VAD 格式）
  列: prompt, images, bbox_images, degraded_bbox_images, data_source, ability, reward_model, extra_info

设计（§14 修正）:
  - images = 原图（学生视图，师生同图，无分辨率特权）
  - bbox_images = 原图（教师正视图 = 学生图，纯擦除对比）
  - degraded_bbox_images = degrade_image(原图)（10% 双线性下采样+最近邻上采样）

退化逻辑复用 VAD 的 degrade_image（prepare_data.py:156-182）。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from PIL import Image

TRAIN_PARQUET = "/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/train.parquet"
OUTPUT_PARQUET = "/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/train_vad.parquet"
DEGRADED_ROOT = "/data4/wumeimei/flash_note/train/degraded_images"
SCALE = 0.10
WORKERS = 16


def degrade_one(args):
    src_path, dst_path, scale = args
    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst), "reused"
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size
            low_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            low = img.resize(low_size, Image.Resampling.BILINEAR)
            result = low.resize((w, h), Image.Resampling.NEAREST)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst, format="PNG")
        return str(dst), "created"
    except Exception as e:
        return str(src), f"ERROR: {e}"


def get_image_path(images_field):
    """从 parquet 的 images 列提取路径。numpy array / list / dict 均支持。"""
    if hasattr(images_field, "tolist"):
        images_field = images_field.tolist()
    if isinstance(images_field, (list, tuple)) and len(images_field) > 0:
        item = images_field[0]
        if isinstance(item, dict):
            return item.get("image") or item.get("path") or item.get("url")
        if isinstance(item, str):
            return item
    if isinstance(images_field, dict):
        return images_field.get("image") or images_field.get("path")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=TRAIN_PARQUET)
    ap.add_argument("--output", default=OUTPUT_PARQUET)
    ap.add_argument("--degraded-root", default=DEGRADED_ROOT)
    ap.add_argument("--scale", type=float, default=SCALE)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=None, help="smoke test 行数限制")
    args = ap.parse_args()

    print(f"[prepare_flashnote_vad] input={args.input}")
    print(f"[prepare_flashnote_vad] output={args.output}")
    print(f"[prepare_flashnote_vad] degraded_root={args.degraded_root}")
    print(f"[prepare_flashnote_vad] scale={args.scale} workers={args.workers}")

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"[prepare_flashnote_vad] {len(df)} rows")

    # 收集所有原图路径
    tasks = []
    records = []
    for idx, row in df.iterrows():
        src_path = get_image_path(row["images"])
        if not src_path or not os.path.exists(src_path):
            print(f"  WARN: idx={idx} image not found: {src_path}")
            continue
        # 退化图路径：degraded_root/<原文件名>.png
        src_name = Path(src_path).stem
        # 保留语种子目录
        parent_name = Path(src_path).parent.name  # e.g. en_image_lr
        dst_path = Path(args.degraded_root) / parent_name / f"{src_name}.png"
        tasks.append((src_path, str(dst_path), args.scale))
        records.append((idx, src_path, str(dst_path)))

    print(f"[prepare_flashnote_vad] {len(tasks)} images to degrade")

    # 并行生成退化图
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(degrade_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            dst, status = fut.result()
            src = futures[fut]  # 原始 src_path
            results[src] = status
            done += 1
            if done % 2000 == 0:
                print(f"  degraded {done}/{len(tasks)}")

    n_errors = sum(1 for s in results.values() if s.startswith("ERROR"))
    print(f"[prepare_flashnote_vad] degraded done: {len(results)} total, {n_errors} errors")

    # 构建 VAD parquet
    out_rows = []
    for idx, src_path, dst_path in records:
        if src_path not in results or results[src_path].startswith("ERROR"):
            continue
        row = df.loc[idx]
        prompt = row["prompt"]
        # VAD 格式：images/bbox_images/degraded_bbox_images 都是 [{'image': path}] 格式
        img_entry = [{"image": src_path}]
        bbox_entry = [{"image": src_path}]  # 教师正视图 = 原图（无 bbox）
        degraded_entry = [{"image": dst_path}]
        extra = row.get("extra_info", {})
        if not isinstance(extra, dict):
            extra = {"index": int(idx), **extra} if isinstance(extra, dict) else {"index": int(idx)}
        extra["vad_task"] = "flash_note_summary"
        out_rows.append({
            "prompt": prompt,
            "images": img_entry,
            "bbox_images": bbox_entry,
            "degraded_bbox_images": degraded_entry,
            "data_source": "flash_note_summary",
            "ability": "image_summarization",
            "reward_model": {"style": "none"},
            "extra_info": extra,
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(args.output, index=False)
    print(f"[prepare_flashnote_vad] wrote {len(out_df)} rows to {args.output}")

    # 校验输出列
    required = {"prompt", "images", "bbox_images", "degraded_bbox_images", "data_source", "ability", "reward_model", "extra_info"}
    missing = required - set(out_df.columns)
    assert not missing, f"missing columns: {missing}"
    print(f"[prepare_flashnote_vad] columns OK: {list(out_df.columns)}")
    # 抽样校验
    sample = out_df.iloc[0]
    print(f"  sample images: {sample['images']}")
    print(f"  sample bbox_images: {sample['bbox_images']}")
    print(f"  sample degraded: {sample['degraded_bbox_images']}")
    print(f"  sample data_source: {sample['data_source']}")


if __name__ == "__main__":
    main()
