#!/usr/bin/env python3
"""flash_note summary jsonl -> SFT 对照数据（gold summary 监督）

================================================================================
 目的：隔离训练范式变量
   同 Qwen3.5-9B / 同 72k 采样(seed42) / 同半分辨率训练视图 / 同原图评测，
   唯一变量 = 训练范式：
     - RP-OPSD：reward-free 自蒸馏（on-policy + Teacher Top-100 reverse KL）
     - SFT    ：gold summary 监督（reward-dependent，标准 teacher forcing）

 gold summary 来源
   自探测源 jsonl 字段，优先级：
     1) messages 里 role=assistant 的 content
     2) 顶层 summary / response / answer / gold / target / output
     3) teacher_prompt（str，或 dict 的 summary/content/response/text/answer）
   源 jsonl: /data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl

 采样一致性（关键）
   与 convert_flashnote_summary.py 完全一致：seed42、每门封顶 2.2W、同遍历顺序、
   rng.sample 同位置选择 → SFT 与 RP-OPSD 用的是同一批 72k 数据。

 4 语种 prompt 模板
   复用 convert_flashnote_summary.py 的 PROMPT_TEMPLATE_<lang>（与 RP-OPSD 同 prompt）。

 训练图
   默认半分辨率（与 RP-OPSD student 训练视图一致，公平隔离范式）；
   --hr 切原图做 SFT-HR 上界对照。

 输出 swift SFT jsonl
   {"messages":[{"role":"user","content":"<image>\\n<语种模板>"},
                 {"role":"assistant","content":"<gold summary>"}],
    "images":["<半分辨率图 or 原图>"]}

 用法
   python convert_flashnote_sft.py --out .runtime/flashnote_summary/sft_train.jsonl
   python convert_flashnote_sft.py --out ... --hr          # 训练图用原图（上界对照）
   python convert_flashnote_sft.py --out ... --show-sample # 打印样例
================================================================================
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# 复用 RP-OPSD 转换脚本的常量与 4 语种模板（同目录），保证 prompt 与采样一致
from convert_flashnote_summary import (  # noqa: E402
    PROMPT_TEMPLATES,
    SRC_DEFAULT,
    detect_lang,
    lr_path,
)


def find_gold(d: dict) -> str | None:
    """自探测 gold summary 字段。运行时打印命中分布，便于核对。"""
    # 1) messages 里的 assistant
    for m in d.get("messages", []) or []:
        if m.get("role") == "assistant":
            c = str(m.get("content", "")).strip()
            if c:
                return c
    # 2) 顶层常见字段
    for k in ("summary", "response", "answer", "gold", "target", "output"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # 3) teacher_prompt（可能是 str，也可能是 dict）
    tp = d.get("teacher_prompt")
    if isinstance(tp, str) and tp.strip():
        return tp
    if isinstance(tp, dict):
        for k in ("summary", "content", "response", "text", "answer"):
            v = tp.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="flash_note summary -> SFT 对照 jsonl")
    p.add_argument("--src", default=SRC_DEFAULT, help="summary aligned jsonl 路径")
    p.add_argument("--out", required=True, help="输出 SFT jsonl 路径")
    p.add_argument(
        "--max-per-lang", type=int, default=22000, help="每门封顶（默认 22000，与 RP-OPSD 同）"
    )
    p.add_argument("--seed", type=int, default=42, help="采样种子（与 RP-OPSD 同 seed42）")
    p.add_argument(
        "--hr",
        action="store_true",
        help="训练图用原图（SFT-HR 上界对照）；默认半分辨率（与 RP-OPSD student 同视图）",
    )
    p.add_argument("--show-sample", action="store_true", help="打印一条样例")
    args = p.parse_args()

    src = Path(args.src)
    by_lang: dict[str, list[tuple[dict, str]]] = {}
    bad_no_gold = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            imgs = d.get("images")
            if not imgs or not isinstance(imgs, list) or not imgs:
                continue
            lang = detect_lang(imgs[0])
            if lang is None:
                continue
            gold = find_gold(d)
            if not gold:
                bad_no_gold += 1
                continue
            by_lang.setdefault(lang, []).append((d, gold))

    print("[read] 采样前语种分布:")
    for k in sorted(by_lang):
        print(f"  {k}: {len(by_lang[k])}")
    if bad_no_gold:
        print(f"  [warn] 无 gold summary 跳过: {bad_no_gold}")

    # 与 RP-OPSD 完全同采样：seed42、每门封顶 2.2W、rng.sample 同位置选择
    for lang in list(by_lang):
        if len(by_lang[lang]) > args.max_per_lang:
            rng = random.Random(args.seed)
            by_lang[lang] = rng.sample(by_lang[lang], args.max_per_lang)
            print(f"[sample] {lang} -> {args.max_per_lang}")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    first: tuple | None = None
    with out_p.open("w", encoding="utf-8") as fo:
        for lang in sorted(by_lang):
            template = PROMPT_TEMPLATES[lang]
            for d, gold in by_lang[lang]:
                orig = d["images"][0]
                img = orig if args.hr else lr_path(orig)
                fo.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": template},
                                {"role": "assistant", "content": gold},
                            ],
                            "images": [img],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if first is None:
                    first = (lang, orig, img, gold[:300])
                n += 1

    print(f"[done] 写出 {n} 行 -> {out_p}  (训练图={'原图' if args.hr else '半分辨率'})")
    if first:
        print(f"[sample] lang={first[0]}")
        print(f"  orig={first[1]}")
        print(f"  train_img={first[2]}")
        print(f"  gold[:300]={first[3]}")
    if args.show_sample and first:
        print(f"  user_template[:200]={PROMPT_TEMPLATES[first[0]][:200]}")


if __name__ == "__main__":
    main()
