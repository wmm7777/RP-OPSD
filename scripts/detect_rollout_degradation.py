#!/usr/bin/env python3
"""
Rollout 重复与乱码检测脚本
扫描 rollout JSONL 文件，检测三类退化信号：
  1. word_rep:  词级 n-gram 重复 (phrase-looping)
  2. char_rep:  字符级连续重复 (stuttering)
  3. gibberish: 乱码 (无意义字符/编码损坏)

用法:
  python scripts/detect_rollout_degradation.py --rollout-dir outputs/flashnote_train_v5_teacher27B/rollouts --sample-every 5
"""

import argparse
import glob
import json
import os
from collections import Counter

# 允许的常规标点（中英文）
ALLOWED_PUNCT = set(".,;:!?\'\"()[]{}「」（）【】《》""''—…·-、。！？：；")


def detect_word_repetition(text, min_n=4, max_n=8):
    """检测词级 n-gram 重复，返回最大重复次数。"""
    words = text.split()
    if len(words) < max_n:
        return 0
    max_repeats = 0
    for n in range(min_n, max_n + 1):
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if counts:
            mx = max(counts.values())
            if mx > max_repeats:
                max_repeats = mx
    return max_repeats


def detect_char_repetition(text):
    """检测字符级连续重复，返回最长连续相同字符数。"""
    max_rep = 0
    cur = ""
    cnt = 0
    for c in text:
        if c == cur:
            cnt += 1
            if cnt > max_rep:
                max_rep = cnt
        else:
            cur = c
            cnt = 1
    return max_rep


def detect_gibberish(text):
    """检测乱码：非常规字符占比过高或出现 U+FFFD 替换符。"""
    if not text:
        return False
    bad = sum(1 for c in text if ord(c) in (0xFFFD, 0x00) or c == "�")
    if bad > 0:
        return True
    unusual = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ALLOWED_PUNCT))
    return len(text) > 0 and unusual / len(text) > 0.3


def analyze_rollout_dir(dir_path, sample_every=5, show_examples=3):
    files = sorted(
        glob.glob(os.path.join(dir_path, "*.jsonl")),
        key=lambda f: int(os.path.basename(f).replace(".jsonl", "")),
    )
    if not files:
        print(f"NO ROLLOUT FILES at {dir_path}")
        return

    print(f"=== {os.path.basename(os.path.dirname(dir_path))} "
          f"({len(files)} rollout files, sampling every {sample_every}) ===")
    print(f"{'step':>6s}  {'n':>5s}  {'word_rep':>10s}  {'char_rep':>10s}  {'gibberish':>10s}")

    for i, f in enumerate(files):
        if i % sample_every != 0 and i != len(files) - 1:
            continue
        step = int(os.path.basename(f).replace(".jsonl", ""))
        with open(f, encoding="utf-8") as fh:
            lines = fh.readlines()
        total = len(lines)
        wr = cr = gb = 0
        examples = []
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = r.get("output", "") or r.get("response", "") or ""
            if not resp:
                continue
            w = detect_word_repetition(resp)
            c = detect_char_repetition(resp)
            g = detect_gibberish(resp)
            if w >= 5:
                wr += 1
            if c >= 10:
                cr += 1
            if g:
                gb += 1
            if (w >= 10 or c >= 20 or g) and len(examples) < show_examples:
                examples.append((step, w, c, g, resp[:200]))
        print(f"{step:6d}  {total:5d}  {wr:3d}({wr/total*100:4.1f}%)  "
              f"{cr:3d}({cr/total*100:4.1f}%)  {gb:3d}({gb/total*100:4.1f}%)")
        for ex_step, ex_w, ex_c, ex_g, preview in examples:
            tags = []
            if ex_w >= 10:
                tags.append(f"word_rep={ex_w}")
            if ex_c >= 20:
                tags.append(f"char_rep={ex_c}")
            if ex_g:
                tags.append("gibberish")
            print(f"        [{'|'.join(tags)}] {preview[:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Detect rollout degradation (repetition & gibberish)")
    parser.add_argument("--rollout-dir", required=True, help="Path to rollouts/ directory")
    parser.add_argument("--sample-every", type=int, default=5, help="Sample every N steps")
    parser.add_argument("--show-examples", type=int, default=3, help="Max example previews per step")
    args = parser.parse_args()
    analyze_rollout_dir(args.rollout_dir, args.sample_every, args.show_examples)


if __name__ == "__main__":
    main()
