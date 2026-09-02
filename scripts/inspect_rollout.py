#!/usr/bin/env python3
"""检查 verl rollout jsonl：每行一条生成（input/output/gts/score/step）。
看退化模式：语言混杂/复读/重复段/冗长。

用法: python scripts/inspect_rollout.py <rollout.jsonl> [看前N条]
"""
import json
import sys
from collections import Counter


def detect_repeat(text):
    """检测末尾重复片段（复读模式）。"""
    if len(text) < 50:
        return ""
    # 找最长的、在末尾重复≥3次的子串
    tail = text[-300:]
    for unit_len in range(2, 40):
        unit = tail[-unit_len:]
        cnt = 0
        i = len(tail)
        while i - unit_len >= 0 and tail[i - unit_len:i] == unit:
            cnt += 1
            i -= unit_len
        if cnt >= 3:
            return f"末尾重复'{unit}'×{cnt}"
    return ""


def main():
    p = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rows = []
    with open(p, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("空文件"); return
    r = rows[0]
    print(f"=== {p} ===")
    print(f"样本数: {len(rows)}  keys: {list(r.keys())}")
    rkey = "output" if "output" in r else None
    if not rkey:
        for k in ("responses", "response", "outputs", "generations"):
            if k in r:
                rkey = k; break
    if not rkey:
        print(json.dumps(r, ensure_ascii=False)[:400]); return

    # 长度分布
    lens = [len(row[rkey]) if isinstance(row[rkey], str) else sum(len(x) for x in row[rkey]) for row in rows]
    lens.sort()
    m = len(lens)
    print(f"生成字符长: n={m} min={lens[0]} p50={lens[m//2]} p90={lens[int(m*0.9)]} max={lens[-1]} mean={sum(lens)//m}")
    print(f"空生成: {sum(1 for l in lens if l==0)}  超长(>5000): {sum(1 for l in lens if l>5000)}")

    # 语言检测（粗）：中英混杂
    mixed = 0
    for row in rows:
        t = row[rkey] if isinstance(row[rkey], str) else " ".join(row[rkey])
        has_cjk = any('一' <= c <= '鿿' for c in t)
        # 简单判：同时有中文且有大段连续英文
        import re
        en_segs = re.findall(r'[A-Za-z]{4,}', t)
        if has_cjk and len(en_segs) > 3:
            mixed += 1
    print(f"中英混杂样本: {mixed}/{m}")

    # 复读检测
    rep = 0
    for row in rows:
        t = row[rkey] if isinstance(row[rkey], str) else ""
        if detect_repeat(t):
            rep += 1
    print(f"末尾复读样本: {rep}/{m}")

    # 打印前 n 条（按长度从长到短，看退化最严重的）
    rows_sorted = sorted(rows, key=lambda x: -(len(x[rkey]) if isinstance(x[rkey], str) else 0))
    print(f"\n--- 最长 {n} 条生成 ---")
    for row in rows_sorted[:n]:
        t = row[rkey] if isinstance(row[rkey], str) else " ".join(row[rkey])
        rep_mark = detect_repeat(t)
        print(f"\n[字长{len(t)} score={row.get('score')} {rep_mark}]")
        print(t[:500])
        if len(t) > 700:
            print(f"  ...[省略{len(t)-700}字]...")
            print(f"  尾300: ...{t[-300:]}")


if __name__ == "__main__":
    main()
