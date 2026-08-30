#!/usr/bin/env python3
"""gold-gen SFT 数据终检清洗器（对 sft_gold_397b_final.jsonl 幂等，可复用任意 gold-gen jsonl）

================================================================================
 定位（与 merge_397b_gold.py 的分工）
   merge_397b_gold.py：跨 3 机 shard 去重 + 基础清洗 → sft_gold_397b_final.jsonl
   本脚本：对 final（或任意 gold-gen jsonl）做 SFT 就绪终检，幂等：
     重复跑不二次破坏已干净数据；只对未过闸的条目丢弃 / 标记 / 报告。

 清洗维度（在 merge 基础上补 merge 未覆盖项）
   1) 结构校验：messages≥2、有 index、有 images（缺失即丢）
   2) index 去重：全文内同 index 保留首次（merge 跨文件去重，本脚本做文件内保险）
   3) assistant 首尾空白剥离 + 丢 reasoning 字段（teacher thinking，label 只取正文）
   4) 红线污染【全文扫描】（merge 只扫 head-300，本脚本全文）：
        - <image> 占位符（image 只允许在 user 侧）
        - prompt 模板：Role Setting / Core Identity / Mandatory Rules
        - 生成标记：【Reference Summary】
        - teacher_prompt 痕迹
      以上任一在 assistant 全文出现即判废（红线，绝不该出现在干净摘要里）
   5) thinking 泄漏【head-300 软扫】（397B 推理开头串场词，全文扫会误伤正文）：
        Thinking Process / Step 1: / Let me / My thought process / Analysis:
      head-300 命中即判废
   6) 长度闸门：min_len(默认20) ~ max_len(默认4000)，过短=空答/拒答，过长=thinking 泄漏
   7) 图片分辨率：images 必须原图（路径不含 _lr/），半分辨率直接丢
   8) 语种一致性【软报告，默认仅报告不丢】：assistant 主体文字脚本 vs 图片语种目录
      --drop-mismatch 时才丢弃不一致条目
   9) 全文精确去重：同 assistant 全文（strip 后）保留首条，重复进 bad-dump
      （近似"同开头不同截图"不丢，只精确全文重复才丢）

 产出
   - 干净 SFT jsonl（默认 <src 同目录>/sft_gold_397b_sft_clean.jsonl）
   - stdout 报告：各维度丢弃/标记计数 + 语种分布 + 样例 bad-case
   - 可选 --bad-dump：把丢弃条目写一份 jsonl 供人工复核

 幂等性
   对 merge 产出的 final.jsonl 跑本脚本：结构/dedup/红线/长度/分辨率全过，
   语种一致性 0% 不符（已实测），产出 = 输入条数，只新增一份报告。后续补跑
   数据增长后重跑，自动只放行过闸条目。

 用法
   cd /data4/wumeimei/flash_note/RP-OPSD
   python3 scripts/gold_gen_sft_clean.py                       # 默认 in/out
   python3 scripts/gold_gen_sft_clean.py --src <gold.jsonl> --out <clean.jsonl>
   python3 scripts/gold_gen_sft_clean.py --bad-dump <bad.jsonl> # 顺带导出 bad-case
   python3 scripts/gold_gen_sft_clean.py --drop-mismatch        # 语种不符也丢
================================================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 默认路径
# ──────────────────────────────────────────────────────────────────────────────
RUN_DIR = Path("/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary")
SRC_DEFAULT = RUN_DIR / "sft_gold_397b_final.jsonl"
OUT_DEFAULT = RUN_DIR / "sft_gold_397b_sft_clean.jsonl"

# 4 语种原图目录（en/fr/ru/zh），任何 _lr/ 都判为半分辨率
LANG_IMAGE_DIRS = ("en_image", "fr_image", "ru_image", "zh_image")
LANG_RE = re.compile(r"/(en|fr|ru|zh)_image")

# 红线污染关键词（全文扫描）——绝不该出现在干净摘要里
POLLUTION_HARD = (
    "<image>",
    "Role Setting",
    "Core Identity",
    "Mandatory Rules",
    "【Reference Summary】",
    "teacher_prompt",
)

# thinking 泄漏关键词（仅 head-300，全文扫会误伤正文）
POLLUTION_THINK_HEAD = (
    "Thinking Process",
    "Thinking process",
    "Step 1:",
    "Step 1 ",
    "Let me think",
    "Let's think",
    "My thought process",
    "Analysis:",
)

MIN_ASSISTANT_LEN = 20
MAX_ASSISTANT_LEN = 4000


# ──────────────────────────────────────────────────────────────────────────────
# 判定函数
# ──────────────────────────────────────────────────────────────────────────────
def is_polluted_hard(text: str) -> str | None:
    """全文红线扫描，命中返回关键词（判废），否则 None。"""
    for kw in POLLUTION_HARD:
        if kw in text:
            return kw
    return None


def is_polluted_think(text: str) -> str | None:
    """head-300 thinking 泄漏扫描，命中返回关键词（判废），否则 None。"""
    head = text[:300]
    for kw in POLLUTION_THINK_HEAD:
        if kw in head:
            return kw
    return None


def is_original_image(images: list[str] | None) -> bool:
    """原图：路径含 4 语种之一且不含 _lr/。"""
    if not images:
        return False
    for p in images:
        if "_lr/" in p:
            return False
        if not any(f"/{d}/" in p for d in LANG_IMAGE_DIRS):
            return False
    return True


def script_counts(text: str) -> tuple[int, int, int]:
    """统计前 600 字符各脚本计数：(cyrillic, latin, cjk)。

    用存在性阈值（见 lang_match）而非众数，避免"中文摘要混 Latin 应用名/人名
    → Latin 略多 → 误判 latin"的假阳性（实测 zh 摘要 cjk 68~191 仍被众数判 latin）。
    """
    cyr = lat = cjk = 0
    for ch in text[:600]:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            cyr += 1
        elif 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        elif unicodedata.category(ch)[0] == "L" and o < 0x2500:
            lat += 1
    return cyr, lat, cjk


# 语种存在性阈值：够多即认定该脚本"出现"（en/fr 同族拉丁无法细分）
SCRIPT_THRESH = 10


def lang_match(images: list[str], assistant: str) -> bool:
    """图片语种目录 vs assistant 主体脚本一致性。

    zh/ru 图：摘要须含足量对应脚本（≥10）——防 397B 对 zh/ru 图输出拉丁文；
    en/fr 图：摘要须 Latin 主导（lat ≥ max(cyr,cjk)）——防输出俄/中文，
        但放行英文摘要合法引用图内外文 UI（如 "Скачанные" (Downloaded)），
        实测 en+cyr 引用 lat=412≫cyr=84 合规，整段俄文 cyr≫lat 才判不符。
    """
    m = LANG_RE.search(images[0]) if images else None
    if not m:
        return True  # 无法判定语种，不报不符
    lang = m.group(1)
    if len(assistant.strip()) < 30:
        return True  # 过短不判
    cyr, lat, cjk = script_counts(assistant)
    if lang == "zh":
        return cjk >= SCRIPT_THRESH
    if lang == "ru":
        return cyr >= SCRIPT_THRESH
    return lat >= max(cyr, cjk)  # en/fr: Latin 须主导


# ──────────────────────────────────────────────────────────────────────────────
# 主清洗
# ──────────────────────────────────────────────────────────────────────────────
def clean(src: Path, out: Path, args) -> int:
    seen_idx: set[int] = set()
    seen_full: set[str] = set()           # 全文精确去重（strip 后）
    kept: list[dict] = []
    bad: list[tuple[str, dict]] = []       # (原因, 原始记录)
    stats = Counter()
    lang_dist = Counter()
    mismatch_ex: list[tuple] = []

    if not src.exists():
        print(f"[ERR] 输入文件不存在: {src}", file=sys.stderr)
        return 1

    with src.open(encoding="utf-8") as fi:
        for ln, line in enumerate(fi, 1):
            line = line.strip()
            if not line:
                stats["blank_line"] += 1
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                bad.append(("bad_json", {"_line": ln}))
                continue

            # 1) 结构校验
            messages = d.get("messages")
            idx = (d.get("extra_info") or {}).get("index")
            images = d.get("images")
            if not (isinstance(messages, list) and len(messages) >= 2):
                stats["bad_structure"] += 1
                bad.append(("bad_structure", d))
                continue
            if idx is None:
                stats["no_index"] += 1
                bad.append(("no_index", d))
                continue

            # 2) index 去重（文件内保险）
            if idx in seen_idx:
                stats["dup_index"] += 1
                bad.append(("dup_index", d))
                continue
            seen_idx.add(idx)

            assistant = messages[1].get("content", "")
            if not isinstance(assistant, str):
                assistant = str(assistant)

            # 4) 红线全文污染
            kw = is_polluted_hard(assistant)
            if kw:
                stats["pollution_hard"] += 1
                bad.append((f"pollution_hard:{kw}", d))
                continue
            # 5) thinking 泄漏 head-300
            kw = is_polluted_think(assistant)
            if kw:
                stats["pollution_think"] += 1
                bad.append((f"pollution_think:{kw}", d))
                continue

            # 剥离空白
            assistant = assistant.strip()

            # 6) 长度闸门
            if len(assistant) < args.min_len:
                stats["too_short"] += 1
                bad.append(("too_short", d))
                continue
            if len(assistant) > args.max_len:
                stats["too_long"] += 1
                bad.append(("too_long", d))
                continue

            # 7) 图片分辨率
            if not is_original_image(images):
                stats["lr_image"] += 1
                bad.append(("lr_image", d))
                continue

            # 8) 语种一致性（软报告）
            if not lang_match(images, assistant):
                stats["lang_mismatch"] += 1
                if len(mismatch_ex) < 6:
                    m = LANG_RE.search(images[0])
                    _cyr, _lat, _cjk = script_counts(assistant)
                    mismatch_ex.append((m.group(1) if m else "?",
                                        f"cyr={_cyr} cjk={_cjk}",
                                        assistant[:80]))
                if args.drop_mismatch:
                    bad.append(("lang_mismatch", d))
                    continue

            # 9) 全文精确去重
            if assistant in seen_full:
                stats["dup_full"] += 1
                bad.append(("dup_full", d))
                continue
            seen_full.add(assistant)

            # 3) 清洗：剥离空白，丢 reasoning 字段
            cleaned = {
                "messages": [
                    {"role": "user", "content": messages[0].get("content", "")},
                    {"role": "assistant", "content": assistant},
                ],
                "images": images,
                "extra_info": {"index": idx},
            }
            kept.append(cleaned)
            m = LANG_RE.search(images[0])
            lang_dist[m.group(1) if m else "?"] += 1

    # 写出干净 jsonl
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fo:
        for d in kept:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")

    # bad-dump
    if args.bad_dump:
        bd = Path(args.bad_dump)
        bd.parent.mkdir(parents=True, exist_ok=True)
        with bd.open("w", encoding="utf-8") as fb:
            for reason, d in bad:
                fb.write(json.dumps({"reason": reason, **d}, ensure_ascii=False) + "\n")

    # ── 报告 ──
    total_in = sum(stats.values()) + len(kept)
    print(f"[clean] src = {src}")
    print(f"[clean] out = {out}")
    print(f"[clean] 输入 {total_in} 行, 保留 {len(kept)}, 丢弃 {len(bad)}")
    if stats:
        print("[clean] 丢弃明细:")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    print(f"[clean] 保留条目语种分布: {dict(sorted(lang_dist.items()))}")
    if mismatch_ex:
        print(f"[clean] 语种不符样例（共 {stats['lang_mismatch']} 条, "
              f"{'已丢' if args.drop_mismatch else '仅报告未丢'}）:")
        for lang, sc, head in mismatch_ex:
            print(f"    img_lang={lang} asst_script={sc} head={head!r}")
    if kept:
        a = kept[0]["messages"][1]["content"]
        print(f"[clean] 首条 assistant[:120] = {a[:120]!r}")
    # DATA_FORMAT_SPEC 红线复核（抽样前 100 条）
    bad_red = 0
    for i, r in enumerate(kept[:100]):
        a = r["messages"][1]["content"]
        if a.startswith("<image>") or "_lr/" in r["images"][0] \
           or "Role Setting" in a[:200] or "Core Identity" in a[:200] \
           or "【Reference Summary】" in a:
            bad_red += 1
    flag = "✅" if bad_red == 0 else "❌"
    print(f"[clean] 红线复核（前100条）: {flag} {bad_red} 条不符")
    print(f"[done] {len(kept)} 行 -> {out}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="gold-gen SFT 终检清洗器")
    p.add_argument("--src", default=str(SRC_DEFAULT), help="输入 gold-gen jsonl")
    p.add_argument("--out", default=str(OUT_DEFAULT), help="输出干净 SFT jsonl")
    p.add_argument("--min-len", type=int, default=MIN_ASSISTANT_LEN, help="assistant 最小字符数")
    p.add_argument("--max-len", type=int, default=MAX_ASSISTANT_LEN, help="assistant 最大字符数")
    p.add_argument("--drop-mismatch", action="store_true",
                   help="语种不符也丢弃（默认仅报告）")
    p.add_argument("--bad-dump", default=None, help="把丢弃条目写到该文件供复核")
    args = p.parse_args()
    raise SystemExit(clean(Path(args.src), Path(args.out), args))


if __name__ == "__main__":
    main()
