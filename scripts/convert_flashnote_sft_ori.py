#!/usr/bin/env python3
"""flash_note summary jsonl -> SFT ori 数据（4 语种 prompt + 训练集自带 summary + 清洗）

================================================================================
 目的：与 gen_gold_397b 隔离 gold 来源变量
   - prompt：用 convert_flashnote_summary.py 的 PROMPT_TEMPLATES（4 语种全文翻译版，
     与 gen_gold_397b / RP-OPSD 完全一致）
   - gold：训练集自带 summary（从源 jsonl 的 【Reference Summary】 标记后提取纯摘要正文，
     不含 ref 标记、不含 prompt 模板）
   - images：默认原图（与 gen_gold_397b 同视图）；--lr 切半分辨率

   与 gen_gold_397b 的唯一区别：gold 来源
     gen_gold_397b：397B 教师看原图生成
     ori          ：训练集自带 summary（零推理成本）

 数据源
   jsonl: /data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl
          - messages[0].content   user 指令（含 <image> 占位符）
          - images[0]             原图路径
          - teacher_prompt        含 【Reference Summary】<gold>（gold 提取源）

 清洗（与 merge_397b_gold.py 一致）
   1) 结构校验：有 messages、有 images、能提取 gold
   2) assistant 首尾空白剥离
   3) 红线污染全文扫描：<image> / Role Setting / Core Identity / Mandatory Rules
                       / 【Reference Summary】 / teacher_prompt
   4) thinking 泄漏 head-300：Thinking Process / Step 1: / Let me / Analysis: 等
   5) 长度闸门：min_len(20) ~ max_len(4000)
   6) 代码块包裹（assistant 以 ``` 开头）丢弃
   7) 剥首行 markdown 标题（连续 # 标题行 + 空行）
   8) markdown ** 不对称（奇数=未闭合）丢弃
   9) images 字段校验：必须原图路径（en_image/fr_image/ru_image/zh_image），不含 _lr/
   10) 全文精确去重：同 assistant 全文保留首条
   11) 语种一致性：assistant 脚本 vs 图片语种目录（默认仅报告不丢，--drop-mismatch 才丢）
   12) token 长度校验：--tokenizer 对应 tokenizer 计 user+assistant 文本 token 数，超 --max-tokens 丢弃

 输出
   sft_train_ori.jsonl       清洗后的 SFT 数据
   sft_train_ori_bad.jsonl   被丢弃条目（带 reason），供复核

 用法
   python convert_flashnote_sft_ori.py --out .runtime/flashnote_summary/sft_train_ori.jsonl
   python convert_flashnote_sft_ori.py --out ... --lr            # 半分辨率训练图
   python convert_flashnote_sft_ori.py --out ... --show-sample   # 打印样例
================================================================================
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from convert_flashnote_summary import (  # noqa: E402
    PROMPT_TEMPLATES,
    SRC_DEFAULT,
    detect_lang,
    lr_path,
)

DEFAULT_TOKENIZER_PATH = "/data4/wumeimei/download_models/Qwen3.5-9B"
DEFAULT_MAX_TOKENS = 4096

LANG_IMAGE_DIRS = ("en_image", "fr_image", "ru_image", "zh_image")
LANG_RE = re.compile(r"/(en|fr|ru|zh)_image")

REF_MARKER = "【Reference Summary】"

POLLUTION_HARD = (
    "<image>",
    "Role Setting",
    "Core Identity",
    "Mandatory Rules",
    "【Reference Summary】",
    "teacher_prompt",
)

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

# 拒答模式（prompt 模板规定 "If the image truly contains no usable information, output: unable to generate"）
# 4 语种各有对应：en=unable to generate / fr=Impossible de générer / ru=Невозможно сгенерировать / zh=无法生成
# 仅匹配整段拒答（strip 后以拒答词开头且 <300 字符），正常摘要含"无法"等不误伤
REFUSAL_RE = re.compile(
    r'^(?:'
    r'unable to generate'
    r'|I am unable to'
    r"|I can(?:not|'t) (?:generate|provide|create)"
    r'|Impossible de générer'
    r'|Je ne peux pas générer'
    r'|incapable de générer'
    r'|Невозможно сгенерировать'
    r'|не могу сгенерировать'
    r'|无法生成'
    r'|无法提供摘要'
    r'|无法提取摘要'
    r'|无法分析'
    r'|对不起.*无法生成'
    r'|抱歉.*无法生成'
    r'|抱歉.*无法提供'
    r')',
    re.IGNORECASE,
)
REFUSAL_MAX_LEN = 300

MD_TITLE_RE = re.compile(r'^(?:#{1,6}\s+[^\n]+\n+)+')

SCRIPT_THRESH = 10


# ──────────────────────────────────────────────────────────────────────────────
# gold 提取（从源 jsonl 的 【Reference Summary】 标记后取纯摘要正文）
# ──────────────────────────────────────────────────────────────────────────────
def extract_gold(d: dict) -> str | None:
    """从源记录提取纯 summary 正文（去 ref 标记 + 去 prompt 模板）。

    源记录的 teacher_prompt 字段形如 "<prompt 全文>\\n\\n【Reference Summary】<摘要正文>"。
    SFT 只需摘要正文，取 【Reference Summary】 之后的纯文本。
    找不到标记则回退到 messages assistant / 顶层 summary 等常见字段。
    """
    tp = d.get("teacher_prompt")
    if isinstance(tp, str) and tp.strip():
        idx = tp.find(REF_MARKER)
        if idx != -1:
            gold = tp[idx + len(REF_MARKER):].strip()
            return gold or None
        return tp.strip() or None
    if isinstance(tp, dict):
        for k in ("summary", "content", "response", "text", "answer"):
            v = tp.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    for m in d.get("messages", []) or []:
        if m.get("role") == "assistant":
            c = str(m.get("content", "")).strip()
            if c:
                return c
    for k in ("summary", "response", "answer", "gold", "target", "output"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 清洗判定函数（与 merge_397b_gold.py 一致）
# ──────────────────────────────────────────────────────────────────────────────
def is_polluted_hard(text: str) -> str | None:
    for kw in POLLUTION_HARD:
        if kw in text:
            return kw
    return None


def is_polluted_think(text: str) -> str | None:
    head = text[:300]
    for kw in POLLUTION_THINK_HEAD:
        if kw in head:
            return kw
    return None


def is_codeblock_wrapped(assistant_text: str) -> bool:
    return assistant_text.lstrip().startswith("```")


def is_refusal(assistant_text: str) -> bool:
    """整段拒答（以拒答词开头且 <300 字符），正常摘要含"无法"等不误伤。"""
    a = assistant_text.strip()
    return len(a) < REFUSAL_MAX_LEN and bool(REFUSAL_RE.match(a))


def has_unbalanced_bold(assistant_text: str) -> bool:
    return assistant_text.count("**") % 2 != 0


def strip_leading_md_title(assistant_text: str) -> str:
    return MD_TITLE_RE.sub("", assistant_text, count=1)


def is_original_image(images: list[str] | None) -> bool:
    if not images:
        return False
    for p in images:
        if "_lr/" in p:
            return False
        if not any(f"/{d}/" in p for d in LANG_IMAGE_DIRS):
            return False
    return True


def script_counts(text: str) -> tuple[int, int, int]:
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


def lang_match(images: list[str], assistant: str) -> bool:
    m = LANG_RE.search(images[0]) if images else None
    if not m:
        return True
    lang = m.group(1)
    if len(assistant.strip()) < 30:
        return True
    cyr, lat, cjk = script_counts(assistant)
    if lang == "zh":
        return cjk >= SCRIPT_THRESH
    if lang == "ru":
        return cyr >= SCRIPT_THRESH
    return lat >= max(cyr, cjk)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────
def convert(
    src: Path,
    out_path: Path,
    bad_path: Path,
    use_lr: bool = False,
    show_sample: bool = False,
    max_per_lang: int = 22000,
    seed: int = 42,
    tokenizer_path: str | None = None,
    max_tokens: int = 0,
    min_len: int = MIN_ASSISTANT_LEN,
    max_len: int = MAX_ASSISTANT_LEN,
    drop_mismatch: bool = False,
) -> int:
    # 1) 读源 jsonl，按语种分组 + 提取 gold
    by_lang: dict[str, list[tuple[dict, str, str]]] = {}  # (d, gold, orig_img)
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
            if lang is None or lang not in PROMPT_TEMPLATES:
                continue
            gold = extract_gold(d)
            if not gold:
                bad_no_gold += 1
                continue
            by_lang.setdefault(lang, []).append((d, gold, imgs[0]))

    print("[read] 采样前语种分布:")
    for k in sorted(by_lang):
        print(f"  {k}: {len(by_lang[k])}")
    if bad_no_gold:
        print(f"  [warn] 无 gold summary 跳过: {bad_no_gold}")

    # 2) 与 RP-OPSD 同采样：seed42、每门封顶 2.2W、rng.sample 同位置选择
    for lang in list(by_lang):
        if len(by_lang[lang]) > max_per_lang:
            rng = random.Random(seed)
            by_lang[lang] = rng.sample(by_lang[lang], max_per_lang)
            print(f"[sample] {lang} -> {max_per_lang}")

    # 3) 懒加载 tokenizer
    tok = None
    if tokenizer_path and max_tokens > 0:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            print(f"[INFO] tokenizer: {tokenizer_path} (max_tokens={max_tokens})", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] tokenizer 加载失败: {e}", file=sys.stderr)
            tok = None

    # 4) 遍历 + 清洗
    seen_full: set[str] = set()
    kept: list[dict] = []
    bad: list[tuple[str, dict]] = []
    soft: list[tuple[str, dict]] = []
    stats = Counter()
    lang_dist = Counter()
    mismatch_ex: list[tuple] = []
    stripped_md_title = 0

    for lang in sorted(by_lang):
        template = PROMPT_TEMPLATES[lang]
        for d, gold, orig in by_lang[lang]:
            images = [orig] if not use_lr else [lr_path(orig)]
            assistant = gold

            # 3) 红线全文污染
            kw = is_polluted_hard(assistant)
            if kw:
                stats["pollution_hard"] += 1
                bad.append((f"pollution_hard:{kw}", {"_lang": lang, "assistant": assistant}))
                continue
            # 4) thinking 泄漏 head-300
            kw = is_polluted_think(assistant)
            if kw:
                stats["pollution_think"] += 1
                bad.append((f"pollution_think:{kw}", {"_lang": lang, "assistant": assistant}))
                continue

            # 拒答（unable to generate / 无法生成 等）
            if is_refusal(assistant):
                stats["refusal"] += 1
                bad.append(("refusal", {"_lang": lang, "assistant": assistant}))
                continue

            # 5a) 过短
            if len(assistant.strip()) < min_len:
                stats["too_short"] += 1
                bad.append(("too_short", {"_lang": lang, "assistant": assistant}))
                continue
            # 5b) 过长
            if len(assistant) > max_len:
                stats["too_long"] += 1
                bad.append(("too_long", {"_lang": lang, "assistant": assistant}))
                continue

            # 6) 代码块包裹
            if is_codeblock_wrapped(assistant):
                stats["codeblock"] += 1
                bad.append(("codeblock", {"_lang": lang, "assistant": assistant}))
                continue

            # 2) 首尾空白剥离
            assistant = assistant.strip()

            # 7) 剥首部 markdown 标题
            stripped = strip_leading_md_title(assistant)
            if stripped != assistant:
                stripped_md_title += 1
                assistant = stripped

            # 8) ** 不对称
            if has_unbalanced_bold(assistant):
                stats["unbalanced_bold"] += 1
                bad.append(("unbalanced_bold", {"_lang": lang, "assistant": assistant}))
                continue

            if len(assistant.strip()) < min_len:
                stats["too_short"] += 1
                bad.append(("too_short", {"_lang": lang, "assistant": assistant}))
                continue

            # 9) 图片分辨率校验
            if not use_lr and not is_original_image(images):
                stats["lr_image"] += 1
                bad.append(("lr_image", {"_lang": lang, "images": images}))
                continue

            # 11) 语种一致性（软报告）
            if not lang_match(images, assistant):
                stats["lang_mismatch"] += 1
                if len(mismatch_ex) < 6:
                    m = LANG_RE.search(images[0])
                    _cyr, _lat, _cjk = script_counts(assistant)
                    mismatch_ex.append((m.group(1) if m else "?",
                                        f"cyr={_cyr} cjk={_cjk}",
                                        assistant[:80]))
                if drop_mismatch:
                    bad.append(("lang_mismatch", {"_lang": lang, "assistant": assistant}))
                    continue
                soft.append(("soft:lang_mismatch", {"_lang": lang, "assistant": assistant}))

            # 10) 全文精确去重
            if assistant in seen_full:
                stats["dup_full"] += 1
                bad.append(("dup_full", {"_lang": lang, "assistant": assistant}))
                continue
            seen_full.add(assistant)

            cleaned = {
                "messages": [
                    {"role": "user", "content": template},
                    {"role": "assistant", "content": assistant},
                ],
                "images": images,
            }
            kept.append(cleaned)
            m = LANG_RE.search(images[0])
            lang_dist[m.group(1) if m else "?"] += 1

    # 12) token 长度校验（批量）
    if tok is not None and max_tokens > 0 and kept:
        texts = [f"{r['messages'][0]['content']}\n{r['messages'][1]['content']}" for r in kept]
        enc = tok(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        survivors = []
        for r, ids in zip(kept, enc):
            if len(ids) > max_tokens:
                stats["over_token"] += 1
                bad.append(("over_token", r))
            else:
                survivors.append(r)
        kept = survivors

    # 写出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for d in kept:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")

    bad_path.parent.mkdir(parents=True, exist_ok=True)
    with bad_path.open("w", encoding="utf-8") as fb:
        for reason, d in bad + soft:
            fb.write(json.dumps({"reason": reason, **d}, ensure_ascii=False) + "\n")

    # 报告
    print(f"输入: {src}")
    print(f"保留: {len(kept)}  (去重前 {len(seen_full) + len([1 for _ in bad if _[0] == 'dup_full'])} unique)")
    print(f"丢弃: {len(bad)} 条 + 软标记 {len(soft)} 条")
    print(f"  复核文件: {bad_path}")
    if stats:
        print("  丢弃/标记明细:")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    print(f"  剥首行 markdown 标题条数: {stripped_md_title}")
    print(f"  保留条目语种分布: {dict(sorted(lang_dist.items()))}")
    if mismatch_ex:
        print(f"  语种不符样例（共 {stats['lang_mismatch']} 条, "
              f"{'已丢' if drop_mismatch else '仅报告未丢'}）:")
        for lang, sc, head in mismatch_ex:
            print(f"    img_lang={lang} asst_script={sc} head={head!r}")
    if tok is not None and max_tokens > 0:
        print(f"  token 长度校验: 阈值={max_tokens} (Qwen3.5-9B 文本部分, 不含 image placeholder)")

    if show_sample and kept:
        print("\n=== 样例（首条）===")
        s = kept[0]
        print(f"image={s['images'][0]}")
        print(f"user[:300]={s['messages'][0]['content'][:300]}")
        print(f"assistant[:300]={s['messages'][1]['content'][:300]}")

    print(f"输出: {out_path}")
    print(f"低质量复核: {bad_path}")
    return len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description="flash_note summary -> SFT ori jsonl（4语种prompt+自带summary+清洗）")
    ap.add_argument("--src", default=SRC_DEFAULT, help="summary aligned jsonl 路径")
    ap.add_argument("--out", required=True, help="输出 SFT jsonl")
    ap.add_argument("--bad", default=None, help="低质量复核文件，默认 <out>_bad.jsonl")
    ap.add_argument("--max-per-lang", type=int, default=22000, help="每门封顶（默认 22000）")
    ap.add_argument("--seed", type=int, default=42, help="采样种子")
    ap.add_argument("--lr", action="store_true", help="训练图用半分辨率（默认原图，与 gen_gold_397b 同视图）")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH,
                    help=f"tokenizer 路径，默认 {DEFAULT_TOKENIZER_PATH}；传空字符串禁用")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"user+assistant 文本 token 上限；默认 {DEFAULT_MAX_TOKENS}；0 禁用")
    ap.add_argument("--min-len", type=int, default=MIN_ASSISTANT_LEN)
    ap.add_argument("--max-len", type=int, default=MAX_ASSISTANT_LEN)
    ap.add_argument("--drop-mismatch", action="store_true", help="语种不符也丢弃（默认仅报告）")
    ap.add_argument("--show-sample", action="store_true", help="打印一条样例")
    args = ap.parse_args()

    out = Path(args.out)
    bad = Path(args.bad) if args.bad else out.with_name(out.name.replace(".jsonl", "_bad.jsonl"))
    tok_path = args.tokenizer.strip() or None
    convert(
        Path(args.src),
        out,
        bad,
        use_lr=args.lr,
        show_sample=args.show_sample,
        max_per_lang=args.max_per_lang,
        seed=args.seed,
        tokenizer_path=tok_path,
        max_tokens=args.max_tokens,
        min_len=args.min_len,
        max_len=args.max_len,
        drop_mismatch=args.drop_mismatch,
    )


if __name__ == "__main__":
    main()
