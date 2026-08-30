#!/usr/bin/env python3
"""合并 3 机 397B gold summary 输出 → 最终 SFT 数据（merge + 终检清洗一体化）

输入（`.runtime/flashnote_summary/`）:
  sft_gold_397b.jsonl      m2 shard 0/3
  sft_gold_397b_m3.jsonl  m3 shard 1/3
  sft_gold_397b_m4.jsonl  m4 shard 2/3

输出:
  sft_gold_397b_final.jsonl       清洗后的最终 SFT 数据（预期 72166 unique）
  sft_gold_397b_final_bad.jsonl   被丢弃条目（带 reason），供规则准确性复核

处理（merge + 终检清洗合流，规则与 gold_gen_sft_clean.py 一致）:
  1) 结构校验：messages≥2、有 index、有 images（缺失即丢）
  2) 按 extra_info.index 去重，保留首次出现（baseline 种子重复自动吸收）
  3) assistant 首尾空白剥离 + 丢弃 reasoning 字段（teacher thinking，label 只取正文）
  4) 红线污染【全文扫描】（绝不该出现在干净摘要里）：
       <image> / Role Setting / Core Identity / Mandatory Rules / 【Reference Summary】 / teacher_prompt
  5) thinking 泄漏【head-300 软扫】（397B 推理开头串场词，全文扫会误伤正文）：
       Thinking Process / Step 1: / Let me / My thought process / Analysis:
  5b) 拒答过滤（整段 unable to generate / 无法生成 等 <300 字符的拒答丢弃）
  6) 长度闸门：min_len(默认20) ~ max_len(默认4000)，过短=空答/拒答，过长=thinking 泄漏
  7) 代码块包裹（assistant 以 ``` 开头）丢弃
  8) 剥首行 markdown 标题（连续 # 标题行 + 空行，全部剥掉直到正文）
  9) markdown ** 不对称（数量为奇数=未闭合）丢弃
 10) images 字段校验：必须是原图路径(en_image/)，丢弃任何指向 _lr/ 半分辨率图的条目
 11) 全文精确去重：同 assistant 全文（strip 后）保留首条，重复进 bad-dump
     （近似"同开头不同截图"不丢，只精确全文重复才丢）
 12) 语种一致性【软报告，默认仅报告不丢】：assistant 主体文字脚本 vs 图片语种目录
       --drop-mismatch 时才丢弃不一致条目
 13) token 长度校验：用 --tokenizer 对应 tokenizer 计 user+assistant 文本 token 数，
     超 --max-tokens 阈值丢弃（不含 image placeholder token，实际训练需另算）
 14) 最终 shard 一致性校验：idx%3 ∈ {0,1,2} 与来源机器对应

  被丢弃条目统一写入 <out>_bad.jsonl（含 reason 字段），便于人工复核规则是否误判。
  merge 产出的 final.jsonl 即 SFT 就绪数据，无需再跑单独 clean 步骤（幂等）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# 默认 SFT student model tokenizer（Qwen3.5-9B）
DEFAULT_TOKENIZER_PATH = "/data4/wumeimei/download_models/Qwen3.5-9B"
# 默认 token 上限（文本部分 user+assistant，不含 image token）
# Qwen3.5-9B model_max_length=262144，但 SFT 训练 max_length 通常设 8192，
# 留 image placeholder (原图~1-2k token) + 系统 margin → 文本上限 4096
DEFAULT_MAX_TOKENS = 4096

# 各机 shard 映射（用于一致性校验）
SHARD_MAP = {
    "sft_gold_397b.jsonl": 0,
    "sft_gold_397b_m3.jsonl": 1,
    "sft_gold_397b_m4.jsonl": 2,
}
SHARD_TOTAL = 3
EXPECTED_TOTAL = 72166

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

# assistant 最小/最大字符数（过短=空答/拒答，过长=thinking 泄漏）
MIN_ASSISTANT_LEN = 20
MAX_ASSISTANT_LEN = 4000

# 首行 markdown 标题正则（连续多个 # 标题行 + 空行，全部剥掉直到正文）
MD_TITLE_RE = re.compile(r'^(?:#{1,6}\s+[^\n]+\n+)+')

# 拒答模式（prompt 规定 "no usable information → unable to generate"）
# 4 语种：en=unable to generate / fr=Impossible de générer / ru=Невозможно сгенерировать / zh=无法生成
# 仅整段拒答（strip 后以拒答词开头且 <300 字符），正常摘要含"无法"等不误伤
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


# ──────────────────────────────────────────────────────────────────────────────
# 判定函数（与 gold_gen_sft_clean.py 一致）
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


def is_codeblock_wrapped(assistant_text: str) -> bool:
    """assistant 以 ``` 开头（整段被代码块包裹，格式异常）"""
    return assistant_text.lstrip().startswith("```")


def is_refusal(assistant_text: str) -> bool:
    """整段拒答（以拒答词开头且 <300 字符），正常摘要含"无法"等不误伤。"""
    a = assistant_text.strip()
    return len(a) < REFUSAL_MAX_LEN and bool(REFUSAL_RE.match(a))


def has_unbalanced_bold(assistant_text: str) -> bool:
    """markdown ** 数量为奇数 = 未闭合"""
    return assistant_text.count("**") % 2 != 0


def strip_leading_md_title(assistant_text: str) -> str:
    """剥首部所有连续 markdown 标题（#/##/... 直到正文）"""
    return MD_TITLE_RE.sub("", assistant_text, count=1)


def is_original_image(images: list[str] | None) -> bool:
    """原图：路径含 4 语种之一且不含 _lr/（半分辨率）"""
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
# 合并 + 清洗
# ──────────────────────────────────────────────────────────────────────────────
def merge(
    run_dir: Path,
    out_path: Path,
    bad_path: Path,
    tokenizer_path: str | None = None,
    max_tokens: int = 0,
    min_len: int = MIN_ASSISTANT_LEN,
    max_len: int = MAX_ASSISTANT_LEN,
    drop_mismatch: bool = False,
) -> int:
    seen_idx: dict[int, str] = {}     # idx -> source file（跨文件 index 去重）
    seen_full: set[str] = set()       # 全文精确去重（assistant strip 后）
    kept: list[dict] = []
    bad: list[tuple[str, dict]] = []  # (原因, 记录) → 硬丢弃，写 bad-dump 供复核
    soft: list[tuple[str, dict]] = []  # (原因, 记录) → 软标记（保留但写入复核文件）
    stats = Counter()
    lang_dist = Counter()
    mismatch_ex: list[tuple] = []
    stripped_md_title = 0
    shard_mismatch = 0

    # 懒加载 tokenizer（仅启用 token 校验时）
    tok = None
    if tokenizer_path and max_tokens > 0:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            print(f"[INFO] 加载 tokenizer: {tokenizer_path} (vocab={tok.vocab_size}, max_tokens={max_tokens})",
                  file=sys.stderr)
        except Exception as e:
            print(f"[WARN] tokenizer 加载失败, 跳过 token 校验: {e}", file=sys.stderr)
            tok = None

    for fname, expected_shard in SHARD_MAP.items():
        fpath = run_dir / fname
        if not fpath.exists():
            print(f"[WARN] 缺失文件: {fpath}", file=sys.stderr)
            continue
        with fpath.open() as fi:
            for ln, line in enumerate(fi):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    bad.append(("bad_json", {"_file": fname, "_line": ln}))
                    continue

                idx = d.get("extra_info", {}).get("index")
                messages = d.get("messages")
                images = d.get("images")

                # 1) 结构校验
                if not (isinstance(messages, list) and len(messages) >= 2):
                    stats["bad_structure"] += 1
                    bad.append(("bad_structure", d))
                    continue
                if idx is None:
                    stats["no_index"] += 1
                    bad.append(("no_index", d))
                    continue

                # 2) index 去重（baseline 种子会在后续文件重复，先丢）
                if idx in seen_idx:
                    stats["dup_index"] += 1
                    bad.append(("dup_index", d))
                    continue

                # shard 一致性校验（仅对首次出现的 idx，避免 baseline 噪声）
                if idx % SHARD_TOTAL != expected_shard:
                    shard_mismatch += 1
                seen_idx[idx] = fname

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

                # 拒答（unable to generate / 无法生成 等）
                if is_refusal(assistant):
                    stats["refusal"] += 1
                    bad.append(("refusal", d))
                    continue

                # 6a) 空答 / 过短（397B 拒答或截断）
                if len(assistant.strip()) < min_len:
                    stats["too_short"] += 1
                    bad.append(("too_short", d))
                    continue

                # 6b) 超长（thinking 泄漏进 assistant）
                if len(assistant) > max_len:
                    stats["too_long"] += 1
                    bad.append(("too_long", d))
                    continue

                # 7) 代码块包裹（格式异常）
                if is_codeblock_wrapped(assistant):
                    stats["codeblock"] += 1
                    bad.append(("codeblock", d))
                    continue

                # 3) 先剥离首尾空白（397B 输出普遍以 \n\n 开头），再剥 markdown 标题
                assistant = assistant.strip()

                # 8) 剥首部所有连续 markdown 标题（#/##/... 直到正文）
                stripped = strip_leading_md_title(assistant)
                if stripped != assistant:
                    stripped_md_title += 1
                    assistant = stripped

                # 9) markdown ** 不对称（未闭合）
                if has_unbalanced_bold(assistant):
                    stats["unbalanced_bold"] += 1
                    bad.append(("unbalanced_bold", d))
                    continue

                # 剥标题后再次校验过短
                if len(assistant.strip()) < min_len:
                    stats["too_short"] += 1
                    bad.append(("too_short", d))
                    continue

                # 10) 图片分辨率校验（必须原图）
                if not is_original_image(images):
                    stats["lr_image"] += 1
                    bad.append(("lr_image", d))
                    continue

                # 12) 语种一致性（软报告）
                if not lang_match(images, assistant):
                    stats["lang_mismatch"] += 1
                    if len(mismatch_ex) < 6:
                        m = LANG_RE.search(images[0])
                        _cyr, _lat, _cjk = script_counts(assistant)
                        mismatch_ex.append((m.group(1) if m else "?",
                                            f"cyr={_cyr} cjk={_cjk}",
                                            assistant[:80]))
                    if drop_mismatch:
                        bad.append(("lang_mismatch", d))
                        continue
                    # 默认仅报告但保留：仍写入复核文件（soft: 前缀）供查规则是否漏网
                    soft.append(("soft:lang_mismatch", d))

                # 11) 全文精确去重
                if assistant in seen_full:
                    stats["dup_full"] += 1
                    bad.append(("dup_full", d))
                    continue
                seen_full.add(assistant)

                # 清洗：剥离 assistant 前后空白，丢 reasoning 字段
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

    # 13) token 长度校验（批量，仅启用时执行）
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

    # 写出干净 jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for d in kept:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 写出低质量复核文件（硬丢弃 + 软标记，带 reason，soft: 前缀区分）
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    with bad_path.open("w", encoding="utf-8") as fb:
        for reason, d in bad + soft:
            fb.write(json.dumps({"reason": reason, **d}, ensure_ascii=False) + "\n")

    # ── 报告 ──
    print(f"输入文件: {list(SHARD_MAP)}")
    print(f"去重保留: {len(kept)} / {EXPECTED_TOTAL}  (unique idx={len(seen_idx)})")
    print(f"丢弃: {len(bad)} 条（硬过滤，已剔除）+ 软标记 {len(soft)} 条（保留，写入复核文件）")
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
    # shard 不一致：baseline 种子会让 idx%3∈{1,2} 的 idx 也出现在 m2 文件里（见文档 §3），
    # 此告警仅在 idx 真正越界（出现在非自己分片该有的文件且非 baseline）时才有意义，
    # baseline 产生的噪声已通过去重吸收，故此处仅打印总数供参考
    print(f"  shard 不一致(含 baseline 噪声, 仅参考): {shard_mismatch}")
    missing = EXPECTED_TOTAL - len(seen_idx)
    if missing > 0:
        print(f"[INFO] 缺失 idx 数: {missing}（任务尚未跑完，可重跑补齐）")
    # DATA_FORMAT_SPEC 红线复核（抽样前 100 条）
    bad_red = 0
    for r in kept[:100]:
        a = r["messages"][1]["content"]
        if a.startswith("<image>") or "_lr/" in r["images"][0] \
           or "Role Setting" in a[:200] or "Core Identity" in a[:200] \
           or "【Reference Summary】" in a:
            bad_red += 1
    flag = "✅" if bad_red == 0 else "❌"
    print(f"红线复核（前100条）: {flag} {bad_red} 条不符")
    print(f"输出: {out_path}")
    print(f"低质量复核: {bad_path}")
    return len(kept)


def main():
    ap = argparse.ArgumentParser(description="合并 3 机 397B gold → 最终 SFT 数据（一体化清洗）")
    ap.add_argument(
        "--run-dir",
        default="/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary",
        help="3 机输出所在目录",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出文件，默认 <run-dir>/sft_gold_397b_final.jsonl",
    )
    ap.add_argument(
        "--bad",
        default=None,
        help="低质量复核文件，默认 <out>_bad.jsonl",
    )
    ap.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help=f"tokenizer 路径，默认 {DEFAULT_TOKENIZER_PATH}；传空字符串禁用 token 校验",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"user+assistant 文本 token 上限，超此丢弃；默认 {DEFAULT_MAX_TOKENS}，传 0 禁用",
    )
    ap.add_argument("--min-len", type=int, default=MIN_ASSISTANT_LEN, help="assistant 最小字符数")
    ap.add_argument("--max-len", type=int, default=MAX_ASSISTANT_LEN, help="assistant 最大字符数")
    ap.add_argument("--drop-mismatch", action="store_true",
                    help="语种不符也丢弃（默认仅报告）")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "sft_gold_397b_final.jsonl"
    bad = Path(args.bad) if args.bad else out.with_name(out.name.replace(".jsonl", "_bad.jsonl"))
    tok_path = args.tokenizer.strip() or None
    merge(run_dir, out, bad, tok_path, args.max_tokens,
          args.min_len, args.max_len, args.drop_mismatch)


if __name__ == "__main__":
    main()
