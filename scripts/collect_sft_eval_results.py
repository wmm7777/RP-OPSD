#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 flash_note summary MOS 评测结果, 支持 ori / sft / rp_opsd 三种实验对比。

读 eval_results/eval_report/<label>/<lang>/{title,summary}_mos_results.json,
label 形如 <exp>_summary_9b_<tag> (sft_summary_9b_epoch1.0 / ori_summary_9b_base / ...),
算各 tag × 语种的 title_mos / summary_mos 平均分, 输出 CSV + 打印对比表。

用法:
  # 单实验汇总
  python collect_sft_eval_results.py --exp sft --tags "epoch1.0 epoch1.5 epoch2.0"
  # 三实验对比 (同一 CSV, 按 exp 分组)
  python collect_sft_eval_results.py --multi ori,sft,rp_opsd
"""
import argparse
import csv
import json
import os
from statistics import mean

NAN = float("nan")


def avg_mos(json_path):
    if not os.path.isfile(json_path):
        return NAN
    try:
        data = json.load(open(json_path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return NAN
    scores = [r.get("平均分", 0) for r in data if r.get("valid_content")]
    return mean(scores) if scores else NAN


def collect_exp(eval_root, exp, tags, langs):
    """返回 [(exp, tag, lang, title_mos, summary_mos), ...]"""
    report_root = os.path.join(eval_root, "eval_report")
    rows = []
    for tag in tags:
        label = f"{exp}_summary_9b_{tag}"
        for lang in langs:
            d = os.path.join(report_root, label, lang)
            t = avg_mos(os.path.join(d, "title_mos_results.json"))
            s = avg_mos(os.path.join(d, "summary_mos_results.json"))
            rows.append((exp, tag, lang,
                         round(t, 3) if t == t else "",
                         round(s, 3) if s == s else ""))
    return rows


def print_table(rows, exp, tags, langs, metric_idx, metric_name):
    print(f"\n=== {exp}  {metric_name} (tag × lang) ===")
    hdr = f"{'tag':<12}" + "".join(f"{l:>10}" for l in langs) + f"{'avg':>10}"
    print(hdr)
    for tag in tags:
        cells = []
        vals = []
        for l in langs:
            r = next((x for x in rows if x[1] == tag and x[2] == l), None)
            v = r[metric_idx] if r else ""
            cells.append(f"{v:>10}" if v != "" else f"{'-':>10}")
            if v != "":
                vals.append(v)
        avg = round(mean(vals), 3) if vals else "-"
        print(f"{tag:<12}" + "".join(cells) + f"{str(avg):>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root",
                    default="/data4/wumeimei/flash_note/auto_eval/evaluators/eval_results")
    ap.add_argument("--exp", default="sft", help="单实验: ori/sft/rp_opsd")
    ap.add_argument("--tags", default=None,
                    help="空格分隔 tag 列表 (sft 默认 epoch1.0~3.5, ori 默认 base)")
    ap.add_argument("--langs", default="en fr ru zh")
    ap.add_argument("--multi", default=None,
                    help="三实验对比, 逗号分隔 exp (如 ori,sft,rp_opsd), 覆盖 --exp/--tags")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    langs = args.langs.split()

    if args.multi:
        exps = [e.strip() for e in args.multi.split(",") if e.strip()]
        default_tags = {
            "sft": "epoch1.0 epoch1.5 epoch2.0 epoch2.5 epoch3.0 epoch3.5".split(),
            "ori": ["base"],
            "rp_opsd": ["step55"],
        }
        all_rows = []
        for e in exps:
            # 自动探测该 exp 下实际存在的 tag (扫描 eval_report 目录)
            report_root = os.path.join(args.eval_root, "eval_report")
            prefix = f"{e}_summary_9b_"
            tags = []
            if os.path.isdir(report_root):
                tags = sorted({d[len(prefix):] for d in os.listdir(report_root)
                               if d.startswith(prefix)})
            if not tags:
                tags = default_tags.get(e, ["base"])
            all_rows.extend(collect_exp(args.eval_root, e, tags, langs))
            print_table(all_rows, e, tags, langs, 3, "Title MOS")
            print_table(all_rows, e, tags, langs, 4, "Summary MOS")
        out_csv = args.out or os.path.join(args.eval_root, "eval_summary_multi.csv")
    else:
        exp = args.exp
        if args.tags:
            tags = args.tags.split()
        elif exp == "ori":
            tags = ["base"]
        elif exp == "rp_opsd":
            tags = ["step55"]
        else:
            tags = "epoch1.0 epoch1.5 epoch2.0 epoch2.5 epoch3.0 epoch3.5".split()
        all_rows = collect_exp(args.eval_root, exp, tags, langs)
        print_table(all_rows, exp, tags, langs, 3, "Title MOS")
        print_table(all_rows, exp, tags, langs, 4, "Summary MOS")
        out_csv = args.out or os.path.join(args.eval_root, f"eval_summary_{exp}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exp", "tag", "lang", "title_mos", "summary_mos"])
        w.writerows(all_rows)
    print(f"\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
