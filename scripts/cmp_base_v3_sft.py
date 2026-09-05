#!/usr/bin/env python3
"""计算 base 9B / v3 step150/300 / SFT 等评测的总均分对比"""
import json, statistics, sys, os

RESULTS_ROOT = "/data4/wumeimei/flash_note/eval_results/eval_res_0904"
LANGS = ["en", "fr", "ru", "zh"]


def load_runs(subdir):
    out = {}
    for L in LANGS:
        f = os.path.join(RESULTS_ROOT, subdir, L, "summary_mos_results.json")
        if not os.path.exists(f):
            out[L] = None
            continue
        try:
            d = json.load(open(f))
        except Exception:
            out[L] = None
            continue
        v = [x for x in d if x.get("valid_content") and x.get("准确性")]
        out[L] = v
    return out


def stats(v):
    if not v:
        return None
    acc = statistics.mean([x["准确性"]["分数"] for x in v])
    con = statistics.mean([x["简洁性"]["分数"] for x in v])
    com = statistics.mean([x["完整性"]["分数"] for x in v])
    fmt = statistics.mean([x["格式"]["分数"] for x in v])
    lf = statistics.mean([x["语种遵循度"]["分数"] for x in v])
    avg = (acc + con + com + fmt) / 4
    bad = sum(1 for x in v if x["准确性"]["分数"] <= 2)
    return {
        "N": len(v), "acc": acc, "con": con, "com": com, "fmt": fmt,
        "lf": lf, "avg": avg, "bad": bad, "bad_pct": bad / len(v) * 100,
    }


def main():
    runs = {
        "base_9b": "qwen35_9b_base_summary",
        "v3_step150": "rp_opsd_v3noema_summary_9b_step150",
        "v3_step300": "rp_opsd_v3noema_summary_9b_step300",
    }
    print(f"{'exp':14} {'lang':4} {'N':4} {'acc':>6} {'con':>6} {'com':>6} {'fmt':>6} {'lf':>5} {'avg':>6} {'bad%':>6}")
    for name, sub in runs.items():
        per = load_runs(sub)
        for L in LANGS:
            r = stats(per[L])
            if r:
                print(f"{name:14} {L:4} {r['N']:4} {r['acc']:.3f} {r['con']:.3f} {r['com']:.3f} {r['fmt']:.3f} {r['lf']:.3f} {r['avg']:.3f} {r['bad_pct']:5.1f}%")
        # total
        all_v = []
        for L in LANGS:
            if per[L]:
                all_v.extend(per[L])
        if all_v:
            r = stats(all_v)
            print(f"{name:14} {'ALL':4} {r['N']:4} {r['acc']:.3f} {r['con']:.3f} {r['com']:.3f} {r['fmt']:.3f} {r['lf']:.3f} {r['avg']:.3f} {r['bad_pct']:5.1f}%")
        print()


if __name__ == "__main__":
    main()
