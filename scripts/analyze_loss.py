#!/usr/bin/env python3
"""解析 verl train.log 的 step 指标行，做趋势健康分析。

用法:
    python scripts/analyze_loss.py [train.log 路径]
    默认: outputs/flashnote_train_v4/logs/train.log
"""
import re
import sys
import os
import statistics

DEFAULT_LOG = "/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/logs/train.log"

# 要提取的指标 (key, 简称, 健康)
METRICS = [
    # ① 蒸馏是否生效
    ("self_distillation/num_distill_tokens", "dist_tok", " >0"),
    ("self_distillation/teacher_image_swap_fraction", "img_swap", " =1.0"),
    ("self_distillation/policy_fallback_fraction", "pg_fb", " =0"),
    ("self_distillation/grpo_fallback_count", "grpo_fb", " =0"),
    ("self_distillation/empty_target_batch", "empty_tb", " =0"),
    ("self_distillation/teacher_always_on_fraction", "t_on", " =1.0"),
    # ② 师生分布对齐
    ("actor/vopd_loss", "vopd", " 缓降"),
    ("rollout_corr/kl", "kl", " 0.2-0.5"),
    ("rollout_corr/k3_kl", "k3kl", " 趋同"),
    ("rollout_corr/ppl_ratio", "ppl_r", " ~1"),
    ("rollout_corr/log_ppl_diff", "lpdiff", " <0.5"),
    ("rollout_corr/chi2_token", "chi2", " <5"),
    # mopd 铁证 (走对分支才有)
    ("mopd_reverse_kl_term_mean", "mopd_rkl", " 存在"),
    ("mopd_bias_correction_mean", "mopd_bc", " 存在"),
    ("teacher_topk_mass_mean", "tkmass", " 存在"),
    ("student_on_teacher_topk_mass_mean", "s_tk", " 存在"),
    ("raw_jsd_token_mean", "raw_jsd", " 通用(两分支都有)"),
    # ③ 重要性采样
    ("rollout_corr/rollout_is_mean", "is_mean", " ~1"),
    ("rollout_corr/rollout_is_std", "is_std", " <1"),
    ("rollout_corr/rollout_is_max", "is_max", " <5"),
    # ④ 训练稳定性
    ("actor/grad_norm", "grad", " 10-30"),
    ("actor/lr", "lr", " 2e-6"),
    ("rollout_corr/training_ppl", "tr_ppl", " 稳/降"),
    ("rollout_corr/rollout_ppl", "ro_ppl", " 近tr"),
    # ⑤ 生成质量
    ("response_length/mean", "resp_len", " ~270稳"),
    ("response_length/clip_ratio", "clip", " <0.05"),
    ("response/aborted_ratio", "abort", " =0"),
    # ⑥ 效率显存
    ("perf/max_memory_allocated_gb", "mem_gb", " <125"),
    ("timing_s/step", "s/step", " ~210"),
    ("perf/mfu/actor", "mfu", " 0.2-0.3"),
    ("perf/throughput", "thru", " >300"),
]


def parse_line(line):
    """从一行 step 指标里提取所有 key->value (float)。"""
    out = {}
    # step 号
    m = re.search(r"step:(\d+)", line)
    if not m:
        return None
    out["step"] = int(m.group(1))
    for key, short, _ in METRICS:
        pat = re.escape(key) + r":(?:np\.\w+\()?([0-9.eE+-]+)\)?"
        mm = re.search(pat, line)
        if mm:
            try:
                out[short] = float(mm.group(1))
            except ValueError:
                pass
    return out


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    if not os.path.exists(log):
        print(f"❌ 找不到 {log}")
        sys.exit(1)

    rows = []
    with open(log, errors="replace") as f:
        for line in f:
            if "step:" in line and "TaskRunner" in line:
                r = parse_line(line)
                if r and r.get("step", -1) >= 0:
                    rows.append(r)

    if not rows:
        print("❌ 没解析到 step 行")
        sys.exit(1)

    n = len(rows)
    steps = [r["step"] for r in rows]
    print(f"=== 解析到 {n} 个 step，范围 {steps[0]}-{steps[-1]} ===\n")

    # 分三段看趋势
    seg = max(1, n // 3)
    segs = [("前段", rows[:seg]), ("中段", rows[seg:2*seg]), ("后段", rows[2*seg:])]

    print(f"{'指标':<10}{'健康':<14}{'前段均':>12}{'中段均':>12}{'后段均':>12}{'末5步均':>12}  趋势")
    print("-" * 90)
    for key, short, health in METRICS:
        vals_all = [r[short] for r in rows if short in r]
        if not vals_all:
            print(f"{short:<10}{health:<14}{'—':>12}{'—':>12}{'—':>12}{'—':>12}  缺失")
            continue
        seg_means = []
        for name, seg_rows in segs:
            sv = [r[short] for r in seg_rows if short in r]
            seg_means.append(statistics.mean(sv) if sv else float("nan"))
        last5 = [r[short] for r in rows[-5:] if short in r]
        last5_m = statistics.mean(last5) if last5 else float("nan")
        # 趋势: 后段 vs 前段
        if seg_means[0] and seg_means[2]:
            delta = seg_means[2] - seg_means[0]
            pct = (delta / abs(seg_means[0]) * 100) if seg_means[0] != 0 else 0
            trend = f"{'↑' if delta>0 else '↓' if delta<0 else '→'}{pct:+.1f}%"
        else:
            trend = "—"
        fmt = lambda x: f"{x:>12.4g}" if x == x else f"{'—':>12}"
        print(f"{short:<10}{health:<14}{fmt(seg_means[0])}{fmt(seg_means[1])}{fmt(seg_means[2])}{fmt(last5_m)}  {trend}")

    # 关键诊断
    print("\n=== 关键诊断 ===")
    last = rows[-1]
    # ① 蒸馏生效
    dt = [r.get("dist_tok") for r in rows if r.get("dist_tok") is not None]
    print(f"① 蒸馏生效: num_distill_tokens 全程 >0? {all(v>0 for v in dt)} (min={min(dt):.0f}, max={max(dt):.0f})")
    img = set(r.get("img_swap") for r in rows if r.get("img_swap") is not None)
    print(f"   img_swap 取值: {img}")
    pg = [r.get("pg_fb") for r in rows if r.get("pg_fb") is not None]
    print(f"   policy_fallback 全=0? {all(v==0 for v in pg)} (max={max(pg) if pg else 'NA'})")
    # 分支确认
    has_mopd = any("mopd_rkl" in r for r in rows)
    has_jsd = any("raw_jsd" in r for r in rows)
    print(f"② 分支: 有 mopd_* 指标={has_mopd} (走对), 有 raw_jsd={has_jsd} (走错)")
    # vopd_loss 趋势
    vopd = [r["vopd"] for r in rows if "vopd" in r]
    if vopd:
        print(f"③ vopd_loss: 起={vopd[0]:.4f} 末5均={statistics.mean(vopd[-5:]):.4f} min={min(vopd):.4f} max={max(vopd):.4f}")
    # response_length mode collapse
    rl = [r["resp_len"] for r in rows if "resp_len" in r]
    if rl:
        print(f"④ resp_len: 起={rl[0]:.0f} 末5均={statistics.mean(rl[-5:]):.0f} max={max(rl):.0f} min={min(rl):.0f}  {'⚠️上涨!' if rl[-1] > rl[0]*1.3 else '稳定'}")
    # kl 趋势
    kl = [r["kl"] for r in rows if "kl" in r]
    if kl:
        print(f"⑤ kl: 起={kl[0]:.4f} 末5均={statistics.mean(kl[-5:]):.4f} min={min(kl):.4f} max={max(kl):.4f}")
    # grad norm
    g = [r["grad"] for r in rows if "grad" in r]
    if g:
        print(f"⑥ grad_norm: 末5均={statistics.mean(g[-5:]):.1f} max={max(g):.1f} {'⚠️>100发散' if max(g)>100 else '正常'}")
    # 显存
    mem = [r["mem_gb"] for r in rows if "mem_gb" in r]
    if mem:
        print(f"⑦ 显存峰值: max={max(mem):.1f}GB {'⚠️逼近145' if max(mem)>130 else 'OK'}")
    # 速度
    sp = [r["s/step"] for r in rows if "s/step" in r]
    if sp:
        print(f"⑧ 速度: 末5均={statistics.mean(sp[-5:]):.0f}s/step (前5均={statistics.mean(sp[:5]):.0f}s)")


if __name__ == "__main__":
    main()
