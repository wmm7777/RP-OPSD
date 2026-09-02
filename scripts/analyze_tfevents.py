#!/usr/bin/env python3
"""从 tensorboard tfevents 解析标量做趋势健康分析（文本 log 不在本机时用）。

用法:
    python scripts/analyze_tfevents.py <tfevents 目录或父目录>
"""
import sys
import os
import statistics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 与 analyze_loss.py 的指标对齐 (tfevents tag 名 = verl log key 名)
METRICS = [
    ("self_distillation/num_distill_tokens", "dist_tok", " >0"),
    ("self_distillation/teacher_image_swap_fraction", "img_swap", " =1.0"),
    ("self_distillation/policy_fallback_fraction", "pg_fb", " =0"),
    ("self_distillation/grpo_fallback_count", "grpo_fb", " =0"),
    ("self_distillation/empty_target_batch", "empty_tb", " =0"),
    ("self_distillation/teacher_always_on_fraction", "t_on", " =1.0"),
    ("actor/vopd_loss", "vopd", " 缓降"),
    ("rollout_corr/kl", "kl", " 0.2-0.5"),
    ("rollout_corr/k3_kl", "k3kl", " 趋同"),
    ("rollout_corr/ppl_ratio", "ppl_r", " ~1"),
    ("rollout_corr/log_ppl_diff", "lpdiff", " <0.5"),
    ("rollout_corr/chi2_token", "chi2", " <5"),
    ("self_distillation/mopd_reverse_kl_term_mean", "mopd_rkl", " 存在"),
    ("self_distillation/mopd_bias_correction_mean", "mopd_bc", " 存在"),
    ("self_distillation/teacher_topk_mass_mean", "tkmass", " 存在"),
    ("self_distillation/student_on_teacher_topk_mass_mean", "s_tk", " 存在"),
    ("self_distillation/raw_jsd_token_mean", "raw_jsd", " 通用"),
    ("rollout_corr/rollout_is_mean", "is_mean", " ~1"),
    ("rollout_corr/rollout_is_std", "is_std", " <1"),
    ("rollout_corr/rollout_is_max", "is_max", " <5"),
    ("actor/grad_norm", "grad", " 10-30"),
    ("actor/lr", "lr", " 2e-6"),
    ("rollout_corr/training_ppl", "tr_ppl", " 稳/降"),
    ("rollout_corr/rollout_ppl", "ro_ppl", " 近tr"),
    ("response_length/mean", "resp_len", " ~270稳"),
    ("response_length/clip_ratio", "clip", " <0.05"),
    ("response/aborted_ratio", "abort", " =0"),
    ("perf/max_memory_allocated_gb", "mem_gb", " <125"),
    ("timing_s/step", "s/step", " ~210"),
    ("perf/mfu/actor", "mfu", " 0.2-0.3"),
    ("perf/throughput", "thru", " >300"),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = args[0] if args else \
        "/data4/wumeimei/flash_note/RP-OPSD/tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B-v3sft"
    if not os.path.isdir(path):
        print(f"❌ {path} 不是目录")
        sys.exit(1)

    ea = EventAccumulator(path, size_guidance={"scalars": 0, "tensors": 0, "histograms": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"=== tfevents 目录: {path} ===")
    print(f"scalar tag 数: {len(tags)}\n")

    # 收集每个指标序列
    series = {}  # short -> [(step, value)]
    all_tags_present = []
    for full, short, _ in METRICS:
        # verl 可能带前缀也可能不带，尝试精确 + 前缀模糊
        if full in tags:
            scals = ea.Scalars(full)
            series[short] = [(s.step, s.value) for s in scals]
            all_tags_present.append(full)
    # 打印未匹配的 tag（前 30 个）供调试
    matched = set(all_tags_present)
    unmatched_keys = [f for f, _, _ in METRICS if f not in matched]

    if not series:
        print("❌ 没匹配到任何预定义指标 tag，全部 tag 列表（前 40）:")
        for t in tags[:40]:
            print(f"  {t}")
        return

    n = max(len(v) for v in series.values())
    any_steps = sorted(set(s for seq in series.values() for s, _ in seq))
    print(f"匹配指标 {len(matched)}/{len(METRICS)}，step 范围 {any_steps[0]}-{any_steps[-1]}，共 {len(any_steps)} 步\n")

    # 分三段
    sn = len(any_steps)
    seg_idx = [any_steps[:sn//3], any_steps[sn//3:2*sn//3], any_steps[2*sn//3:]]
    seg_names = ["前段", "中段", "后段"]

    print(f"{'指标':<10}{'健康':<14}{'前段均':>12}{'中段均':>12}{'后段均':>12}{'末5步均':>12}  趋势")
    print("-" * 92)
    for full, short, health in METRICS:
        if short not in series:
            continue
        seq = series[short]
        step_to_val = dict(seq)
        seg_means = []
        for seg in seg_idx:
            vals = [step_to_val[s] for s in seg if s in step_to_val]
            seg_means.append(statistics.mean(vals) if vals else float("nan"))
        last_steps = sorted(step_to_val.keys())[-5:]
        last5 = [step_to_val[s] for s in last_steps]
        last5_m = statistics.mean(last5) if last5 else float("nan")
        if seg_means[0] == seg_means[0] and seg_means[2] == seg_means[2] and seg_means[0] != 0:
            delta = seg_means[2] - seg_means[0]
            pct = delta / abs(seg_means[0]) * 100
            trend = f"{'↑' if delta > 0 else '↓' if delta < 0 else '→'}{pct:+.1f}%"
        else:
            trend = "—"
        f = lambda x: f"{x:>12.4g}" if x == x else f"{'—':>12}"
        print(f"{short:<10}{health:<14}{f(seg_means[0])}{f(seg_means[1])}{f(seg_means[2])}{f(last5_m)}  {trend}")

    # 关键诊断
    print("\n=== 关键诊断 ===")
    dt = [v for _, v in series.get("dist_tok", [])]
    if dt:
        print(f"① 蒸馏生效: num_distill_tokens 全程>0? {all(v>0 for v in dt)} (min={min(dt):.0f} max={max(dt):.0f})")
    img = set(v for _, v in series.get("img_swap", []))
    if img:
        print(f"   img_swap 取值: {img}")
    pg = [v for _, v in series.get("pg_fb", [])]
    if pg:
        print(f"   policy_fallback 全=0? {all(v==0 for v in pg)} (max={max(pg)})")
    has_mopd = "mopd_rkl" in series
    has_jsd = "raw_jsd" in series
    print(f"② 分支: 有 mopd_*={has_mopd} (走对), 有 raw_jsd={has_jsd} (通用非判据)")
    vopd = [v for _, v in series.get("vopd", [])]
    if vopd:
        print(f"③ vopd_loss: 起={vopd[0]:.4f} 末5均={statistics.mean(vopd[-5:]):.4f} min={min(vopd):.4f} max={max(vopd):.4f}")
    rl = [v for _, v in series.get("resp_len", [])]
    if rl:
        print(f"④ resp_len: 起={rl[0]:.0f} 末5均={statistics.mean(rl[-5:]):.0f} max={max(rl):.0f} min={min(rl):.0f}  {'⚠️上涨' if rl[-1] > rl[0]*1.3 else '稳定'}")
    kl = [v for _, v in series.get("kl", [])]
    if kl:
        print(f"⑤ kl: 起={kl[0]:.4f} 末5均={statistics.mean(kl[-5:]):.4f} min={min(kl):.4f} max={max(kl):.4f}")
    g = [v for _, v in series.get("grad", [])]
    if g:
        print(f"⑥ grad_norm: 末5均={statistics.mean(g[-5:]):.1f} max={max(g):.1f} {'⚠️>100发散' if max(g)>100 else '正常'}")
    mem = [v for _, v in series.get("mem_gb", [])]
    if mem:
        print(f"⑦ 显存峰值: max={max(mem):.1f}GB {'⚠️逼近145' if max(mem)>130 else 'OK'}")
    tk = [(s, v) for s, v in series.get("tkmass", [])]
    sk = [(s, v) for s, v in series.get("s_tk", [])]
    if tk:
        print(f"⑧ tkmass: 起={tk[0][1]:.4f} 末={tk[-1][1]:.4f} (下降=teacher分布变平)")
    if tk and sk:
        gap0 = tk[0][1] - sk[0][1]
        gap1 = tk[-1][1] - sk[-1][1]
        print(f"   tkmass-s_tk 缺口: 起={gap0:.4f} 末={gap1:.4f}")

    if "--detail" in sys.argv:
        print("\n=== 逐 step 细序列（每 30 步采样 + 拐点）===")
        detail_tags = [
            ("actor/vopd_loss", "vopd"),
            ("response_length/mean", "resp_len"),
            ("response_length/clip_ratio", "clip"),
            ("actor/grad_norm", "grad"),
            ("rollout_corr/rollout_ppl", "ro_ppl"),
            ("rollout_corr/training_ppl", "tr_ppl"),
            ("rollout_corr/chi2_token", "chi2"),
        ]
        for full, short in detail_tags:
            if full not in tags:
                continue
            scals = ea.Scalars(full)
            print(f"\n[{short}] 共 {len(scals)} 点，每 30 步采样:")
            for i in range(0, len(scals), 30):
                print(f"  step {scals[i].step}: {scals[i].value:.4g}")
            # 拐点: 连续 3 步涨幅 > 20% 的位置
            vals = [s.value for s in scals]
            steps = [s.step for s in scals]
            spikes = []
            for i in range(3, len(scals)):
                base = statistics.mean(vals[i-3:i]) if vals[i-3:i] else vals[i]
                if base and abs(vals[i] - base) / abs(base) > 0.5:
                    spikes.append((steps[i], vals[i]))
            if spikes:
                print(f"  ⚠️ 突变点(相对前3步均±50%): {spikes[:8]}")


if __name__ == "__main__":
    main()
