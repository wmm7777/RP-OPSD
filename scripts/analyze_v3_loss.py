#!/usr/bin/env python3
"""分析 v3_no_ema 训练 loss 走势是否正常"""
import re, statistics, sys

LOG = "/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v3_no_ema/logs/train.log"

KEYS = [
    "actor/pg_loss",
    "actor/vopd_loss",
    "actor/grpo_loss",
    "actor/kl_loss",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "rollout_corr/training_ppl",
    "rollout_corr/rollout_ppl",
    "rollout_corr/log_ppl_diff",
    "self_distillation/mopd_reverse_kl_term_mean",
    "self_distillation/teacher_always_on_fraction",
    "self_distillation/teacher_image_swap_fraction",
    "self_distillation/policy_fallback_fraction",
    "self_distillation/grpo_fallback_count",
    "rollout_corr/rollout_is_mean",
    "timing_s/agent_loop/generate_sequences/mean",
    "timing_s/update_actor/backward",
    "timing_s/update_actor/teacher_forward",
    "timing_s/update_actor/student_forward",
    "timing_s/update_actor/optimizer_step",
    "perf/time_per_step",
    "global_seqlen/mean",
]

# parse
steps = {}
cur = None
with open(LOG, errors="ignore") as f:
    for ln in f:
        m = re.search(r"step:(\d+) ", ln)
        if m:
            cur = int(m.group(1))
            steps.setdefault(cur, {})
        if cur is None:
            continue
        for k in KEYS:
            mm = re.search(re.escape(k) + r":np\.float64\(([\-0-9.eE]+)\)", ln)
            if mm:
                try:
                    steps[cur][k] = float(mm.group(1))
                except ValueError:
                    pass
        # training/global_step is int
        mm = re.search(r"training/global_step:(\d+)", ln)
        if mm:
            steps[cur]["training/global_step"] = int(mm.group(1))

if not steps:
    print("no steps parsed"); sys.exit(1)

all_steps = sorted(steps)
print(f"parsed {len(all_steps)} steps, range [{all_steps[0]}, {all_steps[-1]}]")
print()

# 按 100 步分桶
buckets = [(0, 99), (100, 199), (200, 299), (300, 399), (400, 499),
           (500, 599), (600, 699), (700, 799), (800, 899), (900, 999)]

def agg(vals, fn="mean"):
    if not vals:
        return None
    if fn == "mean":
        return statistics.mean(vals)
    if fn == "std":
        return statistics.stdev(vals) if len(vals) > 1 else 0.0
    if fn == "min":
        return min(vals)
    if fn == "max":
        return max(vals)
    return None

print(f"{'step_range':12} {'n':>3} | {'pg_loss':>8} {'vopd':>8} {'kl_rollout':>10} {'kl_term':>8} {'tr_ppl':>8} {'roll_ppl':>9} {'logppl_d':>9} {'t_always':>8} {'t_swap':>6} {'fallback':>8} | {'gen_s':>6} {'backward_s':>10}")
print("-" * 140)
for lo, hi in buckets:
    bucket = [s for s in all_steps if lo <= s <= hi]
    if not bucket:
        continue
    rows = [steps[s] for s in bucket]
    def col(key, fn="mean"):
        vals = [r.get(key) for r in rows if key in r]
        return agg(vals, fn)
    n = len(bucket)
    print(f"{lo:4}-{hi:<6} {n:3} | "
          f"{col('actor/pg_loss') or 0:>8.4f} "
          f"{col('actor/vopd_loss') or 0:>8.4f} "
          f"{col('rollout_corr/kl') or 0:>10.4f} "
          f"{col('self_distillation/mopd_reverse_kl_term_mean') or 0:>8.4f} "
          f"{col('rollout_corr/training_ppl') or 0:>8.2f} "
          f"{col('rollout_corr/rollout_ppl') or 0:>9.2f} "
          f"{col('rollout_corr/log_ppl_diff') or 0:>9.4f} "
          f"{col('self_distillation/teacher_always_on_fraction') or 0:>8.3f} "
          f"{col('self_distillation/teacher_image_swap_fraction') or 0:>6.3f} "
          f"{col('self_distillation/policy_fallback_fraction') or 0:>8.3f} | "
          f"{col('timing_s/agent_loop/generate_sequences/mean') or 0:>6.1f} "
          f"{col('timing_s/update_actor/backward') or 0:>10.1f}")

# 关键拐点：是否有 spike
print()
print("=== 异常 spike 检测 ===")
for k in ["actor/pg_loss", "rollout_corr/kl", "self_distillation/mopd_reverse_kl_term_mean"]:
    vals = [(s, steps[s].get(k)) for s in all_steps if k in steps[s]]
    if not vals:
        continue
    mean_v = statistics.mean([v for _, v in vals])
    std_v = statistics.stdev([v for _, v in vals]) if len(vals) > 1 else 0
    outliers = [(s, v) for s, v in vals if abs(v - mean_v) > 3 * std_v and std_v > 0]
    print(f"  {k}: mean={mean_v:.4f}, std={std_v:.4f}, 3σ outliers={len(outliers)}")
    for s, v in outliers[:5]:
        print(f"    step{s}: {v:.4f}")

# 检查 fallback 是否触发（policy_fallback_fraction > 0 说明 teacher 跑挂了）
print()
print("=== teacher 健康度 ===")
for s in [1, 50, 100, 200, 300, 500, 700, 900]:
    if s in steps:
        r = steps[s]
        print(f"  step{s}: teacher_always_on={r.get('self_distillation/teacher_always_on_fraction', '?')}, "
              f"teacher_image_swap={r.get('self_distillation/teacher_image_swap_fraction', '?')}, "
              f"policy_fallback={r.get('self_distillation/policy_fallback_fraction', '?')}, "
              f"grpo_fallback={r.get('self_distillation/grpo_fallback_count', '?')}")

# 输出最后一步的完整字段
print()
print("=== 最新 step 的完整 metrics ===")
last = steps[all_steps[-1]]
for k in KEYS:
    if k in last:
        print(f"  {k}: {last[k]}")
