#!/usr/bin/env python3
"""逐 step 提取可疑指标细序列 + chi2/resp_len 尖峰定位。"""
import re

LOG = "/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/logs/train.log"

FULL = {
    "vopd": "actor/vopd_loss",
    "resp_len": "response_length/mean",
    "clip": "response_length/clip_ratio",
    "chi2": "rollout_corr/chi2_token",
    "ro_ppl": "rollout_corr/rollout_ppl",
    "tr_ppl": "rollout_corr/training_ppl",
    "kl": "rollout_corr/kl",
    "tkmass": "teacher_topk_mass_mean",
    "s_tk": "student_on_teacher_topk_mass_mean",
    "is_max": "rollout_corr/rollout_is_max",
}

rows = []
for line in open(LOG, errors="replace"):
    if "step:" in line and "TaskRunner" in line:
        m = re.search(r"step:(\d+)", line)
        if not m:
            continue
        r = {"step": int(m.group(1))}
        for k, full in FULL.items():
            mm = re.search(re.escape(full) + r":(?:np\.\w+\()?([0-9.eE+-]+)\)?", line)
            if mm:
                r[k] = float(mm.group(1))
        rows.append(r)

# 每 25 step 采样
hdr = ["step"] + list(FULL.keys())
print("  ".join(f"{h:>8}" for h in hdr))
for r in rows[::25]:
    cells = [f"{r['step']:>8}"]
    for k in FULL:
        cells.append(f"{r[k]:>8.4g}" if k in r else f"{'—':>8}")
    print("  ".join(cells))

# chi2 尖峰
chi2_hi = [(r["step"], r["chi2"]) for r in rows if "chi2" in r and r["chi2"] > 10]
print(f"\nchi2>10 的 step: {chi2_hi[:15]}")
# chi2 全序列分位
chi2_all = sorted(r["chi2"] for r in rows if "chi2" in r)
if chi2_all:
    n = len(chi2_all)
    print(f"chi2 分位: p50={chi2_all[n//2]:.2f} p90={chi2_all[int(n*0.9)]:.2f} p99={chi2_all[int(n*0.99)]:.2f} max={chi2_all[-1]:.2f}")

# resp_len 趋势: 用线性回归斜率
rl = [(r["step"], r["resp_len"]) for r in rows if "resp_len" in r]
if rl:
    xs = [s for s, _ in rl]
    ys = [v for _, v in rl]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x,y in rl)
    den = sum((x-mx)**2 for x in xs)
    slope = num/den if den else 0
    print(f"\nresp_len 线性斜率: {slope:.3f} token/step (正=上涨) 起={ys[0]:.0f} 末={ys[-1]:.0f}")
    rl_top = sorted(rl, key=lambda x: -x[1])[:5]
    print(f"resp_len 最高5步: {rl_top}")

# ro_ppl 趋势
rp = [(r["step"], r["ro_ppl"]) for r in rows if "ro_ppl" in r]
if rp:
    print(f"ro_ppl: 起={rp[0][1]:.1f} 末={rp[-1][1]:.1f} max={max(v for _,v in rp):.1f}")

# tkmass / s_tk 缺口 (teacher topk mass - student on teacher topk)
gap = [(r["step"], r.get("tkmass",0) - r.get("s_tk",0)) for r in rows if "tkmass" in r and "s_tk" in r]
if gap:
    print(f"\nteacher-student topk mass 缺口: 起={gap[0][1]:.4f} 末={gap[-1][1]:.4f} (正=teacher覆盖>student, 师生有差)")
