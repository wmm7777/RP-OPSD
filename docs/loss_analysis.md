# flash_note RP-OPSD Loss 分析报告

> 生成时间：2026-09-03
> 数据来源：TensorBoard event 文件，`tensorboard_log/RP-OPSD/` 下 4 个实验目录
> 源码依据：`verl/trainer/ppo/core_algos.py:1090-1240`

---

## 一、实验配置总览

四个实验**唯一变量是教师来源与更新方式**，其余超参完全一致：

| 配置项 | 值 | 说明 |
|--------|------|------|
| 蒸馏目标 | `mopd_topk_reverse_kl` | MOPD 论文 Eq.(5) bias-corrected reverse-KL |
| alpha | 1.0 | 纯反向 KL（不走 generalized JSD 分支） |
| distillation_topk | 100 | 教师 top-100 token 构成支撑集 |
| topk_source | teacher | 教师选支撑集，学生不能改变 |
| add_tail | False | 无 tail bucket，无重归一化 |
| is_clip | 2.0 | IS 权重上限 |
| LR | 2e-6 | 学习率 |
| lr_warmup_steps | 75 | warmup 步数 |
| total_epochs | 2 | 训练轮数 |
| max_prompt / max_response | 3072 / 2048 | 5k 口径 |
| train_batch_size | 96 | 全局 batch |
| rollout_n | 8 | 每 prompt 采样 8 条 |

### 四实验差异表

| | v3sft | v3-no-ema | v4-fixed-teacher | v5-teacher27B |
|---|---|---|---|---|
| **学生起点** | SFT merged (step_2250, MOS 4.06) | base Qwen3.5-9B | SFT merged (step_2250) | base Qwen3.5-9B |
| **教师来源** | legacy (EMA) | legacy (EMA) | legacy (EMA) | fixed (外部模型) |
| **teacher_update_rate** | **0.05** | **0.0** | **0.0** | **0.0** |
| **教师实质** | 慢速跟随学生的 EMA 副本 | 冻结在 base 9B 初始值 | 冻结在 SFT 初始值 | 固定 Qwen3.8-27B |
| **运行机器** | 机器3 (m3, 跨机) | 机器3 (m3, 跨机) | 机器4 (m4, 本机) | 机器4 (m4, 本机) |
| **步数** | 493 (2 epoch 完成) | 242 (进行中) | 216 | ~5 (刚起步) |
| **设计目的** | 原始 OPD（EMA 教师） | 关 EMA 测退化 | 冻教师治退化 | 外部强教师蒸馏 |

### 输出文件目录

| 实验 | TensorBoard 日志 | OUTPUT_DIR | Checkpoints |
|------|------------------|------------|-------------|
| v3sft | `tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B-v3sft/` | `/data3/wumeimei/flash_note/flashnote_train_v3sft` | `global_step_{150,300,450}` |
| v3-no-ema | `tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B-v3-no-ema/` | `/data1/meimei.wu/flash_note/flashnote_train_v3_no_ema` (m3 本地) | 暂无（save_freq=150，242 步应已有 step_150） |
| v4-fixed-teacher | `tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B-v4-fixed-teacher/` | `outputs/flashnote_train_v4_fixed_teacher/` | `global_step_150` |
| v5-teacher27B | `tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B-v5-teacher27B-no-ema/` | `outputs/flashnote_train_v5_teacher27B/` | 暂无（仅 5 步） |

> 注：所有路径相对于 `/data4/wumeimei/flash_note/RP-OPSD/`，除非以绝对路径给出。v3sft 和 v3-no-ema 在机器3跨机运行，OUTPUT_DIR 指向 m3 本地磁盘。

### 启动脚本

| 实验 | 脚本路径 |
|------|----------|
| v3sft | `scripts/run_rp_opsd_v3sft.sh` |
| v3-no-ema | `scripts/run_rp_opsd_v3_no_ema.sh` |
| v4-fixed-teacher | `scripts/run_rp_opsd_v4.sh` |
| v5-teacher27B | `scripts/run_rp_opsd_v5_teacher27B.sh` |

---

## 二、Loss 公式拆解

### 2.1 核心损失公式（MOPD Eq.(5)）

源码：`core_algos.py:1133-1146`

```
L_token = Σ_v [ p_s(v) · (log p_s(v) - log p_t(v)) ]  +  Σ_v [ p_t(v) - p_s(v) ]
          \_________________________________________/     \____________________/
                   reverse_kl_term                         bias_correction
```

其中：
- **p_s(v)** = 学生在教师 top-k 支撑集上的概率（`student_probs = student_logp.exp()`）
- **p_t(v)** = 教师在自身 top-k 支撑集上的概率（`teacher_probs = teacher_logp.exp()`）
- **v** 遍历教师的 top-100 token

#### reverse_kl_term（反向 KL 主项）

```python
reverse_kl_term = student_probs * (student_logp - teacher_logp)
# = Σ_v p_s(v) · log[p_s(v) / p_t(v)]
# = D_KL(p_s ‖ p_t) 的逐 token 值
```

这是反向 KL 散度 D_KL(p_学生 ‖ p_教师)。衡量学生要多少"额外惊讶"才能匹配教师分布。**学生驱动**——梯度从学生概率方向出发，鼓励学生把质量集中到教师高概率的 token 上。

#### bias_correction（偏差校正项）

```python
bias_correction = teacher_probs - student_probs
# = Σ_v [ p_t(v) - p_s(v) ]
```

因为在截断的 top-k 支撑上计算（排除了 vocab 中其余 token），原始反向 KL 会有截断偏差。此项校正使梯度在完整 vocab 上的期望无偏。这是 MOPD 相对于普通 reverse-KL 蒸馏的关键改进。

### 2.2 IS 权重加权

源码：`core_algos.py:1213-1222`

```python
negative_approx_kl = (student_log_probs - old_log_probs).detach()  # log(π_train / π_rollout)
negative_approx_kl = clamp(negative_approx_kl, min=-20, max=20)
ratio = exp(negative_approx_kl).clamp(max=is_clip)  # is_clip=2.0
weighted_per_token_loss = raw_per_token_loss * ratio
```

- **old_log_probs** = rollout 时的旧策略 log 概率
- **student_log_probs** = 当前训练步的策略 log 概率
- **ratio** = π_train / π_rollout，衡量策略漂移程度
- clip 在 2.0 防止 importance sampling 权重爆炸

如果训练策略和 rollout 策略一致（on-policy），ratio ≈ 1.0。ratio 偏离 1.0 越多，说明策略漂移越严重，IS 权重补偿越大。

### 2.3 指标与公式的对应关系

| TensorBoard tag | 公式 | 含义 |
|---|---|---|
| `actor/vopd_loss` | `mean(raw_loss × ratio)` over valid tokens | **实际优化的 loss**，IS 加权后 |
| `self_distillation/raw_jsd_token_mean` | `mean(raw_loss)` over valid tokens | IS 加权前的原始 loss |
| `self_distillation/weighted_jsd_token_mean` | 同 vopd_loss | IS 加权后的 loss（与 vopd_loss 等价） |
| `self_distillation/mopd_reverse_kl_term_mean` | `mean(Σ p_s·(log p_s - log p_t))` | 反向 KL 主项的逐 token 均值 |
| `self_distillation/mopd_bias_correction_mean` | `mean(Σ (p_t - p_s))` | 偏差校正项的逐 token 均值 |
| `self_distillation/student_on_teacher_topk_mass_mean` | `Σ p_s(v)` on top-k | 学生概率落在教师 top-k 上的质量 |
| `self_distillation/teacher_topk_mass_mean` | `Σ p_t(v)` on top-k | 教师自身 top-k 的概率质量 |
| `rollout_corr/kl` | `KL(π_rollout ‖ π_train)` | on-policy 一致性，越低越好 |
| `rollout_corr/ppl_ratio` | `training_ppl / rollout_ppl` | 应接近 1.0，偏离 = IS 失稳 |
| `rollout_corr/log_ppl_diff` | `log(training_ppl) - log(rollout_ppl)` | log 尺度的策略漂移 |
| `actor/grad_norm` | 梯度范数 | 训练稳定性指标 |
| `response_length/mean` | rollout 平均回复长度 | 生成行为变化 |
| `response_length/clip_ratio` | 回复超长被截断比例 | 数据质量 |

> **命名注意**：tag 名含 `jsd` 是历史遗留。`alpha=1.0` + `mopd_topk_reverse_kl` 分支下，实际计算的是反向 KL，不是 JSD。`raw_jsd_token_mean` = `raw_distillation_token_mean` = 原始反向 KL loss。

### 2.4 指标分组：应该看哪些

| 优先级 | 指标 | 关注点 |
|--------|------|--------|
| 🔴 必看 | `actor/vopd_loss` | loss 是否下降 |
| 🔴 必看 | `self_distillation/raw_jsd_token_mean` | 师生分布差异是否缩小 |
| 🔴 必看 | `student_on_teacher_topk_mass` vs `teacher_topk_mass` | 两者是否同步下降（EMA 退化信号） |
| 🔴 必看 | `actor/grad_norm` | 是否爆炸 |
| 🔴 必看 | `rollout_corr/kl` | on-policy 是否成立 |
| 🟡 扫一眼 | `rollout_corr/ppl_ratio` | IS 权重是否稳定 |
| 🟡 扫一眼 | `response_length/mean` | 生成长度是否合理 |
| 🟡 扫一眼 | `mopd_bias_correction_mean` | 偏差校正项是否发散 |
| ⚪ 忽略 | `timing_*`, `perf_*`, `global_seqlen/*`, `rollout_is_*` | 工程统计 |

---

## 三、逐实验趋势分析

### 3.1 v3sft — EMA 教师 + SFT 热启动（493 步，2 epoch 完成）

**配置要点**：student=SFT ckpt，teacher=EMA(rate=0.05) 从 SFT 初始化，教师慢速跟随学生。

#### vopd_loss：0.214 → 0.007（↓97%）

| step | 1 | 55 | 110 | 165 | 219 | 274 | 329 | 383 | 438 | 493 |
|------|---|---|-----|-----|-----|-----|-----|-----|-----|-----|
| loss | 0.214 | 0.141 | 0.069 | 0.053 | 0.036 | 0.023 | 0.015 | 0.012 | 0.008 | 0.007 |

完美指数下降，看似收敛极好。

#### raw_jsd_token_mean：0.214 → 0.0065

与 vopd_loss 几乎重合（IS 权重接近 1，加权无差异）。

#### grad_norm：11.3 → 1.2，中间有 171 尖峰

| step | 1 | 55 | 110 | 165 | 219 | 274 | 329 | 383 | 438 | 493 |
|------|---|---|-----|-----|-----|-----|-----|-----|-----|-----|
| grad | 11.3 | 5.9 | 5.2 | 5.3 | 21.9 | 1.6 | 2.2 | 2.0 | 1.1 | 1.2 |

step 219 附近有一次大幅 spike（max=171.3），之后恢复平稳。可能是某些 batch 的极端 IS 权重导致。

#### ⚠️ student_on_teacher_topk_mass + teacher_topk_mass：双双下降

| step | 1 | 55 | 110 | 165 | 219 | 274 | 329 | 383 | 438 | 493 |
|------|---|---|-----|-----|-----|-----|-----|-----|-----|-----|
| student_mass | 0.994 | 0.994 | 0.986 | 0.972 | 0.950 | 0.926 | 0.903 | 0.882 | 0.861 | 0.841 |
| teacher_mass | 0.997 | 0.996 | 0.989 | 0.976 | 0.954 | 0.929 | 0.907 | 0.886 | 0.864 | 0.845 |
| **gap** | 0.003 | 0.002 | 0.003 | 0.004 | 0.004 | 0.003 | 0.004 | 0.004 | 0.003 | 0.004 |

**这是核心问题**：师生概率质量从 0.99 双双降到 0.84，但 gap 始终 0.003-0.004。教师（EMA 副本）和学生在同步"变平"——分布在变宽、置信度在降。loss 降到 0.007 不是因为学生学到了更好的分布，而是因为 EMA 教师在跟随学生退化，两者始终保持相似 → 反向 KL 自然趋零。

**这就是 EMA 正反馈退化**：教师 = EMA(学生) → 学生更新 → 教师跟着更新 → 师生距离永远很小 → loss 趋零但不代表真实学习。

#### rollout_corr/kl：0.179 → 0.013 ✅

| step | 1 | 55 | 110 | 165 | 219 | 274 | 329 | 383 | 438 | 493 |
|------|---|---|-----|-----|-----|-----|-----|-----|-----|-----|
| kl | 0.179 | 0.065 | 0.040 | 0.030 | 0.020 | 0.016 | 0.013 | 0.013 | 0.014 | 0.013 |

on-policy 一致性极好，策略漂移收敛。但这也可能是 EMA 教师补偿的结果。

#### ppl_ratio：剧烈波动

| step | 1 | 55 | 110 | 165 | 219 | 274 | 329 | 383 | 438 | 493 |
|------|---|---|-----|-----|-----|-----|-----|-----|-----|-----|
| ratio | 6304 | 649 | 7302 | 650 | 84968 | 339080 | 1.05 | 52201 | 9.7 | 65595 |

极不稳定。training_ppl 飙到 10 万+ 说明训练时模型对 rollout 样本的困惑度极高——某些 token 的 IS 权重异常大。但 loss 仍然收敛（EMA 掩盖了问题）。

#### response_length：230 → 503（持续增长）

学生在学习生成更长的摘要，clip_ratio 到 9.4% 说明部分回复已超 2048 被截断。

#### 诊断

表面最漂亮（loss 最低），但 topk_mass 双降 + ppl_ratio 剧烈波动 = **EMA 正反馈退化的典型特征**。loss 趋零是假象，不代表学生学到了更好的摘要能力。

---

### 3.2 v3-no-ema — 冻结 base 9B 教师 + base 9B 学生（242 步，进行中）

**配置要点**：student=base 9B，teacher=EMA(rate=0.0) 从 base 9B 初始化→冻结。框架不支持 `teacher_regularization="none"`，用 `ema + rate=0` 等价关 EMA。

#### vopd_loss：0.122 → 0.061（↓50%，plateau）

| step | 1 | 27 | 54 | 81 | 108 | 134 | 161 | 188 | 215 | 242 |
|------|---|----|----|----|-----|------|------|------|------|------|
| loss | 0.122 | 0.095 | 0.079 | 0.074 | 0.075 | 0.069 | 0.067 | 0.068 | 0.068 | 0.061 |

前 100 步快速下降，之后在 0.06-0.07 **明显 plateau**。这是没有 EMA 退化后的"真实"蒸馏难度。

#### 🔴 grad_norm：14.3 → 46.87（末步爆炸）

| step | 1 | 27 | 54 | 81 | 108 | 134 | 161 | 188 | 215 | 242 |
|------|---|----|----|----|-----|------|------|------|------|------|
| grad | 14.3 | 3.5 | 4.2 | 2.8 | 4.2 | 3.1 | 2.0 | 3.3 | 2.9 | **46.9** |

前 215 步稳定在 2-4，最后一步突然跳到 46.9。需要密切关注后续 step 是否持续发散。

#### student_on_teacher_topk_mass：0.986 → 0.976（稳定，无退化）

| step | 1 | 27 | 54 | 81 | 108 | 134 | 161 | 188 | 215 | 242 |
|------|---|----|----|----|-----|------|------|------|------|------|
| student | 0.986 | 0.988 | 0.983 | 0.980 | 0.981 | 0.980 | 0.975 | 0.976 | 0.977 | 0.976 |
| teacher | 0.989 | 0.990 | 0.988 | 0.987 | 0.989 | 0.988 | 0.985 | 0.986 | 0.987 | 0.986 |

教师冻结 → teacher_mass 稳定在 0.986，学生也稳定在 0.976 附近。**没有退化**，gap ~0.01 是真实蒸馏差距。

#### rollout_corr/kl：0.189 → 0.110

| step | 1 | 27 | 54 | 81 | 108 | 134 | 161 | 188 | 215 | 242 |
|------|---|----|----|----|-----|------|------|------|------|------|
| kl | 0.189 | 0.130 | 0.132 | 0.119 | 0.126 | 0.129 | 0.130 | 0.134 | 0.127 | 0.110 |

下降到 0.11 后 plateau，比 v3sft 的 0.013 高得多——因为没有 EMA 补偿，策略漂移更真实。

#### ✅ ppl_ratio：1.24 → 1.14（稳定接近 1）

| step | 1 | 27 | 54 | 81 | 108 | 134 | 161 | 188 | 215 | 242 |
|------|---|----|----|----|-----|------|------|------|------|------|
| ratio | 1.24 | 1.16 | 1.17 | 1.15 | 1.15 | 1.17 | 1.17 | 1.18 | 1.16 | 1.14 |

**四实验中最稳定**。IS 权重没有爆炸，on-policy 假设成立得最好。

#### mopd_bias_correction：0.003 → 0.009（缓慢上升）

偏差校正项在缓慢增长，说明师生概率质量差异在逐步增大，但仍在可控范围。

#### 诊断

最"诚实"的实验——loss plateau 在真实水平（0.061），topk_mass 稳定无退化，ppl_ratio 最健康。但最后一步 grad_norm 突跳到 46.9 需要关注后续是否发散。

---

### 3.3 v4-fixed-teacher — 冻结 SFT 教师 + SFT 学生（216 步）

**配置要点**：student=SFT ckpt，teacher=EMA(rate=0.0) 从 SFT 初始化→冻结。设计目的是同时拿到 SFT 起点优势 + 消除 EMA 退化。

#### vopd_loss：0.210 → 0.126（↓40%，震荡）

| step | 1 | 24 | 48 | 72 | 96 | 120 | 144 | 168 | 192 | 216 |
|------|---|----|----|----|----|-----|-----|-----|-----|-----|
| loss | 0.210 | 0.169 | 0.157 | 0.138 | 0.137 | 0.140 | 0.125 | 0.133 | 0.115 | 0.126 |

下降但明显震荡（0.137→0.140→0.125→0.133），plateau 在 0.12-0.13，比 v3-no-ema 高一倍。SFT 师生之间的分布更"尖锐"，蒸馏更难收敛。

#### grad_norm：28.4 → 17.08（持续偏高，震荡剧烈）

| step | 1 | 24 | 48 | 72 | 96 | 120 | 144 | 168 | 192 | 216 |
|------|---|----|----|----|----|-----|-----|-----|-----|-----|
| grad | 28.4 | 4.8 | 7.2 | 15.3 | 5.1 | 12.1 | 12.0 | 25.8 | 7.4 | 17.1 |

在 5-77 之间剧烈跳动，不稳定程度远超 v3-no-ema。SFT 起点的分布更尖锐，梯度更敏感。

#### student_on_teacher_topk_mass：0.994 → 0.985（稳定，无退化）

| step | 1 | 24 | 48 | 72 | 96 | 120 | 144 | 168 | 192 | 216 |
|------|---|----|----|----|----|-----|-----|-----|-----|-----|
| student | 0.994 | 0.994 | 0.994 | 0.992 | 0.990 | 0.987 | 0.988 | 0.983 | 0.984 | 0.985 |
| teacher | 0.997 | 0.996 | 0.996 | 0.996 | 0.995 | 0.994 | 0.995 | 0.994 | 0.994 | 0.995 |

教师冻结 → 两个 mass 稳定高位。无退化，但绝对值比 v3-no-ema 更高（0.985 vs 0.976），因为 SFT 模型本身分布更集中。

#### rollout_corr/kl：0.191 → 0.109

与 v3-no-ema 几乎一致，plateau 在 0.11。

#### ppl_ratio：剧烈波动

| step | 1 | 24 | 48 | 72 | 96 | 120 | 144 | 168 | 192 | 216 |
|------|---|----|----|----|----|-----|-----|-----|-----|-----|
| ratio | 26512 | 28376 | 6883 | 58776 | 129523 | 9185 | 6694 | 5195 | 21838 | 9670 |

和 v3sft 一样极不稳定。SFT 模型在某些 token 上极度自信，导致 IS 权重爆炸。

#### response_length：221 → 243（几乎不增长）

与 v3sft 的 230→503 形成鲜明对比——SFT 起点已经会写摘要，不需要"学会变长"。

#### 诊断

冻教师成功治了 EMA 退化（topk_mass 稳定），但 SFT 起点带来 grad_norm 震荡（15-77 范围）和 ppl_ratio 不稳。loss plateau 在 0.12，是真实蒸馏差距，但收敛不稳定。

---

### 3.4 v5-teacher27B — 外部固定 27B 教师 + base 9B 学生（67 步，延迟发散）

**配置要点**：student=base 9B，teacher=fixed Qwen3.8-27B（rate=0.0 永不更新）。27B 比 9B 强很多，提供更高质量 logit。架构兼容：同为 Qwen3_5ForConditionalGeneration，vocab/eos/image_token_id 一致。

#### 🔴 vopd_loss：0.495 → 0.225（前 45 步下降）→ 0.263（反弹上升）

| step | 1 | 10 | 25 | 45 | 50 | 55 | 59 | 65 | 67 |
|------|---|---|---|---|---|---|---|---|---|
| loss | 0.495 | 0.485 | 0.340 | **0.225** | 0.248 | 0.253 | 0.275 | 0.264 | 0.263 |

前 45 步健康下降（0.50→0.22），step 45 达到最低点 0.225 后**反弹**。这是延迟发散——先学后崩。

#### 🔴🔴 grad_norm：23.4 → 95.16（step 5 spike）→ 1.63（最终已收敛到小值）

| step | 1 | 5 | 15 | 25 | 45 | 55 | 65 | 67 |
|------|---|---|---|---|---|---|---|---|
| grad | 23.4 | **95.2** | 38.4 | 9.2 | 4.8 | 2.7 | 1.6 | 1.6 |

step 5 出现 95 的 spike（疑似 warmup 阶段不稳定），之后持续下降到 1.6。grad_norm 最终不大但 loss 反升——梯度虽小但方向错误。

#### 🔴🔴 teacher_topk_mass：0.986 → 0.807（冻结教师异常下降）

| step | 1 | 25 | 45 | 50 | 55 | 59 | 65 | 67 |
|------|---|---|---|---|---|---|---|---|
| t_mass | 0.986 | 0.987 | 0.986 | 0.984 | 0.961 | 0.913 | 0.894 | **0.807** |

**冻结教师权重不变，但 topk_mass 从 0.986 降到 0.807**。这是因为学生从 step 46 起急剧退化，生成异常 token，导致 27B 教师在这些位置分布变平坦。详见 Q11 情况二。

#### 🔴🔴 student_topk_mass：0.974 → 0.603（暴跌，40% 概率在教师支撑域外）

| step | 1 | 25 | 45 | 50 | 55 | 59 | 65 | 67 |
|------|---|---|---|---|---|---|---|---|
| s_mass | 0.974 | 0.970 | 0.975 | 0.954 | 0.902 | 0.773 | 0.742 | **0.603** |

学生只有 60% 概率质量在教师 top-100 内 → **40% 的概率跑到教师不会选的 token 上** → 支撑域外 mode collapse。

#### 🔴🔴 bias_correction：0.012 → 0.204（爆炸 17 倍，发散的真正信号）

| step | 1 | 25 | 45 | 50 | 55 | 59 | 65 | 67 |
|------|---|---|---|---|---|---|---|---|
| bias | 0.012 | 0.015 | 0.012 | 0.026 | 0.059 | 0.140 | 0.152 | **0.204** |

bias_correction = gap = teacher_topk_mass - student_topk_mass。step 45 起 gap 从 0.012 暴涨到 0.204，学生正在快速脱离教师支撑域。

#### mopd_reverse_kl_term：0.482 → 0.110（持续下降的假象）

| step | 1 | 25 | 45 | 55 | 65 | 67 |
|------|---|---|---|---|---|---|
| rkl | 0.482 | 0.323 | 0.206 | 0.193 | 0.110 | 0.110 |

reverse_kl_term 全程在降，看起来在收敛。但这是假象：学生把概率质量移到了教师 top-100 **之外**（student_topk_mass 从 0.974 降到 0.638），参与求和的概率变少了 → reverse_kl_term 自然变小，但学生没有变好 → bias_correction 暴露真相。详见 Q10。**vopd_loss = reverse_kl + bias_correction = 0.110 + 0.204 = 0.314**（实际 raw_jsd 为 0.263，因 IS 加权略有差异）。

#### 🔴 rollout_corr/kl：0.193 → 0.398（上升，on-policy 假设崩塌）

| step | 1 | 25 | 45 | 55 | 65 | 67 |
|------|---|---|---|---|---|---|
| kl | 0.193 | 0.323 | — | — | — | 0.398 |

策略漂移持续加剧，on-policy 假设越来越不成立。

#### 🔴 ppl_ratio：1.26 → 195（爆炸性增长）

training_ppl 从 6.5 飙到 465，IS 权重严重失稳。

#### 诊断

**延迟发散（delayed divergence）**。前 45 步健康收敛（loss 0.50→0.22），step 46 起急剧恶化。核心失败模式是 **支撑域外 mode collapse**：

1. 学生在前 45 步正常学习教师 top-100 内的 token 分布
2. step 45 后，学生开始在教师 top-100 **之外**的 token 上集中概率（概率移出求和范围 → reverse_kl_term 下降假象，详见 Q10）
3. 这些外部 mode 是错误的（教师不会选），bias_correction = Σ(p_t - p_s) 爆炸（0.012→0.204）
4. teacher_topk_mass 被动下降（冻结教师对学生异常输出的不确定性反映）
5. vopd_loss = reverse_kl + bias_correction 最终不降反升

**根因**：27B 教师和 9B 学生初始分布差异过大（JSD 起步 0.5 接近 ln2），反向 KL 在 top-k 截断支撑上梯度不稳定。前 45 步的下降是学生在"容易学"的部分快速对齐，但一旦开始触及师生分布差异大的区域，反向 KL 的 mode-seeking 特性导致学生跳到教师支撑域外。

**建议**：
1. 降学习率到 5e-7（当前 2e-6 对跨尺寸蒸馏过大）
2. 或换蒸馏目标为 forward KL / generalized JSD（alpha=0.5），对分布差异大的师生组合更稳定
3. 或先用 SFT warm-start 学生，缩小初始分布差距
4. 增大 warmup 到 200+ 步，给 IS 权重更多时间稳定
5. 监控 `bias_correction`：一旦超过 0.05 即应降学习率或回退 checkpoint

---

## 四、横向对比与总结

### 4.1 关键指标横向对比

| 指标 | v3sft (EMA) | v3-no-ema (冻base) | v4-fixed (冻SFT) | v5-27B |
|------|-------------|---------------------|-------------------|--------|
| vopd_loss 最终 | **0.007** | 0.061 | 0.126 | 0.263 |
| loss 收敛？ | 是（假象） | plateau（真实） | plateau（真实） | **先降后升（延迟发散）** |
| grad_norm 最终 | 1.2 | 46.9 | 17.1 | 1.6 |
| topk_mass 退化？ | **是，双降至 0.84** | 无 | 无 | **是，降至 0.81（被动）** |
| bias_correction | 稳定 0.003 | 稳定 0.010 | 稳定 0.010 | **爆炸 0.012→0.204** |
| rollout kl 最终 | 0.013 | 0.110 | 0.109 | 0.398 |
| ppl_ratio 稳定？ | 否（震荡） | **是 ≈1.1** | 否（震荡） | 否（爆炸） |
| EMA 退化？ | **是** | 否 | 否 | N/A |
| 整体诊断 | 假收敛 | 真实但末步不稳 | 真实但震荡 | 延迟发散（step 45 转折） |

### 4.2 核心结论

1. **v3sft 的"完美收敛"是假的**：EMA 教师跟随学生退化 → 师生同步变平 → 反向 KL 自然趋零。证据：topk_mass 从 0.99 双降至 0.84（gap 始终 0.003），ppl_ratio 剧烈波动（6304→65595）。这是设计 v4/v3-no-ema 实验要验证的假设——EMA 正反馈退化。

2. **v3-no-ema 是最"诚实"的实验**：ppl_ratio 稳定在 1.1、topk_mass 无退化、loss plateau 在 0.061 代表真实蒸馏差距。但最后一步 grad_norm=46.9 需要关注后续是否持续发散。这是当前最值得继续跑的实验。

3. **v4-fixed-teacher 证明冻教师能治退化**：topk_mass 稳定、无退化。但 SFT 起点导致 grad_norm 震荡（5-77）和 ppl_ratio 不稳（最高 12 万），可能需要降 LR 或增加梯度裁剪。

4. **v5-teacher27B 延迟发散**：前 45 步健康收敛（loss 0.50→0.22），step 46 起急剧恶化。核心失败模式是**支撑域外 mode collapse**——学生将 40% 概率质量移到教师 top-100 外（student_topk_mass 降至 0.603），bias_correction 爆炸（0.012→0.204）。reverse_kl_term 持续下降是假象（学生概率移出 top-k 求和范围导致总和变小，详见 Q10）。冻结教师 topk_mass 被动降至 0.807 反映学生输出退化导致教师不确定性增加。需要降 LR、换 FKL/JSD、或 warm-start。

### 4.3 TensorBoard 查看指南

TensorBoard 已在端口 6007 运行，指向 `tensorboard_log/RP-OPSD/` 父目录，可同时对比所有实验。

在 TensorBoard 左侧 tag 过滤框输入以下正则，只看重点指标：

```
^(actor/vopd_loss|actor/grad_norm|self_distillation/raw_jsd_token_mean|self_distillation/student_on_teacher_topk_mass_mean|self_distillation/teacher_topk_mass_mean|rollout_corr/kl)$
```

其余 tag（timing_*, perf_*, global_seqlen/*, rollout_is_*）为工程统计，无需关注。

---

## 五、FAQ

### Q1: OPD 的 Loss 一般用 Forward KL 还是 Reverse KL？

标准 OPD（On-Policy Distillation，在线策略蒸馏）默认使用 **Reverse KL**，即 KL(π_student ‖ π_teacher)。

本项目的四个实验均配置 `alpha=1.0` + `distillation_objective="mopd_topk_reverse_kl"`，走的就是纯反向 KL 分支（见 `core_algos.py:1133-1146`）。

#### 为什么 OPD 选择 Reverse KL？

**Mode-Seeking（模式寻求）特性**：Reverse KL 惩罚的是"学生模型认为概率高，但教师模型认为概率低"的 token。这会让学生模型主动避开教师不认可的输出，将概率质量集中在教师模型的高概率区域（即高质量答案）。

**符合偏好优化目标**：OPD 的核心目的是让模型"敢选高奖励答案"，而不是盲目覆盖教师分布的所有可能。在数学推理、代码生成等收敛性任务中，我们只需要学生聚焦到那几个正确的解法上，Reverse KL 的"挑剔聚焦"特性正好契合需求。

**避免无效探索**：如果使用 Forward KL，学生会倾向于"宁可覆盖宽，也不漏模式"，导致模型把概率分散到一些平庸或低质量的表达上，降低生成质量。

#### Forward KL 与 Reverse KL 的直观对比

| | Forward KL | Reverse KL |
|---|---|---|
| 公式 | KL(π_teacher ‖ π_student) | KL(π_student ‖ π_teacher) |
| 惩罚方向 | 教师"有"、学生"没"的模式 | 学生"有"、教师"没"的模式 |
| 行为特征 | 覆盖教师的全部分布（mean-searching） | 锁住教师高概率峰（mode-seeking） |
| 适合任务 | 发散性任务（创意写作、头脑风暴） | 收敛性任务（数学、代码、摘要） |
| 在 OPD 中的表现 | 易导致生成平庸、概率分散 | 聚焦高质量解法，是 OPD 主流选择 |

#### 为什么有些改进方案会引入 Forward KL？

虽然标准 OPD 用 Reverse KL，但在实际训练中发现它会导致两个问题：**多样性下降**（Mode Collapse）和**训练后期出现长度膨胀与重复**。为此，一些前沿研究提出了混合方案：

- **EOPD**（Entropy-Aware OPD）：当教师模型在某个 token 上**高熵**（不确定，存在多个合理下一步）时，引入 Forward KL 来保持多样性；在低熵时继续使用 Reverse KL 聚焦。
- **TrOPD**（Trust-Region OPD, arXiv 2606.01249）：在师生分布差异较大的 Outlier 区域，使用 Forward KL 进行模仿学习，防止学生从零开始生成低质量轨迹。同时在信任域内维持 Reverse KL 聚焦。

#### 与本项目四实验的关联

本项目四个实验全部使用纯 Reverse KL（alpha=1.0），这在 flash_note 摘要任务上是合理的——摘要属于收敛性任务，需要学生聚焦到教师的高质量摘要模式上。但 v5-teacher27B 的发散（JSD 起步 0.5，grad_norm 爆炸到 95）说明：**当师生分布差异过大时，纯 Reverse KL 在 top-k 截断支撑上梯度极度不稳定**。此时可考虑：

1. 降 LR 到 5e-7 减小梯度步长
2. 换成 Generalized JSD（alpha=0.5），前向 + 反向 KL 各半，兼顾覆盖与聚焦
3. 先用 SFT warm-start 学生缩小初始分布差距，再切回纯 Reverse KL
4. 参考 TrOPD 在 outlier 区域引入 Forward KL 模仿学习

### Q2: TensorBoard 里的 `raw_jsd_token_mean` 到底是 JSD 还是反向 KL？

**是反向 KL，不是 JSD**。命名含 `jsd` 是历史遗留——代码中 `alpha=1.0` 时走 `mopd_topk_reverse_kl` 分支（`core_algos.py:1133`），计算的原始 loss 是：

```
raw_per_token_loss = reverse_kl_term + bias_correction
                   = Σ p_s·(log p_s - log p_t) + Σ (p_t - p_s)
```

只有 `alpha` 不等于 0 或 1 时才走 Generalized JSD 分支（`core_algos.py:1155-1170`），计算 `kl_student` 和 `kl_teacher` 的线性插值。本项目四个实验 alpha 全为 1.0，所以 `raw_jsd_token_mean` 实际就是原始反向 KL loss。

### Q3: `vopd_loss` 和 `raw_jsd_token_mean` 有什么区别？

`vopd_loss` = `weighted_jsd_token_mean` = `raw_jsd_token_mean` × IS 权重（ratio）。

- `raw_jsd_token_mean`：IS 加权**前**的原始 loss，反映师生分布的真实差距
- `vopd_loss`：IS 加权**后**的 loss，是实际参与梯度反传的值

当 on-policy 假设成立（ratio ≈ 1.0）时两者几乎相等。v3sft 前期两者重合（ratio ≈ 1），后期 IS 权重波动大时会出现分离。v3-no-ema 的 ppl_ratio 稳定在 1.1，所以两者差距始终很小。

### Q4: `student_on_teacher_topk_mass` 和 `teacher_topk_mass` 两个值双双下降意味着什么？

这是 **EMA 正反馈退化**的标志性信号：

- `teacher_topk_mass` 下降 = 教师的 top-100 token 覆盖的概率质量在减少 = 教师分布在变平
- `student_on_teacher_topk_mass` 下降 = 学生落在教师 top-k 上的质量在减少
- 两者 **同步下降且 gap 不变** = 教师在跟随学生退化，师生始终保持相似

只有冻结教师（rate=0.0）的实验中 topk_mass 才会稳定（v3-no-ema 稳定在 0.976/0.986，v4 稳定在 0.985/0.995），因为教师不随学生更新，不会退化。

### Q5: `rollout_corr/ppl_ratio` 为什么有时飙到几万？

`ppl_ratio = training_ppl / rollout_ppl`。training_ppl 是训练步策略对 rollout 样本的困惑度，rollout_ppl 是 rollout 时的困惑度。

当某个 token 在 rollout 时概率极低（rollout_ppl 大）但在训练步更新后概率变得较高（training_ppl 小），ratio 就会极小；反过来如果训练步对某个 token 极度"意外"（training_ppl 飙到 10 万+），ratio 就会飙到几万。这通常发生在：

1. SFT 起点模型——分布尖锐，某些 token 概率极端
2. EMA 教师退化——师生同步变平，但个别 token 的 IS 权重异常
3. 师生尺寸差异大（v5）——分布根本不匹配，IS 权重爆炸

v3-no-ema 的 ppl_ratio 稳定在 1.1 是因为它用 base 9B + 冻结 base 9B 教师，分布最一致。

### Q6: `mopd_bias_correction_mean` 这个偏差校正项什么时候会变大？

`bias_correction = Σ (p_t - p_s)` on top-k。当教师概率质量远大于学生时（学生还没学会教师的高概率 token），bias_correction 为正且较大。随着学生学习靠近教师，此项应趋近 0。

但如果它持续增长（如 v4 从 0.003  Step→0.010），说明师生概率质量差异在扩大而非缩小——学生没有在有效学习教师的分布。这在 plateau 阶段是正常现象：loss 下降放缓，但偏差校正仍在缓慢增长。

### Q7: JSD、FKL、RKL 三者的区别和联系是什么？

JSD（Jensen-Shannon Divergence）、FKL（Forward KL）和 RKL（Reverse KL）都是衡量两个概率分布差异的指标，但在数学定义、优化行为和大模型训练应用上有本质区别。

#### 数学定义

设教师分布为 P，学生分布为 Q：

**FKL — KL(P‖Q)**：

$$FKL = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

以教师分布 P 为基准，惩罚学生分布 Q 未能覆盖 P 的区域。

**RKL — KL(Q‖P)**：

$$RKL = \sum_x Q(x) \log \frac{Q(x)}{P(x)}$$

以学生分布 Q 为基准，惩罚学生在教师 P 概率为 0 的区域分配了概率。

**JSD**：

$$M = \frac{P + Q}{2}, \quad JSD = \frac{1}{2} KL(P\|M) + \frac{1}{2} KL(Q\|M)$$

M 是 P 和 Q 的平均分布。JSD 本质上是对 FKL 和 RKL 的对称化折中。

#### 优化行为的本质区别

这是三者最核心的差异，决定了模型训练时的"性格"：

| | FKL | RKL | JSD |
|---|---|---|---|
| 别名 | Mass-Covering / Zero-Avoiding | Mode-Seeking / Zero-Forcing | 折中 |
| 行为 | 教师 P 有概率处，学生 Q 必须有概率，否则 FKL→∞ | 教师 P 概率为 0 处，学生 Q 强制为 0 | 通过平均分布 M 作为桥梁，兼顾覆盖与聚焦 |
| 结果 | 学生铺开覆盖教师所有模式，可能生成峰间"模糊地带" | 学生死盯教师最高概率峰，忽略次优模式 | 保持一定多样性的同时拟合高概率区域 |
| 风险 | 生成平庸化、概率分散 | 模式崩溃（Mode Collapse）、尾部重复 | 梯度更稳定（有界 ≤ log2），但聚焦力不如 RKL |
| 直觉 | "宁可错杀一千，不可放过一个" | "只选最对的，其他都不碰" | "中庸之道" |
| 对称性 | 不对称 | 不对称 | **对称**：JSD(P‖Q)=JSD(Q‖P) |
| 有界性 | 无界（→∞） | 无界（→∞） | **有界**（∈ [0, log2]） |

#### 三者的内在联系

**JSD 是 FKL 和 RKL 的对称化桥梁**：JSD 可看作 FKL 和 RKL 的平滑组合。数学上 JSD 上界为 log2（≈0.693），而 KL 散度无界。这使得 JSD 在优化时梯度更稳定，不易梯度爆炸——这也是为什么 v5-teacher27B 在纯 RKL 下发散（JSD 起步 0.5 接近 log2 上界，IS 权重爆炸），如果切换到 JSD 会更稳定。

**SFT 与 RL 的本质映射**：

| 训练阶段 | 本质散度 | 原因 |
|----------|----------|------|
| SFT（监督微调） | ≈ FKL | 等价于最大似然估计（MLE），让模型覆盖人类数据分布 |
| RL（PPO/GRPO）中的 KL 约束 | ≈ RKL | 限制策略不偏离参考模型太远，强制在安全区内活动 |
| OPD（在线蒸馏） | RKL 或 JSD | RKL 让学生聚焦教师高奖励模式；JSD 在聚焦与多样性间平衡 |

#### 在大模型训练中的典型应用场景

**FKL（前向 KL）**：
- **预训练**：学习海量语料分布，必须覆盖所有词汇和表达，不能漏掉高频词
- **VAE（变分自编码器）**：重构误差项近似 FKL，导致生成图像偏模糊（需覆盖所有可能细节）
- **SFT**：等价于 FKL/MLE，让模型覆盖人类标注数据的分布

**RKL（反向 KL）**：
- **RLHF / RLAIF**：奖励最大化时加入 RKL 约束，防止模型为刷高分而胡言乱语，强制在参考模型高概率区域内活动
- **OPD（在线蒸馏）**：让学生聚焦教师高概率、高奖励答案，适合数学推理、代码生成等需收敛到唯一正确解的任务
- **本项目**：四个实验全部使用 RKL（alpha=1.0, mopd_topk_reverse_kl）

**JSD（詹森-香农散度）**：
- **在线知识蒸馏（On-policy GKD）**：当教师在高熵 token 上存在多个合理下一步时，RKL 导致多样性崩塌，FKL 导致生成平庸，JSD 平衡质量与多样性
- **RGSD（基于评分量表的自蒸馏）**：在医疗/科学等开放式任务中，用 JSD 逐 token 蒸馏，抑制字数漂移和奖励黑客行为，同时保持事实正确率
- **Generalized JSD（本项目 alpha=0.5 时启用）**：`core_algos.py:1155-1170` 的 `kl_student` 和 `kl_teacher` 线性插值分支

#### 与本项目四实验的关联

本项目四个实验全部用纯 RKL（alpha=1.0），在 flash_note 摘要任务上是合理的——摘要属于收敛性任务。但各实验的表现差异揭示了 RKL 的适用边界：

| 实验 | RKL 表现 | 诊断 |
|------|----------|------|
| v3sft | loss→0.007 但 topk_mass 双降 | RKL 在 EMA 教师下"假收敛"——师生同步退化，RKL 趋零是假象 |
| v3-no-ema | loss plateau 0.061，ppl_ratio 稳定 | RKL + 冻结教师 = 最诚实的蒸馏信号 |
| v4-fixed | loss plateau 0.126，grad_norm 震荡 | RKL + SFT 尖锐分布 = 梯度不稳但无退化 |
| v5-teacher27B | loss 不降反升，grad_norm=95 | **RKL 在师生分布差异大时失效**——JSD 起步 0.5 接近 log2 上界，top-k 截断支撑上梯度爆炸 |

v5 的发散正是 RKL "Zero-Forcing" 特性的极端体现：27B 教师和 9B 学生分布差异过大，RKL 强制学生在教师概率为 0 的区域清零概率，但 top-k 截断后这些区域被排除，导致 bias_correction 项和 IS 权重同时爆炸。

**对 v5 的改进建议（基于散度选择）**：
1. 切换到 Generalized JSD（alpha=0.5）：有界梯度，不易爆炸，兼顾覆盖与聚焦
2. 或先 SFT warm-start 缩小师生初始分布差距，再切回纯 RKL
3. 参考 TrOPD（arXiv 2606.01249）：在 outlier 区域引入 FKL 模仿学习，信任域内维持 RKL

#### 总结

**FKL 追求"面面俱到"，RKL 追求"精准聚焦"，JSD 是两者的"中庸之道"**。在大模型后训练中，选择哪种散度本质上是在权衡"生成质量"与"多样性"：

- 需要收敛到唯一正确解（数学/代码/摘要）→ RKL
- 需要覆盖多种表达（预训练/创意写作）→ FKL
- 师生分布差异大、需防梯度爆炸、需平衡质量与多样性 → JSD

### Q8: `ppl_ratio` 的精确公式是什么？怎么理解？

> 源码位于 `verl/trainer/ppo/rollout_corr_helper.py:874-921`。

#### 逐步拆解

**第 1 步**：每个 token 有两个 log 概率值

```
log π_rollout(y_t) = rollout 时（写的时候）模型给 token y_t 的 log 概率
log π_train(y_t)  = train 时（看的时候）模型给同一个 token y_t 的 log 概率
```

**第 2 步**：逐序列取 token 平均

```python
mean_log_prob_rollout  = mean_t[ log π_rollout(y_t) ]   # 每条序列 rollout 时的平均 log 概率
mean_log_prob_training = mean_t[ log π_train(y_t)  ]   # 每条序列 train 时的平均 log 概率
```

**第 3 步**：转成困惑度 ppl

```python
training_ppl = exp(-mean_log_prob_training)   # 训练时的困惑度
rollout_ppl  = exp(-mean_log_prob_rollout)    # rollout 时的困惑度
```

> ppl = exp(-平均log概率)。模型自信（概率高）→ ppl 低；模型不自信 → ppl 高。

**第 4 步**：逐序列算比值

```python
log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
# 等价于 log(training_ppl / rollout_ppl)

ppl_ratio_i = exp(log_ppl_diff) = training_ppl_i / rollout_ppl_i
```

**第 5 步**：对 batch 内所有序列取算术平均

```python
ppl_ratio = mean_i[ exp(log_ppl_diff_i) ] = mean_i[ training_ppl_i / rollout_ppl_i ]
```

#### 完整公式

$$\text{ppl\_ratio} = \frac{1}{B} \sum_{i=1}^{B} \frac{\exp\!\left(-\frac{1}{|T_i|}\textstyle\sum_t \log \pi_{\text{train}}(y_t)\right)}{\exp\!\left(-\frac{1}{|T_i|}\textstyle\sum_t \log \pi_{\text{rollout}}(y_t)\right)}$$

#### 具体数字示例

假设一条回复有 2 个 token：

| | token A | token B | 平均 log_prob | ppl = exp(-avg) |
|---|---|---|---|---|
| **rollout 时**（写） | log π = -1（π=0.37） | log π = -2（π=0.14） | -1.5 | exp(1.5) = **4.48** |
| **train 时**（看） | log π = -3（π=0.05） | log π = -5（π=0.007） | -4.0 | exp(4.0) = **54.6** |

```
ppl_ratio = 54.6 / 4.48 = 12.2
```

训练时模型对自己写的东西的困惑度是写时的 12 倍 → 模型变化很大 → off-policy 严重。如果 rollout 和 train 时的 log_prob 一样 → `ppl_ratio = exp(0) = 1.0` → 完美 on-policy。

#### 关键细节：先算比值再平均（对极端值敏感）

```python
# 代码实际做的是：
ppl_ratio = mean_i[ exp(log_ppl_diff_i) ]    # 先对每条序列算 exp，再平均（算术平均）

# 不是：
ppl_ratio = exp(mean_i[log_ppl_diff_i])     # 先平均 log，再 exp（几何平均）
# 也不是：
ppl_ratio = mean(training_ppl) / mean(rollout_ppl)  # 先平均 ppl 再除
```

这导致 `ppl_ratio` 对极端值非常敏感——如果 batch 里 32 条序列中有 1 条的 training_ppl/rollout_ppl = 100 万，即使其他 31 条都是 1.0，`ppl_ratio` 也会被拉到 ~31000。这就是为什么 v3sft/v4 的 ppl_ratio 经常飙到几十万——不是所有 token 都疯了，而是 batch 里有极个别序列的比值爆炸，拉高了算术平均值。

#### ppl_ratio 的含义与影响

| ppl_ratio | 含义 | on-policy 状态 |
|-----------|------|----------------|
| ≈ 1.0 | 训练时和写时困惑度一样 | ✅ 完美 on-policy |
| >> 1 | 训练时比写时更困惑 | ❌ 学生变差了（忘记了自己写的东西） |
| << 1 | 训练时比写时更自信 | ⚠️ 学生变好了（但变化太大仍有 IS 失稳风险） |

**在本项目中的实际影响**：虽然 ppl_ratio 波动大，但 IS 权重被 `clamp(max=2.0)` 截断，数据显示所有 `rollout_is_*` 指标恒为 1.0——IS 权重没有实际生效。所以 ppl_ratio 是**仪表盘上的警告灯**（告诉你训练是否 on-policy），不是刹车（不直接影响 loss 数值）。

### Q9: ppl_ratio 飙到几十万，到底是多少步梯度更新导致的？

#### 配置事实

| 参数 | 值 | 来源 |
|------|-----|------|
| `ppo_epochs` | **1** | 默认值（`actor.yaml:225`），脚本未覆盖写 |
| `train_batch_size` | **96** | `run_rp_opsd_v3sft.sh:40` |
| `ppo_mini_batch_size` | **96** | `run_rp_opsd_v3sft.sh:41`，与 train_batch 相同 |
| `LR` | **2e-6** | `run_rp_opsd_v3sft.sh:52` |
| `teacher_update_rate` | **0.05**（v3sft）| EMA 教师每步跟踪学生 5% |

因为 `ppo_epochs=1` 且 `train_batch == mini_batch`，所以：

```
1 次 rollout（96 条）→ 1 次梯度更新 → 记录 1 次 ppl_ratio
```

**不是多步累积导致的**。ppl_ratio 记录的是梯度更新**之前**的两个概率值（见下文），学生权重在两个时刻完全相同。

#### 那为什么 ppl_ratio 飙到 90 万？

ppl_ratio 比的是 `old_log_prob`（FSDP FP32 算的）vs `rollout_log_prob`（vLLM BF16 算的），两者用的是**同一份学生权重**。差异来自：

```
vLLM 推理引擎 (BF16)  vs  FSDP 训练框架 (FP32)
```

**根因：SFT 模型的 logit 分布极端尖锐。**

| 模型类型 | 分布特征 | BF16 vs FP32 差异 | ppl_ratio |
|----------|----------|-------------------|-----------|
| **base 9B**（v3-no-ema） | 概率分布平滑，无近零 token | 微小（同数量级内） | ✅ 1.1 稳定 |
| **SFT 9B**（v3sft, v4） | 概率分布极端尖锐，大量 token 概率 ≈ 1e-10 | **爆炸**：BF16 下 1e-10 可能变成 1e-20 或 0，FP32 下保持 1e-10 → 比值 = 1e10 | 🔴 几万~百万 |

**具体机制**：SFT 模型对某些 token 极度自信（概率 0.99+），对其他 token 极度不自信（概率 ≈ 0）。BF16 的精度范围有限（~7 位有效数字），无法精确表示这些近零值：

```
FP32: p(token_x) = 1e-10  → log_p = -23.0
BF16: p(token_x) ≈ 5e-8   → log_p = -16.8     (精度丢失，概率被放大了 500 倍)

→ ppl_ratio 贡献 = exp(-16.8) / exp(-23.0) = 500 倍
→ 一条序列有多个这样的 token → ppl_ratio 飙到几万
```

v3-no-ema 用 base 模型，分布平滑，没有近零概率 → BF16/FP32 差异极小 → ppl_ratio 稳定 1.1。

#### EMA 教师更新对 ppl_ratio 的影响

EMA 教师更新（rate=0.05）不直接改变学生权重，但会间接影响：

1. EMA 教师每步跟踪学生 5% → 教师分布变化 → 蒸馏目标变化
2. 蒸馏梯度方向变化 → 学生梯度更新幅度可能更大
3. 但这影响的是**下一步**的 ppl_ratio，不是当前步的

所以 v3sft ppl_ratio 飙到 90 万**主要不是梯度更新导致的**，而是 SFT 模型在 BF16/FP32 精度差异下的数值爆炸。

### Q10: `mopd_reverse_kl_term` 的精确公式是什么？为什么说它会"骗人"？

> 源码：`core_algos.py:1144`

#### 公式

```python
reverse_kl_term = student_probs * (student_logp_fp32 - teacher_logp_fp32)
```

逐 token 展开：

$$\text{reverse\_kl\_term}(y_t) = \sum_{v \in \text{top-k}} p_s(v) \cdot \left[\log p_s(v) - \log p_t(v)\right] = \sum_{v \in \text{top-k}} p_s(v) \cdot \log\frac{p_s(v)}{p_t(v)}$$

这就是反向 KL 散度 **KL(p_学生 ‖ p_教师)**，但**只在教师 top-100 token 上求和**。

#### 与标准反向 KL 的关键区别

标准反向 KL 对**全部词表**求和：

$$\text{KL}_{\text{full}}(p_s \| p_t) = \sum_{\text{all } v} p_s(v) \cdot \log\frac{p_s(v)}{p_t(v)}$$

代码里只对**教师 top-100** 求和：

$$\text{reverse\_kl\_term} = \sum_{v \in \text{top-100}} p_s(v) \cdot \log\frac{p_s(v)}{p_t(v)}$$

**top-100 之外的 token 不参与计算**。这导致一个致命的"骗人"效应：

#### 为什么 reverse_kl_term 持续下降却是在发散（v5 实例）

| step | student_topk_mass | reverse_kl_term | bias_correction | 实际状态 |
|------|-------------------|-----------------|-----------------|----------|
| 1 | 0.974 | 0.470 | 0.012 | 正常 |
| 45 | 0.975 | 0.206 | 0.012 | 正常收敛 |
| 53 | 0.925 | 0.213 | 0.045 | 开始发散 |
| 65 | 0.638 | **0.062** ↓↓ | **0.199** ↑↑↑ | 严重发散 |

**reverse_kl_term 在下降**（0.47→0.06），看起来在收敛，但实际在发散。原因：

```
reverse_kl_term = Σ_{v ∈ top-100} p_s(v) × log[p_s(v)/p_t(v)]

当学生把概率从 top-100 内移到 top-100 外时：
  · p_s(v) 在 top-100 内变小 → 每一项都被乘以更小的 p_s → 总和变小
  · 但 log[p_s/p_t] 可能变大（学生更不像教师了）
  · 然而乘以变小的 p_s 后，总和还是变小

→ reverse_kl_term 下降 = 假象！
→ 学生不是在靠近教师，而是把概率质量"逃"到了求和范围之外
```

**类比**：考试只批前 100 题的分数。学生把答案从"前 100 题的空"移到了"第 101-200 题"。前 100 题的扣分减少了（看起来成绩变好），但实际总分更差了。`bias_correction` 就是第 101 题之后的扣分——它暴露了真相。

#### 为什么之前说"学生自身熵在降"——这个说法不够准确

reverse_kl_term 可以拆成两部分：

$$\text{reverse\_kl\_term} = \underbrace{\sum_{v \in \text{top-k}} p_s(v) \log p_s(v)}_{-H_{\text{top-k}}(p_s)} - \underbrace{\sum_{v \in \text{top-k}} p_s(v) \log p_t(v)}_{\text{cross-entropy}}$$

- 第一项 = -H(p_s) 在 top-k 上的负熵（学生越尖锐，熵越低，负熵越高 → 这项增大）
- 第二项 = 交叉熵（学生与教师越不像，这项越大）

**单纯"熵在降"不能解释 reverse_kl 下降**。真正的机制更简单：

> 学生在 top-100 上的总概率质量 `Σ p_s(v)` 从 0.974 降到 0.638。每项 `p_s(v) × log(p_s/p_t)` 被乘以更小的 `p_s(v)`，所以总和变小。**不是因为学生变好了，而是因为参与求和的概率变少了。**

#### 诊断口诀

```
reverse_kl_term 下降 + bias_correction 稳定 → 真收敛
reverse_kl_term 下降 + bias_correction 上升   → 假收敛（支撑域外 mode collapse）
reverse_kl_term 上升                         → 学生在 top-k 内远离教师
```

**永远不要单独看 reverse_kl_term，必须配 bias_correction 一起看。**

### Q11: `teacher_topk_mass_mean` 下降意味着什么？冻结教师为什么也会下降？

`teacher_topk_mass_mean` = 教师在自身选出的 top-100 token 上的概率质量总和 `Σ p_t(v)`，衡量教师分布的"锐度"。下降意味着教师对这些 token 的置信度降低、分布变平坦。

**关键区分：下降原因取决于教师是否可更新。**

#### 情况一：EMA 教师（v3sft）— 教师自身在退化

v3sft 用 EMA(rate=0.05) 教师，教师权重跟踪学生。当学生退化（生成质量下降）时：
- EMA 教师跟随学生退化 → 教师自身分布变差 → `teacher_topk_mass` 下降
- `student_topk_mass` 同步下降 → 两者**同速下降、gap 保持 0.003-0.004**
- 这是**假收敛**：loss 下降不是因为师生对齐，而是因为师生**一起退化**导致分布趋同

#### 情况二：冻结教师（v5-27B）— 学生输出退化的**被动反映**

v5 用冻结的 27B 外部教师（rate=0.0，权重不更新）。教师权重不变，`teacher_topk_mass` 为什么还下降？

因为 `topk_source=teacher` 意味着：**对每个 token 位置，教师根据自己的上下文选 top-100**。当输入上下文（学生的 rollout 回复）变化时，教师的预测分布也变。学生退化 → 生成越来越多的"异常 token" → 教师在这些异常 token 位置上更不确定（分布变平坦）→ `teacher_topk_mass` 被动下降。

**v5-27B 完整曲线揭示的延迟发散（67 步）：**

| 阶段 | step | teacher_topk_mass | student_topk_mass | bias_correction | vopd_loss |
|------|------|-------------------|-------------------|-----------------|-----------|
| 健康收敛 | 1 | 0.986 | 0.974 | 0.012 | 0.484 |
| | 25 | 0.987 | 0.970 | 0.015 | 0.340 |
| 最优点 | 45 | 0.986 | 0.975 | 0.012 | **0.225** |
| 转折点 | 46 | 0.983 | 0.968 | 0.016 | 0.239 |
| 急剧下降 | 53 | 0.970 | 0.925 | 0.045 | 0.260 |
| | 59 | 0.913 | 0.773 | 0.140 | 0.275 |
| 发散末段 | 65 | 0.894 | 0.742 | 0.152 | 0.264 |
| 最新 | 67 | **0.807** | **0.603** | **0.204** | 0.263 |

**v5 的核心失败模式 — `bias_correction` 爆炸：**

- `mopd_reverse_kl_term` **持续下降**（0.21→0.11）：表面看"收敛"
- `mopd_bias_correction` **爆炸增长**（0.012→0.204，17 倍）：师生概率质量 gap 急剧扩大
- `vopd_loss = reverse_kl + bias_correction`：从 step 45 起不降反升（0.225→0.264）

**根因**：学生发生了**教师支撑域外 mode collapse**。学生将 40% 概率质量移到了教师 top-100 token **之外**的位置（student_topk_mass=0.603 → 39.7% 在外面）。因为 reverse_kl_term 只在 top-k 上求和，学生概率移出后参与求和的质量变少 → reverse_kl_term 下降（假象，详见 Q10），但这些 mode 是错误的（教师不会生成 → bias_correction 爆炸）。

**v3sft vs v5-27B 下降对比：**

| | v3sft (EMA 教师) | v5-27B (冻结教师) |
|---|---|---|
| 教师权重 | 跟随学生退化 | 不变（冻结） |
| teacher_topk_mass 下降原因 | 教师自身退化 | 学生输出退化 → 教师被动不确定 |
| gap (teacher - student) | 恒定 0.003-0.004 | 爆炸 0.012→0.204 |
| bias_correction | 稳定 0.003-0.007 | 爆炸 0.012→0.204 |
| reverse_kl_term | 持续下降 | 持续下降（假象） |
| 实际状态 | 假收敛 | 延迟发散 |
| 诊断信号 | teacher/student 双降 + gap 不变 | teacher 降 + student 降更快 + gap 扩大 |

**判别口诀**：看 `bias_correction`（= gap = teacher_topk_mass - student_topk_mass）：
- gap **稳定且小**（< 0.01）→ 师生同步，可能假收敛 → 检查教师是否 EMA
- gap **持续扩大**（> 0.05）→ 学生脱离教师支撑域 → 真发散

---

### Q12: 为什么 9B←9B 正常，27B→9B 就发散？

这是四个实验对比中最核心的问题。v3-no-ema（9B 冻结教师 + 9B 学生）全程稳定收敛，v5-27B（27B 冻结教师 + 9B 学生）step 46 起延迟发散。两者唯一变量是**教师尺寸**。以下从五个层面逐层拆解根因，并与指标日志逐条对应。

#### 层面一：初始分布差距——4 倍鸿沟

| 指标 | v3-no-ema (9B←9B) | v5-27B (27B→9B) | 倍数 |
|------|-------------------|-----------------|------|
| `raw_jsd` step 1 | **0.122** | **0.494** | **4×** |
| ln2 上界参考 | 0.693 | 0.693 | — |
| 初始 gap 占 ln2 比例 | 17.6% | **71.2%** | — |

- **9B←9B**：师生是**同一份 checkpoint**（base 9B）。raw_jsd 不为 0 仅仅因为 BF16 vs FP32 精度差 + vLLM vs FSDP 推理路径差，但学到的分布本质相同。0.122 是"精度噪声"，不是"能力差距"。
- **27B→9B**：师生是**不同模型**（Qwen3.8-27B vs Qwen3.5-9B），同族同词表但参数量差 3 倍。raw_jsd 起步 0.494，已逼近 ln2（0.693）上界——两者分布**接近最大程度不同**。

> **指标对应**：v5 的 `raw_jsd` 起步 0.494 对应 §3.4 记录的 vopd_loss 0.495；v3-no-ema 起步 0.122 对应 vopd_loss 0.122。这 4 倍差距是后续一切分化的种子。

#### 层面二：top-k 支撑域重叠——假象与真相

step 1 时两者的 `student_on_teacher_topk_mass` 看起来差不多：

| step 1 | v3-no-ema | v5-27B |
|--------|-----------|--------|
| student_topk_mass | 0.986 | 0.974 |
| teacher_topk_mass | 0.989 | 0.986 |

**但这 0.986 和 0.974 的含义截然不同：**

- **9B←9B**：教师 top-100 = 学生自然会选的 100 个 token（同一模型）。学生 98.6% 的概率质量本来就在这些 token 上 → 训练只需**微调权重**让分布更尖锐 → 稳定收敛。
- **27B→9B**：27B 教师的 top-100 包含 9B 学生"不自然倾向"的 token（27B 有 3 倍容量，学到了不同的 token 偏好）。学生 97.4% 的质量在教师 top-k 上是**表面巧合**——一旦反向 KL 开始推学生向 27B 的 mode 靠拢，学生被迫离开自己的自然分布，概率开始泄漏到 top-k 之外。

> **指标对应**：v3-no-ema 的 student_topk_mass 全程稳定在 0.976~0.988（§3.2）；v5 从 0.974 暴跌到 0.603（§3.4）——**40% 概率质量逃出教师支撑域**。

#### 层面三：反向 KL 的 mode-seeking 特性——同源时是利器，跨尺寸时是毒药

反向 KL `D_KL(p_s||p_t) = Σ p_s(v)·log(p_s(v)/p_t(v))` 是 **mode-seeking**（seeking-whole-support）散度：

- **9B←9B 时**：教师的 mode = 学生的 mode（同分布）。反向 KL 只是在让学生**更尖锐**地集中在自己已有的 mode 上 → 安全收敛。`reverse_kl_term` 下降是真实的。
- **27B→9B 时**：教师的 mode ≠ 学生的自然 mode（不同模型）。反向 KL 强行把学生推向 27B 的 mode，但 9B 的参数空间**没有能力表示 27B 的分布**（3 倍容量差距）→ 学生无法精确匹配 → 创造**退化解**：把极端概率堆到少数 token 上（mode collapse）→ 生成的摘要变成短语循环（word_rep 9.8%），最终退化为乱码（char_rep 6.1%、gibberish 4.3%）。

> **指标对应**：v5 的 `mopd_reverse_kl_term` 全程下降（0.482→0.110），看似收敛，但这是假象——学生概率移出 top-k 求和范围后参与求和的质量变少，详见 Q10。`bias_correction` 从 0.012 爆炸到 0.204（17 倍）才暴露真相。

#### 层面四：top-k 截断——假收敛的温床

MOPD loss 只在教师 top-100 上求和（`DISTILLATION_TOPK=100`）：

```
raw_per_token_loss = reverse_kl_term + bias_correction
reverse_kl_term = Σ_{v∈top-100} p_s(v)·log(p_s/p_t)   ← 只看 top-k 内
bias_correction = Σ_{v∈top-100} (p_t(v) - p_s(v))     ← 暴露 top-k 内外的质量差
```

- **9B←9B**：学生概率始终在教师 top-k 内（同分布，top-k 重叠天然高）→ `reverse_kl_term` 和 `bias_correction` 都诚实反映蒸馏进度 → 前者降、后者稳定 0.003~0.009。
- **27B→9B**：学生被反向 KL 推到教师 top-100 **之外**的 token 上。这些外部概率**不参与 reverse_kl_term 求和** → 该项虚降（假收敛）；但 `bias_correction = Σ(p_t - p_s)` 会暴露：学生 mass 从 top-k 流出 → p_s 在 top-k 内变小 → gap 扩大 → bias_correction 爆炸。

> **指标对应**：v5 `reverse_kl_term` 0.482→0.110（假降）+ `bias_correction` 0.012→0.204（真涨）= vopd_loss 最终 0.225→0.263 反弹。v3-no-ema 没有这个分裂：reverse_kl 和 bias_correction 同步下降，loss 单调收敛到 0.061。

#### 层面五：on-policy 假设崩塌——rollout 侧的连锁反应

学生退化 → rollout 生成质量下降 → on-policy 采样分布与训练分布脱节：

| 指标 | v3-no-ema | v5-27B | 含义 |
|------|-----------|--------|------|
| `ppl_ratio` | 1.14（稳定 ≈ 1） | **195**（爆炸） | 学生 train/rollout 分布严重分裂 |
| `rollout_ppl` | 5.7（稳定） | **915**（爆炸） | rollout 生成低概率无意义 token |
| `training_ppl` | 6.6（稳定） | **465** | 训练侧也跟着崩 |
| `rollout_corr/kl` | 0.189→0.110（↓） | 0.193→0.398（↑） | v3 策略漂移缩小，v5 持续扩大 |
| `word_rep`（rollout） | < 2% | **9.8%**（step 31） | 短语循环 mode collapse |
| `char_rep`（rollout） | < 0.4% | **6.1%**（step 69） | 字符级崩塌 |
| `gibberish`（rollout） | < 0.1% | **4.3%**（step 69） | 乱码接管 |

- **9B←9B**：师生同分布 → rollout 生成正常 → `ppl_ratio ≈ 1` → IS 权重稳定 → on-policy 假设成立 → 训练-推理闭环正反馈。
- **27B→9B**：学生退化 → rollout 生成短语循环/乱码 → `rollout_ppl` 从 5 飙到 915 → `ppl_ratio` 爆炸到 195 → IS 权重失稳 → 高 ppl_ratio 样本被错误放大 → 进一步推偏学生 → **恶性正反馈**。

#### 完整因果链

```
27B ≠ 9B（参数差 3 倍）
  ↓
初始 raw_jsd = 0.494（接近 ln2 上界）    ← 层面一
  ↓
师生 top-k 表面重叠但本质不同              ← 层面二
  ↓
反向 KL mode-seeking 推学生离开自然分布     ← 层面三
+ top-k 截断让泄漏概率不被 reverse_kl 惩罚  ← 层面四
  ↓
reverse_kl 假降 + bias_correction 真涨      ← Q10/Q11 机制
  ↓
学生 mode collapse → rollout 退化
  ↓
rollout_ppl 爆炸 → ppl_ratio 爆炸          ← 层面五
→ IS 权重失稳 → 恶性正反馈
  ↓
三阶段文本退化：健康 → 短语循环(9.8%) → 乱码(4.3%)
```

**而 9B←9B 从起点就阻断了这个链路**：初始 jst 0.122（精度噪声非能力差距）→ top-k 本质重叠 → 反向 KL 只做尖锐化不做人挪 → bias_correction 稳定 → rollout 正常 → ppl_ratio ≈ 1 → 闭环正反馈。

#### 修复方向（已在 §3.4 建议，此处汇总）

| 方案 | 针对的层面 | 原理 |
|------|-----------|------|
| **降 LR 到 5e-7** | 层面三 | 减缓反向 KL 的推力，给学生更多时间在自然分布附近微调而非跳出去 |
| **换 FKL / JSD (alpha=0.5)** | 层面三 | 前向 KL 是 mean-seeking（覆盖全部 mode），不会把学生推到单一 mode 上爆炸 |
| **SFT warm-start 学生** | 层面一 | 缩小初始 raw_jsd（SFT 后的 9B 离 27B 更近，v4 的 raw_jsd 起步更低） |
| **增大 warmup 200+ 步** | 层面五 | 给 IS 权重更多时间稳定，防止 ppl_ratio 早期爆炸触发恶性正反馈 |
| **监控 bias_correction > 0.05** | 全局 | bias_correction 是最早暴露发散的信号，比 loss 反弹更早（step 45 时 bias 已升，loss step 46 才反弹） |

> **核心教训**：跨尺寸蒸馏（27B→9B）不能用纯反向 KL + top-k 截断。初始分布差距过大时，反向 KL 的 mode-seeking 特性会逼学生跳出教师支撑域，top-k 截断又让这个错误在 reverse_kl_term 上不可见（假收敛），直到 bias_correction 爆炸和 rollout 文本退化才暴露。同尺寸蒸馏（9B←9B）没有这个问题，因为师生同分布，反向 KL 只做尖锐化不做模式转移。

---

## 六、TensorBoard 全部指标释义与四实验数据

> 本节覆盖全部 107 个 scalar tag，按前缀分组。每条包含：含义、四实验的 first/last/min/max 统计、趋势点评。
> 数据截止 2026-09-03，v3sft=493 步，v3-no-ema=273 步，v4=216 步，v5=67 步。

### 6.1 actor/* — 学生模型训练指标（8 tags）

| Tag | 含义 |
|-----|------|
| `actor/vopd_loss` | **实际优化的 VOPD loss**，= raw_loss × IS 权重，取 valid token 均值 |
| `actor/vopd_loss_weighted` | 与 vopd_loss 等价（IS 加权后的 loss） |
| `actor/pg_loss` | policy gradient loss，在 vopd 模式下与 vopd_loss 相同 |
| `actor/grad_norm` | 梯度范数，训练稳定性指标 |
| `actor/kl_loss` | KL 约束 loss，配置 `use_kl_loss=False` 所以恒为 0 |
| `actor/grpo_loss` | GRPO 策略梯度 loss，在纯蒸馏模式下不用，恒为 0 |
| `actor/lr` | 当前学习率，恒为 0.0（日志记录的峰值/瞬时值，实际由调度器管理） |
| `actor/policy_fallback_fraction` | 策略回退比例（蒸馏不可用时回退到 GRPO），恒为 0 |

#### 四实验数据

| Tag | 实验 | first | last | min | max |
|-----|------|-------|------|-----|-----|
| **vopd_loss** | v3sft | 0.2142 | 0.0068 | 0.0058 | 0.2142 |
| | v3-no-ema | 0.1219 | 0.0571 | 0.0566 | 0.1219 |
| | v4-fixed | 0.2099 | 0.1256 | 0.1084 | 0.2099 |
| | v5-27B | 0.4952 | 0.2634 | 0.2246 | 0.5060 |
| **grad_norm** | v3sft | 11.35 | 1.20 | 0.32 | 171.33 |
| | v3-no-ema | 14.29 | 6.98 | 1.53 | 46.87 |
| | v4-fixed | 28.38 | 17.08 | 2.97 | 76.89 |
| | v5-27B | 23.43 | 1.63 | 1.56 | 95.16 |
| **pg_loss** | v3sft | 0.2142 | 0.0068 | 0.0058 | 0.2142 |
| | v3-no-ema | 0.1219 | 0.0571 | 0.0566 | 0.1219 |
| | v4-fixed | 0.2099 | 0.1256 | 0.1084 | 0.2099 |
| | v5-27B | 0.4952 | 0.2634 | 0.2246 | 0.5060 |
| **kl_loss** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |
| **grpo_loss** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |
| **lr** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |
| **policy_fallback** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |

**趋势点评**：
- vopd_loss/pg_loss/vopd_loss_weighted 三个值在所有实验中完全一致——vopd 模式下它们是同一个 loss 的不同别名
- grad_norm：v3sft 峰值 171（step 219 spike），v5-27B 峰值 95（发散），v4 峰值 77（震荡），v3-no-ema 峰值 47（末步突跳）
- kl_loss/grpo_loss/lr/policy_fallback 恒为 0，无需关注——配置上关闭了 KL 约束和 GRPO 策略梯度

---

### 6.2 self_distillation/* — 蒸馏核心指标（17 tags）

| Tag | 含义 |
|-----|------|
| `raw_jsd_token_mean` | IS 加权前的原始 loss（实际是反向 KL，非 JSD），= reverse_kl_term + bias_correction |
| `raw_distillation_token_mean` | 同 raw_jsd_token_mean（别名） |
| `weighted_jsd_token_mean` | IS 加权后的 loss，与 vopd_loss 等价 |
| `mopd_reverse_kl_term_mean` | 反向 KL 主项 = Σ p_s·(log p_s - log p_t)，逐 token 均值 |
| `mopd_bias_correction_mean` | 偏差校正项 = Σ (p_t - p_s)，逐 token 均值 |
| `student_on_teacher_topk_mass_mean` | 学生概率落在教师 top-k 上的质量 = Σ p_s(v) |
| `teacher_topk_mass_mean` | 教师自身 top-k 概率质量 = Σ p_t(v)，参考值 |
| `num_distill_tokens` | 蒸馏 mask 覆盖的有效 token 数（batch 汇总） |
| `self_distillation_mask.mean()` | 蒸馏 mask 的平均覆盖率，1.0 = 全部 token 参与蒸馏 |
| `teacher_always_on_fraction` | 教师始终开启的比例，1.0 = 每个 token 都有教师 logit |
| `teacher_image_swap_fraction` | 教师图像替换比例，1.0 = 教师始终看特权图像（bbox 裁剪图） |
| `empty_target_batch` | 空目标 batch 数，0 = 无异常 |
| `grpo_fallback_count` | 回退到 GRPO 的次数，0 = 蒸馏全程生效 |
| `policy_fallback_fraction` | 蒸馏策略回退比例，0 = 无回退 |

#### 四实验数据

| Tag | 实验 | first | last | min | max |
|-----|------|-------|------|-----|-----|
| **raw_jsd_token_mean** | v3sft | 0.2142 | 0.0065 | 0.0055 | 0.2142 |
| | v3-no-ema | 0.1223 | 0.0569 | 0.0562 | 0.1223 |
| | v4-fixed | 0.2099 | 0.1246 | 0.1082 | 0.2099 |
| | v5-27B | 0.4935 | 0.2634 | 0.2179 | 0.5051 |
| **mopd_reverse_kl_term_mean** | v3sft | 0.2113 | 0.0033 | 0.0017 | 0.2113 |
| | v3-no-ema | 0.1188 | 0.0465 | 0.0455 | 0.1188 |
| | v4-fixed | 0.2071 | 0.1148 | 0.1003 | 0.2071 |
| | v5-27B | 0.4816 | 0.1102 | 0.1102 | 0.4920 |
| **mopd_bias_correction_mean** | v3sft | 0.0030 | 0.0032 | 0.0016 | 0.0068 |
| | v3-no-ema | 0.0034 | 0.0104 | 0.0018 | 0.0133 |
| | v4-fixed | 0.0028 | 0.0097 | 0.0018 | 0.0125 |
| | v5-27B | 0.0119 | 0.2035 | 0.0070 | 0.2035 |
| **student_on_teacher_topk_mass** | v3sft | 0.9941 | 0.8414 | 0.8333 | 0.9954 |
| | v3-no-ema | 0.9860 | 0.9758 | 0.9693 | 0.9893 |
| | v4-fixed | 0.9939 | 0.9849 | 0.9802 | 0.9946 |
| | v5-27B | 0.9756 | 0.6034 | 0.6034 | 0.9829 |
| **teacher_topk_mass** | v3sft | 0.9971 | 0.8446 | 0.8372 | 0.9973 |
| | v3-no-ema | 0.9894 | 0.9862 | 0.9826 | 0.9915 |
| | v4-fixed | 0.9967 | 0.9946 | 0.9921 | 0.9968 |
| | v5-27B | 0.9861 | 0.8069 | 0.8069 | 0.9899 |
| **num_distill_tokens** | v3sft | 1378 | 2194 | 1234 | 2513 |
| | v3-no-ema | 1532 | 1637 | 1471 | 1799 |
| | v4-fixed | 1329 | 1459 | 1245 | 1532 |
| | v5-27B | 1524 | 1667 | 1343 | 1774 |
| **mask.mean()** | 全部 | 1.0 | 1.0 | 1.0 | 1.0 |
| **teacher_always_on** | 全部 | 1.0 | 1.0 | 1.0 | 1.0 |
| **teacher_image_swap** | 全部 | 1.0 | 1.0 | 1.0 | 1.0 |
| **empty_target_batch** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |
| **grpo_fallback** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |
| **policy_fallback** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |

**趋势点评**：
- **raw_jsd = reverse_kl_term + bias_correction**：可验证，如 v5-27B last: 0.263 ≈ 0.110 + 0.204 ✅（⚠️ 注意 reverse_kl 在降但 bias_correction 爆炸，总 loss 反升）
- **topk_mass 双降（v3sft）**：student 0.994→0.841，teacher 0.997→0.845，gap 始终 0.003-0.004 = EMA 假收敛标志（详见 Q11 情况一）
- **冻结教师 topk_mass 应稳定**：v3-no-ema/v4 全程不变（±0.005），因为教师不更新且学生输出质量稳定
- ⚠️ **v5-27B teacher_topk_mass 异常下降**（0.986→0.807）：冻结教师不应下降，但学生从 step 46 起急剧退化，生成异常 token 导致 27B 教师在这些位置分布变平坦（详见 Q11 情况二）
- **v5-27B bias_correction 爆炸**（0.012→0.204，17 倍）：学生 40% 概率质量移到教师 top-100 外，发生支撑域外 mode collapse
- **v5-27B reverse_kl_term 持续下降**（0.48→0.11）是假象：学生概率移出 top-k 求和范围导致总和变小（详见 Q10），但 bias_correction 暴露真实状态
- **num_distill_tokens**：v3sft 从 1378 增到 2194（回复变长 → 更多 token 参与蒸馏）；其余实验稳定
- mask/always_on/image_swap/fallback 全恒 1.0 或 0.0，说明蒸馏机制全程正常工作无异常

---

### 6.3 rollout_corr/* — on-policy 一致性指标（25 tags）

| Tag | 含义 |
|-----|------|
| `kl` | KL(π_rollout ‖ π_train)，策略漂移量，越低越好 |
| `k3_kl` | 三阶近似 KL，与 kl 近似，用于交叉验证 |
| `ppl_ratio` | training_ppl / rollout_ppl，应 ≈ 1.0，偏离 = IS 失稳 |
| `training_ppl` | 训练步策略对 rollout 样本的困惑度 |
| `rollout_ppl` | rollout 时的困惑度 |
| `training_log_ppl` | log(training_ppl)，对数尺度 |
| `rollout_log_ppl` | log(rollout_ppl)，对数尺度 |
| `log_ppl_diff` | training_log_ppl - rollout_log_ppl，对数尺度策略漂移 |
| `log_ppl_abs_diff` | |log_ppl_diff| 的绝对值 |
| `log_ppl_diff_max` | batch 内最大 log_ppl_diff（极端 token 的漂移） |
| `log_ppl_diff_min` | batch 内最小 log_ppl_diff |
| `chi2_seq` | 序列级 χ² 统计量，衡量 rollout/train 分布差异 |
| `chi2_token` | token 级 χ² 统计量 |
| `rollout_is_eff_sample_size` | IS 权重的有效样本大小，1.0 = 无 IS 效应 |
| `rollout_is_mean` | IS 权重均值，1.0 = on-policy 完美 |
| `rollout_is_max` | IS 权重最大值 |
| `rollout_is_min` | IS 权重最小值 |
| `rollout_is_std` | IS 权重标准差，0 = 无方差 |
| `rollout_is_seq_mean` | 序列级 IS 权重均值 |
| `rollout_is_seq_max` | 序列级 IS 最大值 |
| `rollout_is_seq_min` | 序列级 IS 最小值 |
| `rollout_is_seq_std` | 序列级 IS 标准差 |
| `rollout_is_seq_max_deviation` | 序列级 IS 最大偏差 |
| `rollout_is_ratio_fraction_high` | IS ratio > 阈值的比例，0 = 无超标 |
| `rollout_is_seq_fraction_high` | 序列级 IS > 阈值的比例 |
| `rollout_is_ratio_fraction_low` | IS ratio < 阈值的比例 |
| `rollout_is_seq_fraction_low` | 序列级 IS < 阈值的比例 |

#### 四实验数据（核心指标）

| Tag | 实验 | first | last | min | max |
|-----|------|-------|------|-----|-----|
| **kl** | v3sft | 0.1787 | 0.0128 | 0.0116 | 0.1920 |
| | v3-no-ema | 0.1889 | 0.1147 | 0.1102 | 0.2033 |
| | v4-fixed | 0.1907 | 0.1085 | 0.0931 | 0.2147 |
| | v5-27B | 0.1932 | 0.3980 | 0.1932 | 0.3980 |
| **k3_kl** | v3sft | 0.1715 | 0.0132 | 0.0114 | 0.1932 |
| | v3-no-ema | 0.1910 | 0.1150 | 0.1087 | 0.2237 |
| | v4-fixed | 0.1847 | 0.1120 | 0.0917 | 0.2048 |
| | v5-27B | 0.1914 | 0.3932 | 0.1914 | 0.3932 |
| **ppl_ratio** | v3sft | 6304 | 65595 | 1.025 | 901356 |
| | v3-no-ema | 1.240 | 1.149 | 1.137 | 3.623 |
| | v4-fixed | 26512 | 9670 | 37.7 | 1042620 |
| | v5-27B | 1.255 | 194.5 | 1.253 | 5175 |
| **training_ppl** | v3sft | 16226 | 109534 | 2.74 | 1387396 |
| | v3-no-ema | 4.52 | 6.55 | 3.60 | 2225 |
| | v4-fixed | 36393 | 22420 | 73.8 | 1456464 |
| | v5-27B | 6.54 | 464.6 | 5.37 | 9712 |
| **rollout_ppl** | v3sft | 4.87 | 148.3 | 1.76 | 180.6 |
| | v3-no-ema | 3.56 | 5.67 | 3.10 | 1711 |
| | v4-fixed | 3.53 | 4.88 | 1.96 | 21.28 |
| | v5-27B | 4.99 | 4.78 | 3.87 | 136.7 |
| **log_ppl_diff** | v3sft | 0.2108 | 0.0685 | 0.0239 | 0.2957 |
| | v3-no-ema | 0.2014 | 0.1307 | 0.1222 | 0.2191 |
| | v4-fixed | 0.4208 | 0.2276 | 0.1576 | 0.4208 |
| | v5-27B | 0.2070 | 0.6193 | 0.2070 | 0.6193 |
| **log_ppl_diff_max** | v3sft | 15.39 | 17.65 | 0.59 | 20.36 |
| | v3-no-ema | 1.07 | 0.86 | 0.60 | 7.67 |
| | v4-fixed | 15.93 | 15.46 | 9.46 | 20.19 |
| | v5-27B | 1.86 | 11.33 | 1.48 | 14.33 |
| **chi2_seq** | v3sft | 1081 | 23.3 | -0.82 | 556792 |
| | v3-no-ema | -0.52 | -0.41 | -0.82 | 17912 |
| | v4-fixed | -0.84 | -0.60 | -0.94 | 12095 |
| | v5-27B | -0.61 | 26.25 | -0.97 | 35.02 |
| **chi2_token** | v3sft | 1.24 | 0.068 | 0.029 | 49.6 |
| | v3-no-ema | 2.11 | 1.20 | 0.45 | 1698 |
| | v4-fixed | 1.47 | 5.69 | 0.40 | 38.4 |
| | v5-27B | 1.35 | 15.10 | 1.34 | 172.7 |

**IS 权重统计（全部恒为 1.0/0.0 的 tag）**：

| Tag | 全实验 |
|-----|--------|
| rollout_is_eff_sample_size | 恒 1.0 |
| rollout_is_mean / max / min | 恒 1.0 / 1.0 / 1.0 |
| rollout_is_std | 恒 0.0 |
| rollout_is_seq_mean / max / min | 恒 1.0 / 1.0 / 1.0 |
| rollout_is_seq_std / max_deviation | 恒 0.0 / 0.0 |
| rollout_is_ratio_fraction_high / low | 恒 0.0 / 0.0 |
| rollout_is_seq_fraction_high / low | 恒 0.0 / 0.0 |

**趋势点评**：
- **kl/k3_kl 几乎一致**（差异 < 0.005），交叉验证通过
- **v3sft kl 最低**（0.013）但这是 EMA 退化导致的假象；v3-no-ema/v4 稳定在 0.11；v5 上升到 0.40（发散）
- **ppl_ratio 稳定性排序**：v3-no-ema（1.1-1.2）✅ > v5（1.3-5175）🔴 > v3sft（1.0-90 万）🔴 > v4（38-104 万）🔴
- **IS 权重全部恒 1.0**：说明 rollout_correction 的 IS 权重没有实际生效（token 级 IS 计算后 clip 到 1.0，因为 ratio ≤ 1.0 且 clip 到 is_clip=2.0 下界不触发）。这意味着 ppl_ratio 虽然波动大，但实际 IS 权重没有放大 loss
- **log_ppl_diff_max**：v3sft/v4 的 SFT 起点导致极端 token 的 log_ppl_diff 达 15-20（e^15 ≈ 327 万倍 ppl 差异），但 IS 权重 clip 后不影响 loss
- **v5-27B 所有指标持续恶化**：kl 上升、ppl_ratio 爆炸、log_ppl_diff 翻倍 = 确认发散

---

### 6.4 global_seqlen/* — 全局序列长度统计（6 tags）

| Tag | 含义 |
|-----|------|
| `mean` | 全局 batch 平均 token 总数 |
| `max` | 最大序列长度 |
| `min` | 最小序列长度 |
| `minmax_diff` | max - min，衡量 batch 内长度差异 |
| `balanced_max` | 负载均衡后的最大值 |
| `balanced_min` | 负载均衡后的最小值 |

#### 四实验数据

| Tag | 实验 | first | last | min | max |
|-----|------|-------|------|-----|-----|
| **mean** | v3sft | 79536 | 107560 | 76282 | 114226 |
| | v3-no-ema | 83531 | 88464 | 80898 | 90941 |
| | v4-fixed | 78751 | 81141 | 75631 | 85107 |
| | v5-27B | 83404 | 90507 | 78028 | 90507 |
| **max** | v3sft | 86865 | 121133 | 81504 | 132620 |
| | v3-no-ema | 90613 | 93024 | 85281 | 105526 |
| | v4-fixed | 85849 | 87930 | 79854 | 97395 |
| | v5-27B | 92427 | 107951 | 82246 | 107951 |
| **min** | v3sft | 72695 | 97312 | 65707 | 106726 |
| | v3-no-ema | 77595 | 84536 | 69645 | 86933 |
| | v4-fixed | 72171 | 76911 | 65522 | 81536 |
| | v5-27B | 76965 | 83684 | 71198 | 85148 |
| **minmax_diff** | v3sft | 14170 | 23821 | 5107 | 46556 |
| | v3-no-ema | 13018 | 8488 | 4229 | 27471 |
| | v4-fixed | 13678 | 11019 | 3995 | 23083 |
| | v5-27B | 15462 | 24267 | 5494 | 24516 |

**趋势点评**：
- v3sft 的 seqlen 持续增长（79K→108K），因为回复长度从 230 增到 503 → 更长回复贡献更多 token
- v3-no-ema/v4 稳定（80-88K），回复长度变化小
- balanced_max/min 差距极小（<100），说明负载均衡工作正常

---

### 6.5 perf/* — 性能指标（7 tags）

| Tag | 含义 |
|-----|------|
| `throughput` | 吞吐量（tokens/sec） |
| `time_per_step` | 每步耗时（秒） |
| `mfu/actor` | Actor 模型利用率（model FLOPs utilization） |
| `total_num_tokens` | 总处理 token 数 |
| `max_memory_allocated_gb` | GPU 最大已分配显存（GB） |
| `max_memory_reserved_gb` | GPU 最大预留显存（GB） |
| `cpu_memory_used_gb` | CPU 内存使用量（GB） |

#### 四实验数据

| Tag | 实验 | first | last | min | max |
|-----|------|-------|------|-----|-----|
| **throughput** | v3sft | 178 | 549 | 178 | 594 |
| | v3-no-ema | 187 | 522 | 187 | 564 |
| | v4-fixed | 205 | 493 | 205 | 522 |
| | v5-27B | 151 | 359 | 151 | 402 |
| **time_per_step** | v3sft | 446s | 196s | 146s | 446s |
| | v3-no-ema | 446s | 169s | 149s | 446s |
| | v4-fixed | 384s | 165s | 152s | 384s |
| | v5-27B | 552s | 252s | 198s | 552s |
| **mfu/actor** | v3sft | 0.103 | 0.328 | 0.103 | 0.354 |
| | v3-no-ema | 0.108 | 0.324 | 0.108 | 0.351 |
| | v4-fixed | 0.121 | 0.337 | 0.121 | 0.345 |
| | v5-27B | 0.079 | 0.209 | 0.079 | 0.225 |
| **total_num_tokens** | v3sft | 636K | 860K | 610K | 914K |
| | v3-no-ema | 668K | 708K | 647K | 728K |
| | v4-fixed | 630K | 649K | 605K | 681K |
| | v5-27B | 667K | 724K | 624K | 724K |
| **max_mem_alloc_gb** | v3sft | 102.6 | 113.1 | 102.6 | 113.1 |
| | v3-no-ema | 101.8 | 112.0 | 101.8 | 112.0 |
| | v4-fixed | 102.0 | 112.6 | 102.0 | 112.6 |
| | v5-27B | 102.4 | 113.6 | 102.1 | 113.6 |
| **max_mem_reserved_gb** | v3sft | 123.1 | 134.5 | 123.1 | 134.5 |
| | v3-no-ema | 123.0 | 133.9 | 123.0 | 133.9 |
| | v4-fixed | 122.7 | 133.5 | 122.7 | 133.5 |
| | v5-27B | 124.8 | 135.9 | 124.0 | 135.9 |
| **cpu_mem_gb** | v3sft | 310 | 495 | 310 | 499 |
| | v3-no-ema | 310 | 493 | 310 | 500 |
| | v4-fixed | 323 | 497 | 323 | 509 |
| | v5-27B | 605 | 746 | 605 | 746 |

**趋势点评**：
- v5-27B 的 time_per_step 最高（252s vs 其他 165-196s），因为 27B 教师前向推理更慢
- v5-27B 的 mfu 最低（0.21 vs 0.33），27B 教师占用了大量计算但不在 actor MFU 统计内
- v5-27B 的 CPU 内存最高（746GB vs 495-500GB），27B 教师模型参数在 CPU offload
- 首步 time_per_step 均偏高（384-552s），因为 warmup + triton JIT 编译 + KV cache 预分配
- GPU 显存 8 卡 H20 每卡约 13-14GB reserved（134GB / 8 ≈ 16.8GB/卡，含 teacher+student+ref）

---

### 6.6 prompt_length/* & response_length*/* — 长度统计（12 tags）

| Tag | 含义 |
|-----|------|
| `prompt_length/mean` | prompt 平均长度 |
| `prompt_length/max` | prompt 最大长度 |
| `prompt_length/min` | prompt 最小长度 |
| `prompt_length/clip_ratio` | prompt 被截断比例 |
| `response_length/mean` | 回复平均长度 |
| `response_length/max` | 回复最大长度 |
| `response_length/min` | 回复最小长度 |
| `response_length/clip_ratio` | 回复被截断（达到 max_response_length=2048）比例 |
| `response/aborted_ratio` | 回复中止比例（生成失败） |
| `response_length_non_aborted/*` | 排除中止后的回复长度统计（与 response_length 几乎一致，因为 aborted=0） |

#### 四实验数据

| Tag | 实验 | first | last | min | max |
|-----|------|-------|------|-----|-----|
| **prompt_length/mean** | v3sft | 599 | 618 | 569 | 641 |
| | v3-no-ema | 599 | 615 | 574 | 634 |
| | v4-fixed | 599 | 602 | 574 | 633 |
| | v5-27B | 599 | 630 | 585 | 630 |
| **prompt_length/clip_ratio** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |
| **response_length/mean** | v3sft | 230 | 503 | 206 | 576 |
| | v3-no-ema | 271 | 307 | 261 | 337 |
| | v4-fixed | 221 | 243 | 199 | 271 |
| | v5-27B | 270 | 313 | 224 | 323 |
| **response_length/max** | 全部 | ~615-1176 | 2048 | ~515-1176 | 2048 |
| **response_length/clip_ratio** | v3sft | 0.0 | 0.094 | 0.0 | 0.128 |
| | v3-no-ema | 0.0 | 0.001 | 0.0 | 0.009 |
| | v4-fixed | 0.0 | 0.007 | 0.0 | 0.012 |
| | v5-27B | 0.0 | 0.034 | 0.0 | 0.035 |
| **response_length/min** | 全部 | 1-5 | 1-5 | 1 | 5-23 |
| **response/aborted_ratio** | 全部 | 0.0 | 0.0 | 0.0 | 0.0 |

**趋势点评**：
- **prompt_length 全实验一致**（599-630）：同一数据集，prompt 含文本+图像 token，长度固定
- **response_length 差异显著**：
  - v3sft：230→503（翻倍），学生在 EMA 退化下生成越来越长
  - v3-no-ema/v4：稳定在 220-310（SFT/base 起点已会写摘要，长度稳定）
  - v5-27B：270→313（微增，但刚开始）
- **v3sft clip_ratio 9.4%**：回复变长后 ~10% 超过 2048 被截断，说明 EMA 退化还伴随着长度膨胀
- **aborted_ratio 全 0**：无生成失败

---

### 6.7 num_turns/* — 对话轮数（3 tags）

| Tag | 含义 | 全实验值 |
|-----|------|----------|
| `num_turns/mean` | 平均对话轮数 | 恒 2.0 |
| `num_turns/max` | 最大对话轮数 | 恒 2.0 |
| `num_turns/min` | 最小对话轮数 | 恒 2.0 |

**点评**：固定两轮对话（图片→摘要），无需关注。

---

### 6.8 training/* — 训练进度（2 tags）

| Tag | 含义 | 四实验 last |
|-----|------|-------------|
| `training/global_step` | 全局步数 | v3sft=493, v3-no-ema=273, v4=216, v5=29 |
| `training/epoch` | 当前 epoch | 全部恒 0.0（日志四舍五入，实际在 0-2 之间） |

**点评**：v3sft 已完成 2 epoch（493/751 步 ≈ 但 batch_size 对 epoch 计算有影响），其余实验进行中。

---

### 6.9 timing_s/* — 耗时分解（~28 tags）

| Tag | 含义 |
|-----|------|
| `timing_s/step` | 单步总耗时 = gen + adv + update_actor |
| `timing_s/gen` | rollout 生成耗时 |
| `timing_s/adv` | advantage 计算耗时 |
| `timing_s/update_actor` | actor 更新总耗时 = student_forward + teacher_forward + backward + optimizer_step + loss_compute + teacher_ema_update |
| `timing_s/update_actor/student_forward` | 学生前向（计算 logp） |
| `timing_s/update_actor/teacher_forward` | 教师前向（计算 teacher logp） |
| `timing_s/update_actor/backward` | 反向传播 |
| `timing_s/update_actor/optimizer_step` | 优化器 step |
| `timing_s/update_actor/loss_compute` | loss 计算 |
| `timing_s/update_actor/teacher_ema_update` | 教师 EMA 更新 |
| `timing_s/agent_loop/generate_sequences/{mean,max,min}` | rollout 生成耗时统计 |
| `timing_s/agent_loop/slowest/*` | 最慢 worker 的统计 |
| `timing_s/agent_loop/tool_calls/{mean,max,min}` | 工具调用耗时（本项目无工具，恒 0） |
| `timing_s/dump_rollout_generations` | rollout 序列化耗时 |
| `timing_s/save_checkpoint` | 保存 checkpoint 耗时 |
| `timing_s/start_profile` / `stop_profile` | 性能 profile 起止 |
| `timing_per_token_ms/{adv,gen,update_actor}` | 每 token 毫秒耗时 |

#### 四实验关键耗时对比（last 值，秒）

| Tag | v3sft | v3-no-ema | v4-fixed | v5-27B |
|-----|-------|-----------|----------|--------|
| step | 196s | 169s | 165s | 252s |
| gen | 45s | 41s | 41s | 43s |
| adv | 23s | 21s | 21s | 25s |
| update_actor | 127s | 107s | 102s | 183s |
| ├ student_forward | 20s | 17s | 15s | 18s |
| ├ teacher_forward | 37s | 32s | 26s | **85s** |
| ├ backward | 55s | 46s | 43s | 48s |
| ├ optimizer_step | 0.03s | 0.03s | 0.03s | 0.03s |
| ├ loss_compute | 0.02s | 0.02s | 0.02s | 0.02s |
| └ teacher_ema_update | 2.5s | 0.0001s | 0.0001s | 0.0001s |

**趋势点评**：
- **v5-27B teacher_forward 85s**（vs 其他 26-37s）：27B 教师前向是最大瓶颈，占 update_actor 的 46%
- **v3sft teacher_ema_update 2.5s**（vs 其他 0.0001s）：唯一开 EMA 的实验，每步需更新教师权重
- **gen 耗时稳定**（41-45s）：rollout 生成耗时与模型大小无关（vllm 优化后 9B 生成很快）
- **backward 是 update_actor 大头**（43-55s）：9B 模型反向传播占主要时间
- 首步耗时偏高（step 384-552s）因 JIT 编译 + KV cache 预分配，之后稳定
- **tool_calls 全 0**：本项目不使用工具调用
- **save_checkpoint 12-15s**：每 150 步存一次，可接受

---

### 6.10 指标优先级速查表

| 优先级 | Tag | 看什么 |
|--------|-----|--------|
| 🔴 核心 | `actor/vopd_loss` | loss 是否下降（注意 v5 先降后升=延迟发散） |
| 🔴 核心 | `self_distillation/raw_jsd_token_mean` | 师生分布差异是否缩小 |
| 🔴 核心 | `self_distillation/mopd_reverse_kl_term_mean` | 反向 KL 主项（⚠️ 单看会误判，须配 bias_correction） |
| 🔴 核心 | `self_distillation/mopd_bias_correction_mean` | **偏差校正是否爆炸（v5 发散的真正信号）** |
| 🔴 核心 | `self_distillation/student_on_teacher_topk_mass_mean` | 学生是否脱离教师支撑域（<0.9=危险） |
| 🔴 核心 | `self_distillation/teacher_topk_mass_mean` | 教师是否退化（EMA=自身退化，冻结=学生输出退化） |
| 🔴 核心 | `actor/grad_norm` | 是否爆炸 |
| 🔴 核心 | `rollout_corr/kl` | on-policy 是否成立 |
| 🟡 重要 | `rollout_corr/ppl_ratio` | IS 权重是否稳定 |
| 🟡 重要 | `rollout_corr/log_ppl_diff` | 策略漂移对数尺度 |
| 🟡 重要 | `self_distillation/mopd_bias_correction_mean` | 偏差校正是否发散 |
| 🟡 重要 | `response_length/mean` | 生成长度是否合理 |
| 🟡 重要 | `response_length/clip_ratio` | 截断比例 |
| ⚪ 辅助 | `rollout_corr/training_ppl` / `rollout_ppl` | 困惑度绝对值 |
| ⚪ 辅助 | `rollout_corr/chi2_*` | χ² 统计 |
| ⚪ 辅助 | `self_distillation/num_distill_tokens` | 蒸馏 token 数 |
| ⚪ 工程 | `perf/*` | GPU 利用率/显存 |
| ⚪ 工程 | `timing_s/*` | 耗时分解 |
| ⚪ 工程 | `global_seqlen/*` | 序列长度 |
| ⚪ 忽略 | `actor/{kl_loss,grpo_loss,lr,policy_fallback}` | 配置关闭，恒 0 |
| ⚪ 忽略 | `rollout_corr/rollout_is_*` (16 个) | IS 权重统计，恒 1.0/0.0 |
| ⚪ 忽略 | `num_turns/*` | 固定 2 轮 |
| ⚪ 忽略 | `training/epoch` | 日志精度不足 |
| ⚪ 忽略 | `self_distillation/{mask,always_on,image_swap,fallback}` | 恒 1.0/0.0 |
| ⚪ 忽略 | `timing_s/agent_loop/tool_calls/*` | 无工具调用 |
| ⚪ 忽略 | `prompt_length/*` | 固定 prompt |

---

### 6.11 TensorBoard 筛选核心指标的操作方法

TensorBoard Web UI 左侧 **Filter tags** 搜索框支持正则表达式，直接粘贴以下正则即可分层筛选。

#### 🔴 核心指标（8 条）— Loss 健康度

```
^(actor/vopd_loss|actor/grad_norm|self_distillation/raw_jsd_token_mean|self_distillation/mopd_reverse_kl_term_mean|self_distillation/mopd_bias_correction_mean|self_distillation/student_on_teacher_topk_mass_mean|self_distillation/teacher_topk_mass_mean|rollout_corr/kl)$
```

#### 🟡 重要指标（5 条）— 稳定性与生成长度

```
^(rollout_corr/ppl_ratio|rollout_corr/log_ppl_diff|response_length/mean|response_length/clip_ratio|rollout_corr/training_ppl)$
```

#### 🔴+🟡 合并（一条正则筛选全部核心+重要）

```
^(actor/vopd_loss|actor/grad_norm|self_distillation/raw_jsd_token_mean|self_distillation/mopd_reverse_kl_term_mean|self_distillation/mopd_bias_correction_mean|self_distillation/student_on_teacher_topk_mass_mean|self_distillation/teacher_topk_mass_mean|rollout_corr/kl|rollout_corr/ppl_ratio|rollout_corr/log_ppl_diff|response_length/mean|response_length/clip_ratio|rollout_corr/training_ppl)$
```

#### ⚪ 只看非零有效指标（排除恒 0/恒 1 的噪音 tag）

```
^(actor/(vopd_loss|grad_norm)|self_distillation/(raw_jsd|mopd_reverse_kl|mopd_bias|student_on_teacher|teacher_topk|num_distill)|rollout_corr/(kl|k3_kl|ppl_ratio|training_ppl|rollout_ppl|log_ppl_diff|chi2_seq|chi2_token)|response_length/(mean|clip_ratio)|perf/(throughput|time_per_step|mfu/actor)|timing_s/(step|gen|update_actor))
```

> ⚠️ 最后这条用前缀模糊匹配（去掉 `$` 结尾符），会匹配到前缀开头的所有子 tag。

#### 操作步骤

1. 打开 TensorBoard（6007 端口）
2. 左侧 **SCALARS** 面板顶部找到 **Filter tags** 输入框
3. 粘贴上面任一正则
4. 右侧图表区只显示匹配的曲线

#### 按前缀分组筛选（单类深入查看）

TensorBoard 搜索框不支持反向正则，但可用前缀快速定位某一类指标：

| 想看 | 输入 |
|------|------|
| 蒸馏全部 | `self_distillation/` |
| rollout 全部 | `rollout_corr/` |
| 耗时分解 | `timing_s/` |
| 性能 | `perf/` |
| 回复长度 | `response_length/` |

#### 命令行替代方案（tbparse + pandas，导出 CSV 做横向对比图）

TensorBoard UI 适合实时监控，但做四实验同图横向对比时用 tbparse 更灵活：

```bash
# 安装 tbparse
sudo -u meimei.wu -i bash -lc 'pip install tbparse pandas -q'

# 导出核心指标为 CSV
sudo -u meimei.wu -i python3 << 'PYEOF'
from tbparse import SummaryReader
import pandas as pd

log_dir = "/data4/wumeimei/flash_note/RP-OPSD/tensorboard_log"
reader = SummaryReader(log_dir)

# 核心指标列表
core_tags = [
    "actor/vopd_loss", "actor/grad_norm",
    "self_distillation/raw_jsd_token_mean",
    "self_distillation/mopd_reverse_kl_term_mean",
    "self_distillation/mopd_bias_correction_mean",
    "self_distillation/student_on_teacher_topk_mass_mean",
    "self_distillation/teacher_topk_mass_mean",
    "rollout_corr/kl",
]

df = reader.scalars
filtered = df[df.tag.isin(core_tags)]
filtered.to_csv("/tmp/core_metrics.csv", index=False)
print(f"导出 {len(filtered)} 行，{filtered.tag.nunique()} 个指标")
PYEOF
```

> 日常看曲线用 UI 正则筛选最快；需要做四实验同图横向对比时用 tbparse 导出 CSV 更灵活。

---

## 七、逐实验×逐指标严格分析

> 每个实验分**必看**（8 条核心）和**扫一眼**（7 条辅助）两组，逐条给出：趋势 → 现象 → 原因。
> 数据截止 2026-09-03，v3sft=493 步，v3-no-ema=273 步，v4=216 步，v5=69 步。

### 7.1 v3sft — EMA 教师 + SFT 学生（493 步，2 epoch 完成）

#### 必看指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 1 | `vopd_loss` | 0.214→**0.007** ↓↓↓ | 单调下降 30 倍，看似完美收敛 | **假象**：EMA 教师跟随学生退化，师生分布同步变差 → KL 自然趋零 |
| 2 | `raw_jsd_token_mean` | 0.214→0.006 ↓↓↓ | 与 vopd_loss 完全重合 | 同一指标的 IS 加权前/后值，IS 权重恒 1.0 所以一致 |
| 3 | `mopd_reverse_kl_term_mean` | 0.211→**0.003** ↓↓↓ | 占 loss 的 99%+，单调下降 | 师生同步退化 → 分布趋同 → 反向 KL 自然趋零。bias_correction 仅 0.003 说明 gap 极小 |
| 4 | `mopd_bias_correction_mean` | 0.003→0.003 → | **全程稳定在 0.002-0.007**，gap 不变 | EMA 教师跟踪学生 → 师生 topk_mass 同速下降 → gap 恒定 = 假收敛的铁证 |
| 5 | `student_topk_mass` | 0.994→**0.841** ↓↓ | 持续下降 15 个百分点 | 学生在教师 top-100 上的概率质量降低 → 学生分布变平坦/退化 |
| 6 | `teacher_topk_mass` | 0.997→**0.845** ↓↓ | 与 student 几乎平行下降 | EMA 教师自身退化（权重被学生拉差）→ 教师分布也变平坦 |
| 7 | `grad_norm` | 11.3→1.2 ↓ | 整体下降，但 step 219 spike 到 **171** | 训练基本稳定，219 步的 spike 可能是 bad batch 或 LR warmup 后的阶段性不稳定 |
| 8 | `rollout_corr/kl` | 0.179→**0.013** ↓↓ | 下降到极低值 | 同样是假象：rollout 和 train 策略漂移小不是因为学生学好了，而是因为策略"锁定"到退化的窄分布上 |

#### 扫一眼指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 9 | `ppl_ratio` | 6304→65596 🔴 | **极度不稳定**，在 1.0 到 90 万之间剧烈震荡 | SFT 起点的 training_ppl 极高（1.6 万→11 万），个别 token 的 ppl 差异巨大 → IS 权重不稳定（但实际 IS 被 clip 到 1.0 不影响 loss） |
| 10 | `log_ppl_diff` | 0.211→0.069 ↓ | 下降但震荡 | 对数尺度看策略漂移在缩小，但结合 ppl_ratio 震荡说明是"窄分布锁定"而非"真正对齐" |
| 11 | `training_ppl` | 16226→**109534** ↑↑↑ | 持续上升 7 倍 | 学生策略对 rollout 样本的困惑度飙升 = 学生生成的文本质量在恶化 |
| 12 | `rollout_ppl` | 4.87→**148** ↑↑↑ | 持续上升 30 倍 | rollout 时学生自身困惑度飙升 = 学生生成的 token 越来越"意外"/低概率 = 退化实锤 |
| 13 | `chi2_token` | 1.24→0.067 ↓ | 下降到接近 0 | token 级 χ² 统计趋零 = rollout/train 分布趋同（但这是同向退化导致的趋同） |
| 14 | `response_length/mean` | 230→**503** ↑↑ | 回复长度翻倍 | 学生退化伴随长度膨胀——生成冗长但低质量的文本 |
| 15 | `clip_ratio` | 0→**0.094** ↑ | 9.4% 回复超过 2048 被截断 | 长度膨胀的后果——回复越来越长直到撞到 max_length |

#### v3sft 一句话诊断

**假收敛**。loss/kl/reverse_kl 全部漂亮地降到接近 0，但 topk_mass 双降至 0.84、ppl 飙升、回复翻倍。根因：EMA 正反馈退化——教师跟随学生变差，师生同步退化导致分布趋同。`bias_correction` 全程稳定 0.003 是假收敛的铁证（真收敛 gap 应缩小到 0，假收敛 gap 不变）。

---

### 7.2 v3-no-ema — 冻结 base 教师 + base 学生（273 步，进行中）

#### 必看指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 1 | `vopd_loss` | 0.122→**0.057** ↓ | 下降约一半后 plateau 在 0.06 | **真实收敛**：冻结教师不变，学生学到了能学的部分，剩余 gap 0.06 是师生真实能力差 |
| 2 | `raw_jsd_token_mean` | 0.122→0.057 ↓ | 与 vopd_loss 完全重合 | IS 权重恒 1.0，加权前后一致 |
| 3 | `mopd_reverse_kl_term_mean` | 0.119→**0.047** ↓ | 占 loss 的 82%，下降后 plateau | 反向 KL 主项真实下降，学生确实在靠近教师分布 |
| 4 | `mopd_bias_correction_mean` | 0.003→**0.010** ↑ | 缓慢增长 3 倍但绝对值小 | 师生 gap 从 0.003 增到 0.010：学生在 top-100 内的概率质量略低于教师 = 正常的蒸馏残差，非发散 |
| 5 | `student_topk_mass` | 0.986→0.976 ↓ | 微降 1 个百分点 | 学生在教师 top-100 上的质量微降，但 0.976 仍然很高 = 学生大部分概率在教师支撑域内 |
| 6 | `teacher_topk_mass` | 0.989→0.986 → | **全程稳定**（±0.005） | 冻结教师权重不变 + 学生输出质量稳定 → 教师分布锐度不变 |
| 7 | `grad_norm` | 14.3→6.98 ↓ | 整体下降，但 step 242 spike 到 **46.9** | 训练基本稳定，末步 spike 需关注后续是否持续；可能是 bad batch 或 plateau 阶段梯度方向震荡 |
| 8 | `rollout_corr/kl` | 0.189→0.115 ↓ | 下降后 plateau 在 0.11 | 策略漂移收敛到 0.11 = on-policy 假设基本成立但有残差漂移 |

#### 扫一眼指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 9 | `ppl_ratio` | 1.24→**1.15** ✅ | **全程稳定在 1.1-1.2**，max 仅 3.6 | 四实验中唯一稳定的 ppl_ratio = base 学生起点自然，IS 权重无失稳 |
| 10 | `log_ppl_diff` | 0.201→0.131 ↓ | 下降后 plateau | 策略漂移在缩小并稳定 = 真实收敛信号 |
| 11 | `training_ppl` | 4.52→6.55 → | **稳定在 4-7** | 学生策略对 rollout 的困惑度低且稳定 = 生成质量没有恶化 |
| 12 | `rollout_ppl` | 3.56→5.67 → | **稳定在 3-6** | rollout 困惑度低且稳定 = 学生生成正常文本 |
| 13 | `chi2_token` | 2.11→1.20 ↓ | 下降 | rollout/train 分布差异在缩小 = 真实收敛 |
| 14 | `response_length/mean` | 271→307 → | 稳定在 260-337 | 回复长度正常，无膨胀 |
| 15 | `clip_ratio` | 0→0.001 → | 几乎为 0 | 无截断 = 回复长度健康 |

#### v3-no-ema 一句话诊断

**真实收敛，最健康的实验**。loss plateau 在 0.06 代表师生真实能力差（base 9B 蒸馏 base 9B，能力提升空间有限）。ppl_ratio 稳定 1.1、topk_mass 稳定、ppl 稳定、回复长度稳定。唯一需关注：末步 grad_norm=46.9 的 spike。

---

### 7.3 v4-fixed — 冻结 SFT 教师 + SFT 学生（216 步，进行中）

#### 必看指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 1 | `vopd_loss` | 0.210→**0.126** ↓ | 下降后在 0.11-0.14 之间震荡 plateau | SFT 师生初始差距小，学生学到了一部分，但 plateau 在 0.12 = SFT 模型间残差 |
| 2 | `raw_jsd_token_mean` | 0.210→0.125 ↓ | 与 vopd_loss 基本一致 | IS 权重恒 1.0 |
| 3 | `mopd_reverse_kl_term_mean` | 0.207→**0.115** ↓ | 占 loss 的 92%，下降后 plateau | 反向 KL 在缩小，学生确实在靠近教师 |
| 4 | `mopd_bias_correction_mean` | 0.003→**0.010** ↑ | 缓慢增长 3 倍 | 与 v3-no-ema 相同模式：师生 gap 从 0.003 增到 0.010 = 正常蒸馏残差 |
| 5 | `student_topk_mass` | 0.994→0.985 ↓ | 微降 0.9 个百分点 | 学生在教师 top-100 上质量微降但 0.985 仍高 = 正常 |
| 6 | `teacher_topk_mass` | 0.997→0.995 → | **非常稳定**（±0.002） | 冻结 SFT 教师 + 学生输出稳定 → 教师分布锐度几乎不变 |
| 7 | `grad_norm` | 28.4→17.1 ↓ | **震荡明显**，范围 3-77，无收敛趋势 | SFT 起点导致 loss landscape 复杂，梯度方向不稳定；max 77 说明偶有大梯度 batch |
| 8 | `rollout_corr/kl` | 0.191→0.108 ↓ | 下降后 plateau 在 0.10-0.11 | 策略漂移收敛 = on-policy 假设基本成立 |

#### 扫一眼指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 9 | `ppl_ratio` | 26512→9670 🔴 | **极度不稳定**，在 38 到 104 万之间震荡 | SFT 起点的 training_ppl 极高（3.6 万→2.2 万），个别 token ppl 差异巨大 → IS 不稳定（但实际 IS clip 到 1.0 不影响 loss） |
| 10 | `log_ppl_diff` | 0.421→0.228 ↓ | 下降但绝对值高 | SFT 起点初始漂移大，在缩小但仍有 0.23 的对数差异 |
| 11 | `training_ppl` | 36393→22420 ↓ | 高但下降中 | SFT 模型对特定 token 预测很自信 → ppl 极端值多，但整体在降 |
| 12 | `rollout_ppl` | 3.53→4.88 → | **稳定在 2-15** | rollout 困惑度正常 = 学生生成质量没有恶化 |
| 13 | `chi2_token` | 1.47→**5.69** ↑ | **上升**，从 1.5 到 5.7 | rollout/train 分布差异在扩大 = 震荡 plateau 阶段的正常现象，但需关注是否持续上升 |
| 14 | `response_length/mean` | 221→243 → | 稳定在 199-271 | 回复长度正常 |
| 15 | `clip_ratio` | 0→0.007 → | 低，偶尔 0.7% | 基本无截断 |

#### v4-fixed 一句话诊断

**真实收敛但震荡**。loss plateau 在 0.12（SFT 师生残差），topk_mass 稳定（冻结教师有效防止退化），但 grad_norm 震荡（3-77）和 ppl_ratio 不稳（SFT 起点导致极端 ppl 值）。chi2_token 上升需关注。整体健康但不如 v3-no-ema 稳定。

---

### 7.4 v5-27B — 冻结 27B 教师 + base 学生（69 步，延迟发散）

#### 必看指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 1 | `vopd_loss` | 0.484→0.262 ↓↑ | 前 45 步降 0.48→0.22，step 46 起**反弹**到 0.26 | **延迟发散**：学生先学了容易对齐的部分，触及师生差异大的区域后崩溃 |
| 2 | `raw_jsd_token_mean` | 0.483→0.261 ↓↑ | 与 vopd_loss 同步先降后升 | IS 权重恒 1.0，趋势一致 |
| 3 | `mopd_reverse_kl_term_mean` | 0.470→**0.062** ↓↓↓ | **持续下降**，从 0.47 到 0.06 | **假象**：学生概率移出 top-k 求和范围 → 总和变小（详见 Q10），看起来 KL 在收敛，但概率跑到了教师 top-k 之外 |
| 4 | `mopd_bias_correction_mean` | 0.012→**0.199** ↑↑↑ | **爆炸 17 倍**，从 0.01 到 0.20 | **发散的真正信号**：师生 topk_mass gap 从 0.012 暴涨到 0.20 = 学生 36% 概率质量在教师 top-100 外 |
| 5 | `student_topk_mass` | 0.974→**0.638** ↓↓↓ | 暴跌 34 个百分点，step 46 起加速 | **支撑域外 mode collapse**：学生将概率集中到教师不会选的 token 上 |
| 6 | `teacher_topk_mass` | 0.986→**0.837** ↓↓ | 冻结教师异常下降 15 个百分点 | **被动反映**：学生输出退化为异常 token → 冻结 27B 教师在这些位置分布变平坦（详见 Q11 情况二） |
| 7 | `grad_norm` | 19.7→6.5 ↓ | 整体下降，step 5 spike 到 95 | warmup 阶段 spike，之后持续下降到 1.6——**梯度不大但方向错误**（学生集中到错误 mode） |
| 8 | `rollout_corr/kl` | 0.227→**0.321** ↑ | **上升**，on-policy 假设崩塌 | 学生策略持续漂移远离 rollout 策略 = 发散标志 |

#### 扫一眼指标

| # | 指标 | 趋势 | 现象 | 原因 |
|---|------|------|------|------|
| 9 | `ppl_ratio` | 760→8.9 ↓ | 从 760 降到 9（有波动） | 初始 SFT-less 导致高 ppl 差异，逐步稳定——但 ppl_ratio 下降不代表 loss 下降（bias_correction 在爆炸） |
| 10 | `log_ppl_diff` | 0.304→**0.364** ↑ | **上升** | 师生 log 困惑度差异在扩大 = 学生在远离教师 |
| 11 | `training_ppl` | 1354→1289 → | 高但相对稳定 | 学生策略对 rollout 的困惑度高 = base 9B 对 27B 生成的文本"看不懂" |
| 12 | `rollout_ppl` | 4.98→**915** ↑↑↑ | **爆炸 184 倍** | **最触目惊心的指标**：学生自身生成困惑度从 5 飙到 915 = 学生在生成自己都认为极不可能的 token = 近乎乱码 |
| 13 | `chi2_token` | 2.98→**7.56** ↑ | 上升 2.5 倍 | rollout/train 分布差异在扩大 = 确认发散 |
| 14 | `response_length/mean` | 255→294 → | moderate 增加，峰值 388（step 38） | 长度膨胀不严重，但偶有长回复 |
| 15 | `clip_ratio` | 0→0.003 → | 低，峰值 0.035（step 31） | 截断不严重 |

#### v5-27B 一句话诊断

**延迟发散（step 45 转折）**。前 45 步健康收敛（loss 0.48→0.22），step 46 起急剧恶化。核心失败模式：**支撑域外 mode collapse**——reverse_kl_term 持续下降是假象（学生概率移出 top-k 求和范围导致总和变小，详见 Q10），bias_correction 爆炸（0.012→0.199）暴露真相。`rollout_ppl` 从 5 飙到 915 是最直观的发散证据。根因：27B→9B 跨尺寸蒸馏 + base 学生起点 → 师生初始分布差异过大（JSD 起步 0.5 ≈ ln2），反向 KL 的 mode-seeking 特性导致学生跳到教师支撑域外。

---

### 7.5 四实验横向速查矩阵

| 指标 | v3sft (EMA) | v3-no-ema (冻base) | v4-fixed (冻SFT) | v5-27B (冻27B) |
|------|-------------|---------------------|-------------------|----------------|
| **vopd_loss 最终** | 0.007（假） | 0.057（真） | 0.126（震荡） | 0.262（反弹） |
| **reverse_kl 最终** | 0.003 | 0.047 | 0.115 | 0.062（假象↓） |
| **bias_correction 最终** | 0.003（稳定） | 0.010（缓增） | 0.010（缓增） | **0.199（爆炸）** |
| **student_topk 最终** | 0.841（降） | 0.976（稳） | 0.985（稳） | **0.638（暴跌）** |
| **teacher_topk 最终** | 0.845（降=退化） | 0.986（稳） | 0.995（稳） | 0.837（被动降） |
| **gap (t-s) 最终** | 0.004（不变） | 0.010（小） | 0.010（小） | **0.199（爆炸）** |
| **grad_norm 最终** | 1.2 | 6.98 | 17.1 | 6.5 |
| **kl 最终** | 0.013（假） | 0.115 | 0.108 | 0.321（↑） |
| **ppl_ratio** | 🔴 90 万 | ✅ 1.15 | 🔴 104 万 | 8.9 |
| **training_ppl** | 🔴 11 万 | ✅ 6.6 | 2.2 万 | 1289 |
| **rollout_ppl** | 🔴 148 | ✅ 5.7 | 4.9 | 🔴 **915** |
| **response_length** | 🔴 503 | ✅ 307 | 243 | 294 |
| **诊断** | 假收敛 | ✅ 真实收敛 | 真实但震荡 | 延迟发散 |

---

## §8 Rollout 重复与乱码检测

### 8.1 背景：为什么需要检测

TensorBoard 指标只能从数值层面反映训练健康度（loss/ppl/kl），但无法直接回答两个工程问题：

1. **生成的摘要有没有重复**？——同一个短语/句子反复出现（phrase-looping），是 RL 蒸馏常见的 mode collapse 表现
2. **生成的摘要有没有乱码**？——无意义字符堆叠、编码损坏、token 退化，是模型彻底崩塌的表现

代码中 `repetition_penalty=1.0`（即不施加任何重复惩罚），也没有内置的重复/乱码检测 TensorBoard 指标。因此需对 rollout JSONL 做离线文本检测作为补充。

### 8.2 检测方法

对每步 rollout 的 768 条样本逐条扫描三类信号：

| 信号 | 检测逻辑 | 阈值 | 含义 |
|------|----------|------|------|
| **word_rep**（词级重复） | 扫描 4~8-gram，统计同一 n-gram 出现次数 | ≥5 次 | 同一短语反复出现（phrase-looping） |
| **char_rep**（字符级重复） | 连续相同字符计数 | ≥10 | 同一字符连续堆叠（stuttering） |
| **gibberish**（乱码） | 非常规字符占比或出现 U+FFFD 替换符 | unusual_ratio > 0.3 或 bad_chars > 0 | 无意义文本/编码损坏 |

> rollout 文件字段：`{input, output, gts, score, step}`，取 `output` 字段为生成的摘要文本。

### 8.3 四实验时间序列

#### v3-no-ema（42 步，EMA 教师）

| step | word_rep | char_rep | gibberish |
|------|----------|----------|-----------|
| 1 | 0.8% | 0.0% | 0.0% |
| 16 | 1.0% | 0.3% | 0.1% |
| 31 | 1.8% | 0.4% | 0.1% |
| 42 | 1.7% | 0.3% | 0.0% |

**结论**：全程稳定，无退化。重复率在 0.5~1.8% 正常区间波动。

#### v4-fixed（216 步，冻 SFT 教师）

| step | word_rep | char_rep | gibberish |
|------|----------|----------|-----------|
| 1 | 0.0% | 0.5% | 0.3% |
| 61 | 1.4% | 0.4% | 0.5% |
| 121 | 1.7% | 0.4% | 0.4% |
| 181 | 2.0% | 0.3% | 0.4% |
| 216 | 1.2% | 0.1% | 0.7% |

**结论**：全程稳定，无系统性退化。偶有极端单样本事件（step 99 出现连续 2015 个 '5'、step 197 出现连续 1680 个 '0'），但属个别离群样本而非趋势性恶化。gibberish 稳定在 0.1~0.8%，为 SFT 模型固有的少量低质量输出。

#### v5-27B（69 步，冻 27B 外部教师）——三阶段退化轨迹

| step | word_rep | char_rep | gibberish | 阶段 |
|------|----------|----------|-----------|------|
| 1 | 1.2% | 0.4% | 0.0% | ① 健康 |
| 11 | 1.3% | 0.3% | 0.4% | ① 健康 |
| 21 | 2.3% | 0.8% | 0.3% | ①→② 过渡 |
| **26** | **6.4%** | 0.9% | 0.3% | ② 短语循环爆发 |
| **31** | **9.8%** | 0.9% | 0.1% | ② 短语循环峰值 |
| **41** | **9.5%** | 0.1% | 0.1% | ② 短语循环峰值 |
| 46 | 8.9% | 0.5% | **1.4%** | ②→③ 过渡 |
| 56 | 1.8% | 1.4% | **1.8%** | ③ 乱码接管 |
| 61 | 1.3% | 2.6% | **3.1%** | ③ 字符崩塌 |
| 66 | 0.9% | **4.4%** | **3.4%** | ③ 字符崩塌 |
| 69 | 0.3% | **6.1%** | **4.3%** | ③ 彻底崩塌 |

**三阶段退化轨迹**：

- **阶段①（step 1~21）健康期**：word_rep 1~2%，char_rep < 1%，gibberish < 0.5%。对应 TensorBoard loss 0.48→0.22 平稳下降。
- **阶段②（step 26~51）短语循环爆发**：word_rep 从 2.3% 跳涨到 9.8%，75/768 条样本出现严重短语重复。单条样本最高重复同一短语 **139 次**（`"Вот подробный анализ предоставленной скриншота социальной сети"`）。对应 TensorBoard bias_correction 爆炸（0.012→0.204）和 rollout_ppl 飙升（5→915）。**模型进入 phrase-looping mode**：不断重复同一句式而非生成有效摘要。
- **阶段③（step 56~69）字符级崩塌**：word_rep 反而下降（1.8%→0.3%），**但这不是好转**——模型退化到连完整短语都无法循环，转为字符级 stuttering（连续重复同一字符 10~65 次）和乱码生成。char_rep 从 1.4% 暴涨到 6.1%，gibberish 从 1.8% 涨到 4.3%。

> **word_rep 下降 + char_rep/gibberish 上升 = 模型从"能循环短语"退化为"只能吐乱码"**，这是比短语循环更严重的崩塌。

### 8.4 乱码样本示例

**v5-27B step 46（阶段②→③ 过渡）**：
```
Cette image montre une curseité, générée numotique,
unhorp les_toptori tr_à_à_à_à_à_à_à_à_à_à_à_à_à_à_à...
```
→ 法语摘要框架尚在，但内容已退化为无意义 token 串和重复的 `à`。

**v5-27B step 69（阶段③）**：
```
The image displays a smartphone navigation interface map
showing an outdoor route to a destination located at the
North-East of the "mnah-m. South Refugee Mush. The
distancejazzzm houses attained at the northern-House...
```
→ 英文摘要结构残存，但充斥 `mnah-m`、`distancejazzzm` 等生造词。

**v4-fixed step 1（离群样本，非系统性）**：
```
Summary:The image displaysleftrightarrow4extendsFaConn[;
dialogue^^contains является--------------------------------
```
→ 中英俄混排 + 符号堆叠，单条离群，不影响整体稳定性判断。

### 8.5 与 TensorBoard 指标的交叉验证

| rollout 文本检测信号 | 对应 TensorBoard 指标 | 一致性 |
|----------------------|----------------------|--------|
| word_rep 爆发（step 26~51） | bias_correction 爆炸（step 46+）、rollout_ppl 飙升 | ✅ 完全一致 |
| char_rep/gibberish 爆发（step 56~69） | rollout_ppl 915（极高）、response_length 波动 | ✅ 一致 |
| v3-no-ema/v4 无退化 | bias_correction 稳定 < 0.02、rollout_ppl < 6 | ✅ 完全一致 |

**代理指标用法**（无 rollout 文本时可从 TensorBoard 推断）：

| TensorBoard 指标 | 重复/乱码含义 |
|-------------------|---------------|
| `response_length/mean` 持续增大 | 可能短语循环导致输出变长 |
| `clip_ratio` > 0 | 有样本达到 max_length 截断 = 极端重复 |
| `rollout_ppl` 飙升 | 生成低概率/无意义 token = 乱码或循环 |
| `rollout_ppl` > 100 | 几乎确定有严重乱码 |

> v5-27B 的 rollout_ppl 从 5 飙到 915，与 word_rep 9.8% + char_rep 6.1% + gibberish 4.3% 完全吻合，三者交叉验证形成闭环。

### 8.6 检测脚本

检测脚本位于 `RP-OPSD/scripts/detect_rollout_degradation.py`，用法：

```bash
# 检测单个实验
python scripts/detect_rollout_degradation.py \
    --rollout-dir outputs/flashnote_train_v5_teacher27B/rollouts \
    --sample-every 5

# 四实验对比
for exp in flashnote_train_v3 flashnote_train_v3_no_ema \
           flashnote_train_v4_fixed_teacher flashnote_train_v5_teacher27B; do
    python scripts/detect_rollout_degradation.py \
        --rollout-dir outputs/$exp/rollouts --sample-every 10
done
```

输出格式：`step  n  word_rep(%)  char_rep(%)  gibberish(%)`，并打印极端样本预览。

---

## §9 修复方案：让 27B→9B 不再发散

### 9.1 代码约束：不能只改 ALPHA

分析代码后发现一个**关键约束**（`core_algos.py:1133-1203`）：

```python
if distillation_objective == "mopd_topk_reverse_kl":
    # ← 纯反向 KL，不支持 tail bucket（line 1135 报错）
    # ← 且 line 1203 assert alpha == 1.0
    reverse_kl_term = student_probs * (student_logp - teacher_logp)
    bias_correction = teacher_probs - student_probs
    raw_loss = (reverse_kl_term + bias_correction).sum(-1)

elif not (use_topk and objective == "mopd_topk_reverse_kl"):
    # ← generalized JSD 分支，支持 alpha=0.0/0.5/1.0
    if alpha == 0.0:   # 前向 KL
    elif alpha == 1.0: # 反向 KL（renorm/tail 版）
    else:              # 广义 JSD = lerp(kl_student, kl_teacher, alpha)
```

**四个实验当前配置全都是** `DISTILLATION_OBJECTIVE="mopd_topk_reverse_kl"` + `DISTILLATION_ADD_TAIL=False` + `ALPHA=1.0`，即纯反向 KL、无 tail bucket、无 renorm。

这意味着：
- ❌ **只改 `ALPHA=0.5` 不行**——`mopd_topk_reverse_kl` 分支会触发 `assert alpha == 1.0`
- ❌ **只开 `add_tail=True` 也不行**——`mopd_topk_reverse_kl` 会报 `does not support a tail bucket`
- ✅ **必须换 `DISTILLATION_OBJECTIVE="generalized_jsd"`**，才能走 JSD 分支 + tail bucket

### 9.2 推荐方案：三改 + 一前置

> **实证支撑**：dialog_title 任务的 OPD（27B→4B，跨度更大）用 `swift rlhf --rlhf_type gkd --beta 0.5` 稳定收敛，其 `beta=0.5` = 标准 JSD。该任务同样是跨尺寸蒸馏，同样无 tail bucket（但对 top-k 做 renorm），却稳定收敛——**关键差异就是 loss 类型（JSD vs 纯反向 KL）**，直接验证了下面的推荐。

| 改动 | 当前值 | 推荐值 | 针对的根因层面 | 理由 |
|------|--------|--------|---------------|------|
| **OBJECTIVE** | `mopd_topk_reverse_kl` | `generalized_jsd` | 层面③+④ | 换掉 mode-seeking 反向 KL；JSD 梯度有界（上界 log2），不易爆炸。title 任务已实证有效 |
| **ADD_TAIL** | `False` | `False`（走 renorm）或 `True` | 层面④ | 见下方 9.2.1 详解：renorm 和 tail 两种截断处理都能消除假收敛温床；title 用 renorm 也成功 |
| **ALPHA** | `1.0` | `0.5` | 层面③ | 广义 JSD = 0.5×FKL + 0.5×RKL；FKL 是 mean-seeking（覆盖全部 mode），不会逼学生跳到单一 mode 上爆炸。等价于 swift GKD `beta=0.5` |
| **LR** | `2e-6` | `5e-7` | 层面③ | 降 4 倍，减缓 JSD 推力，给学生更多时间在自然分布附近微调 |
| **WARMUP** | `75` | `200` | 层面⑤ | IS 权重更充分稳定，防止 ppl_ratio 早期波动触发恶性正反馈 |

#### 9.2.1 ADD_TAIL 详解：renorm vs tail bucket

代码 `core_algos.py:1153-1158` 中，`generalized_jsd` + top-k 有两条路径：

```python
if self_distillation_config.distillation_add_tail:
    # 路径 A：tail bucket = top-k 外概率装进第 101 维
    student_distill_log_probs = add_tail(student_log_probs)  # [K+1] 维
    teacher_distill_log_probs = add_tail(teacher_log_probs)  # [K+1] 维
else:
    # 路径 B：renorm = top-k 内重归一化到 sum=1
    student_distill_log_probs = renorm_topk_log_probs(student_log_probs)  # [K] 维，sum=1
    teacher_distill_log_probs = renorm_topk_log_probs(teacher_log_probs)  # [K] 维，sum=1
```

| | 路径 A：tail bucket（ADD_TAIL=True） | 路径 B：renorm（ADD_TAIL=False） |
|---|---|---|
| 维度 | top-k + 1（第 101 维 = 尾部） | top-k（重归一化） |
| 泄漏可见性 | ✅ 尾部维度直接捕获 top-k 外概率 | ✅ renorm 放大剩余概率，形状差异仍被 JSD 捕获 |
| title 任务用哪种 | — | ✅ swift GKD 用的是这条路径（log_softmax on top-k） |
| 推荐 | 更精确（保留 mass 信息） | 更简单，已有实证成功 |

> **结论**：优先 `ADD_TAIL=False`（走 renorm），因为 title 任务已证明 renorm + JSD 足够稳定。如果 renorm 仍有问题再试 `ADD_TAIL=True`。

**一前置**：学生起点从 base 9B 换成 **SFT 9B**（层面①）

> v4 已证明 SFT 学生 + 冻结 9B SFT 教师稳定收敛（loss 0.126，bias_correction 稳定 0.010）。v5 用 base 9B 学生起点，初始 raw_jsd 0.494 太高。换 SFT 学生后预计初始 raw_jsd 降到 ~0.15~0.20（SFT 后的 9B 离 27B 输出空间更近），从源头缩小分布差距。

### 9.3 推荐配置（基于 v5 脚本修改）

```bash
# ===== 改这 5 行 =====
MODEL_PATH="<SFT_9B_ckpt_path>"                    # base 9B → SFT 9B
ALPHA=0.5                                           # 1.0 → 0.5 (广义 JSD，等价 swift GKD beta=0.5)
DISTILLATION_OBJECTIVE="generalized_jsd"             # mopd_topk_reverse_kl → generalized_jsd
DISTILLATION_ADD_TAIL=False                         # 保持 False（走 renorm，与 title 任务一致）
LR=5e-7                                             # 2e-6 → 5e-7

# ===== 改这 1 行 =====
LR_WARMUP_STEPS=200                                 # 75 → 200

# ===== 不变的 =====
TEACHER_MODEL_PATH="/data4/wumeimei/download_models/Qwen3.8-27B"
TEACHER_MODEL_SOURCE="fixed"
TEACHER_UPDATE_RATE=0.0
DISTILLATION_TOPK=100
DISTILLATION_TOPK_SOURCE="teacher"
TRAIN_BATCH_SIZE=96
```

### 9.4 为什么 JSD + tail bucket 能修

**三个机制对比**：

```
当前（mopd_reverse_kl + no_tail）:
  loss = Σ_{v∈top100} p_s·log(p_s/p_t) + Σ_{v∈top100} (p_t - p_s)
                    ↑ 只看 top-k 内                ↑ 暴露 gap 但不惩罚泄漏
  学生概率泄漏到 top-k 外 → reverse_kl 看不见 → 假降
  bias_correction 看得见但只报警不惩罚 → 仍然发散

推荐（generalized_jsd + tail + alpha=0.5）:
  loss = 0.5×FKL(top101) + 0.5×RKL(top101)
  其中 top101 = [top-100 logp, tail_logp]
  tail = log(1 - Σ p(v∈top100))  ← 第 101 维

  ① FKL = Σ p_t·log(p_t/p_s)：教师 top-k 内 + tail 都参与
     → 学生概率泄漏到 tail → FKL 直接增大 → 惩罚泄漏 → 消除假收敛
  ② RKL = Σ p_s·log(p_s/p_t)：学生 top-k 内 + tail 都参与
     → 但有 FKL 平衡，不会单方向 mode-seeking
  ③ JSD 梯度有界（≤ log2 ≈ 0.693），不会像纯 RKL 那样在分布差异大处爆炸
```

**用指标语言说**：

| 指标 | 当前（v5 纯 RKL） | 预期（JSD+tail） |
|------|-------------------|------------------|
| reverse_kl_term | 假降（概率泄漏不可见） | 不再有此项（JSD 不分离） |
| bias_correction | 爆炸 0.204（只报警不惩罚） | 不再有此项（tail 直接惩罚泄漏） |
| student_topk_mass | 暴跌到 0.603 | 预计稳定 > 0.95（tail 惩罚会拉回泄漏的概率） |
| raw_jsd (loss) | 0.494→0.263 反弹 | 预计单调下降（梯度有界+泄漏可见） |
| rollout_ppl | 爆炸到 915 | 预计稳定 < 20 |

### 9.5 监控判据

训练时盯以下三个信号，**任一触发即应暂停降 LR 或回退 ckpt**：

| 信号 | 阈值 | 含义 | 检查频率 |
|------|------|------|----------|
| `bias_correction` > 0.05 | (若仍存在) | 学生开始脱离教师支撑域 | 每 5 步 |
| `student_topk_mass` < 0.90 | | 概率泄漏超过 10% | 每 5 步 |
| `rollout_ppl` > 50 | | rollout 生成低质量 token | 每步 |
| `raw_jsd` 连续 10 步不降 | | 训练停滞 | 每 10 步 |
| rollout `word_rep` > 3% | | 短语循环苗头 | 每 10 步（跑检测脚本） |

> **bias_correction 是最早暴露发散的信号**：v5 在 step 45 时 bias_correction 已开始升（0.012→0.016），而 loss 反弹到 step 46 才出现——bias 比 loss 早 1 步。切换到 JSD 后 bias_correction 可能不再作为独立 tag 出现，此时改盯 `student_topk_mass` 和 `rollout_ppl`。

### 9.6 实验计划

| 实验 | 学生起点 | 教师目标 | LR | ALPHA | TAIL | 预期 |
|------|---------|----------|-----|-------|------|------|
| v5（已跑） | base 9B | RKL 1.0 | 2e-6 | 1.0 | False(raw) | ❌ 延迟发散 |
| **v6（推荐）** | **SFT 9B** | **JSD 0.5** | **5e-7** | **0.5** | **False(renorm)** | ✅ 预期稳定 |
| v6-ablation-A | SFT 9B | RKL 1.0 | 5e-7 | 1.0 | False(raw) | 验证 SFT warm-start 单独效果 |
| v6-ablation-B | base 9B | JSD 0.5 | 2e-6 | 0.5 | False(renorm) | 验证 JSD+renorm 单独效果 |

> **v6 是主推实验**，三改+一前置一起上。两个 ablation 作为备选——如果 v6 稳定收敛，就不需要跑 ablation；如果 v6 仍有问题，ablation A/B 可以定位是 SFT warm-start 还是 JSD 起了主要作用。预计 v6 训练时间与 v5 相同（~21h，72k 样本 × 2 epoch）。

### 9.7 实证参照：dialog_title OPD（27B→4B，JSD，已成功）

dialog_title 任务的 OPD 脚本（`/data4/wumeimei/dialog_title/train_script/opsd.sh`）配置：

```bash
swift rlhf \
    --rlhf_type gkd \          # Generalized Knowledge Distillation
    --model Qwen3.5-4B \        # 学生：4B（比 9B 更小）
    --teacher_model_server http://127.0.0.1:8000 \  # 教师：Qwen3.8-27B（通过 vLLM API）
    --gkd_logits_topk 64 \      # top-64（比 v5 的 100 更少）
    --beta 0.5 \                # ← 标准 JSD（等价 verl ALPHA=0.5）
    --lmbda 1.0 \               # 100% on-policy（等价 verl on-policy rollout）
    --sft_alpha 0 \             # 纯蒸馏，不加 SFT loss
    --temperature 1.0 \         # 无温度缩放
    --lora_rank 32 \             # LoRA（v5 是 full fine-tune）
    --learning_rate 2e-5 \       # LR 比 v5 高 10×（LoRA 需要更高 LR）
    --max_completion_length 64   # 标题任务输出短（64 token）
```

**swift GKD beta 参数公式**（源码 `ms-swift/swift/rlhf_trainers/gkd_loss.py:94-148`）：

```
beta=0   → Forward KL:  KL(p_T || p_S)   = Σ p_t·log(p_t/p_s)  （mean-seeking）
beta=1   → Reverse KL:  KL(p_S || p_T)   = Σ p_s·log(p_s/p_t)  （mode-seeking）
beta=0.5 → 标准 JSD: 0.5·KL(M||T) + 0.5·KL(M||S), M = 0.5·S + 0.5·T  （对称有界）
```

**与 v5 的逐项对比**：

| 维度 | title OPD（✅ 成功） | flash_note v5（❌ 发散） |
|------|---------------------|--------------------------|
| 跨尺寸 | 27B→**4B**（差距更大） | 27B→9B |
| **Loss 类型** | **JSD (beta=0.5)** | **Reverse KL (alpha=1.0)** |
| 截断处理 | **renorm**（log_softmax on top-k） | **raw probs**（不 renorm 不 tail） |
| 框架 | swift GKD | verl RP-OPSD（mopd_topk_reverse_kl） |
| 训练方式 | LoRA r=32 | Full fine-tune |
| LR | 2e-5（LoRA 需要高 LR） | 2e-6 |
| 输出长度 | 64 token（标题短） | ~512 token（摘要长） |
| 教师部署 | vLLM API（http） | 框架内置（fixed model） |

**结论**：title 任务跨尺寸更大、top-k 更少，但用 JSD 成功了。v5 失败的唯一关键变量是 **loss 类型**（反向 KL vs JSD）。这直接验证了 §9.2 推荐：换 `generalized_jsd` + `ALPHA=0.5`。

> **swift GKD 的 top-k renorm 细节**：`gkd_loss.py:135-136` 对 top-k logits 做 `log_softmax`，等价于在 top-k 子集上重归一化。丢弃的词表其余 token 概率被忽略（无 tail bucket）。renorm 使得学生无法通过"把概率移出 top-k"让 loss 虚降——移出后剩余概率被 inflate，形状差异仍被 JSD 捕获。verl 的 `generalized_jsd` + `ADD_TAIL=False`（路径 B: `renorm_topk_log_probs`）走的是完全相同的逻辑。

### 9.8 风险与备选

| 风险 | 应对 |
|------|------|
| JSD renorm 显存（logsumexp + mixture 分布 M） | 先跑 10 步冒烟，监控显存；不够则降 `PPO_MICRO_BATCH_SIZE_PER_GPU` 或减 `DISTILLATION_TOPK` 到 64（与 title 一致） |
| SFT 9B + 27B 教师初始 raw_jsd 仍偏高 | 进一步用 27B 教师离线生成摘要做 SFT（off-policy 先对齐再 on-policy 精修） |
| JSD 收敛后 loss plateau 比 RKL 高 | 可接受——稳定的高 plateau 优于看似低但假收敛/发散的 RKL |
| 训练 200 步 warmup 太长（200/1502≈13%） | 72k×2=1502 步，200 步 warmup 占 13%，合理；v5 的 75 步 warmup 仅占 5%，IS 权重未充分稳定 |
| full fine-tune 比 LoRA 容易过拟合 | title 用 LoRA r=32 稳定；v6 是 full fine-tune，可考虑加 LoRA 或降 LR 到 3e-7 |

> **终极备选**：如果 v6 仍发散，最后退路是放弃 27B→9B 在线蒸馏，回到已验证稳定的 v3-no-ema（9B←9B）——该实验已证明真实收敛（loss 0.061、ppl_ratio 1.14、rollout 无退化）。27B 教师的价值通过 **离线 SFT**（用 27B 生成摘要数据做 SFT，等价 title 任务的 `lmbda=0` 模式）来兑现，而非在线蒸馏。

---

## §10 失效时的处理顺序：训练前预检清单

v5 发散花了 67 步约 40 分钟才暴露。以下预检清单的目的是**在训练开始前**（0 步）就把会导致发散的问题排查掉，避免浪费 21h 算力。按因果链从最底层往上排，**必须严格按顺序排查**——底层不一致会让上层检查变得毫无意义。

### 第 0 层：tokenizer + chat template + 多模态 processor 完全一致

**为什么最先查**：如果师生 tokenizer 不一致，logit 的 token index 错位，整个 KL 计算的就是两个无关分布之间的散度——loss 看起来在降，实际是噪声。这是**一切的地基**，后面所有检查都假设这一层通过。

| 检查项 | 方法 | 通过标准 | v5 状态 |
|--------|------|----------|---------|
| tokenizer 一致 | `assert teacher.tokenizer.get_vocab() == student.tokenizer.get_vocab()` | vocab 完全相同（同 token → 同 id） | ✅ 同族 Qwen3.5，vocab 65536 一致 |
| chat template 一致 | 对同一 prompt 跑 `teacher.apply_chat_template()` vs `student.apply_chat_template()` | 输出完全一致 | ✅ v5 用同一 `perception_chat_template_qwen35.jinja` |
| 多模态 processor 一致 | 对同一图片跑 `processor(images=...)` | image_token_id、num_image_tokens 完全一致 | ✅ 同架构 `Qwen3_5ForConditionalGeneration`，image_token_id 一致 |
| special token id 一致 | `eos_token_id`, `bos_token_id`, `pad_token_id`, `image_token_id` | 全部相同 | ✅ eos=7, bos=1, pad=0, image=151646 一致 |

> **v5 这一层是过的**——架构兼容性已确认（§3.4 "同为 Qwen3_5ForConditionalGeneration，vocab/eos/image_token_id 一致"）。但跨族蒸馏（如 Qwen→LLaMA）必须查这层。

### 第 1 层：训练前 per-token KL（初始分布差距）

**为什么第二查**：§Q12 已证明初始 raw_jsd 0.494（接近 ln2 上界）是发散的种子。如果训练前 KL 就很高，说明师生分布差异太大，纯反向 KL 注定发散——此时应换 JSD 或做 SFT warm-start，**不要开训练**。

| 检查项 | 方法 | 通过标准 | v5 实际值 |
|--------|------|----------|-----------|
| 训练前 per-token RKL | 取 100 条训练样本，师生各跑一次前向，算 `KL(p_S‖p_T)` 逐 token 均值 | < 0.20 = 安全；0.20~0.40 = 需 JSD；> 0.40 = 必须先 SFT warm-start | **0.494** ❌ |
| 训练前 per-token FKL | 同上，算 `KL(p_T‖p_S)` | 对照看 FKL/RKL 比值，比值大说明分布形状差异大 | （未单独记录，但 JSD ≈ RKL 说明 FKL 也很高） |
| 训练前 JSD | 算 `0.5×FKL + 0.5×RKL`（alpha=0.5） | < 0.15 = 可用纯 RKL；> 0.15 = 应降级到 JSD | **0.494**（与 RKL 相近，说明 FKL ≈ RKL，分布近乎正交）|

> **v5 在这层就不该开训练**：初始 raw_jsd 0.494 已亮红灯，但当时没有预检机制，直接跑了 67 步才暴露。

### 第 2 层：top-probability overlap（师生 top-k 重叠度）

**为什么第三查**：即使初始 KL 不算极端，如果师生 top-k token 的**集合重叠率**很低，反向 KL 的 mode-seeking 会逼学生跳到它不认识的 token 上 → 支撑域外 collapse。

| 检查项 | 方法 | 通过标准 | v5 实际值 |
|--------|------|----------|-----------|
| top-k 集合重叠率 | 每个位置取师生各 top-100，算 `|S∩T| / 100` 的均值 | > 80% = 安全；50~80% = 需 JSD；< 50% = 必须先 SFT | （未直接记录，但 student_topk_mass=0.974 说明初始重叠看似高，是假象——见 Q12 层面二） |
| top-1 一致率 | 师生 argmax 一致的比例 | > 70% = 安全 | （未记录） |
| student_on_teacher_topk_mass | 学生概率质量落在教师 top-k 上的占比 | > 0.95 = 安全 | 0.974（初始看似安全，后续暴跌到 0.603） |

> **注意**：初始 student_topk_mass=0.974 是**表面安全**——因为 top-k 是教师选的，学生本来就有一定概率在这些 token 上。真正的问题是训练后学生概率移出 top-k（§Q12 层面二）。所以这个检查只能排除"一开始就不对齐"的情况，训练中的动态偏移要靠 §9.5 监控判据。

### 第 3 层：学生熵（分布锐度）

**为什么第四查**：学生分布太尖锐（低熵）→ 概率集中在少数 token 上 → 反向 KL 推力容易把它们推到极端 → mode collapse。学生分布太平坦（高熵）→ 概率分散 → 蒸馏效率低但不容易 collapse。SFT 模型通常更尖锐（v4 的 raw_jsd 起步比 v3-no-ema 低但 plateau 更高）。

| 检查项 | 方法 | 通过标准 | 说明 |
|--------|------|----------|------|
| 学生 per-token 熵 | `H(p_S) = -Σ p_s·log(p_s)` 逐 token 均值 | 与教师熵 `H(p_T)` 比较：差距 < 0.5 nat = 安全 | v5 未单独记录，但 SFT 模型熵更低（v4 §3.3 "SFT 起点分布更尖锐"） |
| 师生熵差 | `|H(p_S) - H(p_T)|` | < 0.5 nat | 差距大说明一个分布尖锐一个平坦，蒸馏困难 |
| 学生 top-1 概率均值 | `mean(max(p_S))` | 0.3~0.7 = 正常；> 0.9 = 过度自信（SFT 常见） | — |

> **v5 的 base 9B 熵比 SFT 9B 高**（base 分布更平坦），但 27B 教师熵可能更低（更强模型更自信）→ 师生熵差大 → 蒸馏困难。v4 的 SFT 学生熵更低 → 离 27B 教师更近 → 初始 raw_jsd 更低。这解释了为什么 SFT warm-start 能缩小初始差距。

### 第 4 层：on-policy 假设基线（IS 权重 / ppl_ratio）

**为什么第五查**：训练还没开始，但可以用训练前的模型跑一次 rollout → 算 ppl_ratio 基线。如果基线就很高，说明 vLLM BF16 和 FSDP FP32 之间精度差太大，IS 权重会从一开始就不稳定。

| 检查项 | 方法 | 通过标准 | v5 实际值 |
|--------|------|----------|-----------|
| ppl_ratio 基线 | 训练前跑 1 步 rollout + 1 步 train forward，算 ppl_ratio | < 2.0 = 安全；2~10 = 需增大 warmup；> 10 = BF16/FP32 精度问题 | 1.26（初始安全，训练后爆炸到 195） |
| rollout_ppl 基线 | 训练前 rollout 的 perplexity | < 20 = 模型能生成合理 token | 5（初始安全，训练后爆炸到 915） |

> v5 这层初始是通过的（ppl_ratio=1.26），问题出在训练中逐步恶化。但如果是 SFT 模型做学生（极端 logit），ppl_ratio 基线可能很高——这也是 v3sft 的 ppl_ratio 飙到 90 万的原因（§Q9）。

### 预检流程图

```
第 0 层 tokenizer/template/processor 一致？
  ├─ ❌ → 修对齐，不要开训练
  └─ ✅ → 第 1 层 训练前 per-token KL < 0.20？
           ├─ ✅ → 可用纯 RKL，正常开训练
           ├─ 0.20~0.40 → 换 JSD (alpha=0.5) 再开训练
           └─ > 0.40 → 先 SFT warm-start 缩小差距
              ↓
              第 2 层 SFT 后重测 KL < 0.20？
              ├─ ✅ → 开训练（JSD 或 RKL）
              └─ ❌ → 放弃在线蒸馏，走离线 SFT
```

### 预检脚本

```bash
# 训练前预检：跑 100 条样本，输出 KL/overlap/entropy 基线
python scripts/preflight_distillation_check.py \
    --student-model /path/to/student \
    --teacher-model /path/to/teacher \
    --dataset data/train.jsonl \
    --num-samples 100 \
    --topk 100
```

> 脚本输出：per-token RKL / FKL / JSD、top-k 集合重叠率、学生/教师 per-token 熵、top-1 一致率。**输出 KL > 0.40 时不建议开训练**。

---

## §11 当前运行任务续训进展（2026-09-04 更新）

> 四实验中目前**只有 v3-no-ema 仍在跑**（tmux session `rp_opsd_v3_no_ema`），其余三个（v3sft/v4-fixed/v5-27B）已停止/完成。本节基于最新 TensorBoard 数据（截至 step 425 详细表 + step 532 rollout 检测）和 tmux 实时日志更新 §3.2/§4/§7.2 的结论。

### 11.1 当前状态

| 项 | 值 |
|---|---|
| global_step | **533 / 1502**（35%） |
| 已训练时长 | 26h27m |
| 预计剩余 | ~49h |
| 最新 checkpoint | `global_step_450`（另有 150/300） |
| GPU 占用 | 8 卡 H20，7 卡 99-100%，1 卡 56% |

### 11.2 loss 依旧稳定 plateau，未发散

step 242→425 区间统计（衔接 §3.2 表格）：

| 指标 | first(242) | last(425) | mean | min | max |
|---|---|---|---|---|---|
| `vopd_loss` | 0.0601 | 0.0542 | 0.0576 | 0.0490 | 0.0682 |
| `raw_jsd_token_mean` | 0.0599 | 0.0537 | 0.0571 | 0.0486 | 0.0679 |
| `mopd_reverse_kl_term` | 0.0501 | 0.0405 | 0.0445 | 0.0343 | 0.0571 |

loss 从 273 步的 0.061 到 425 步的 0.054，**仍在 0.049-0.068 窄带内波动，没有系统性下降也没有反弹**——是 §3.2/§7.2 判断的"真实 plateau"的延续，不是新现象。

### 11.3 修正结论：grad_norm 尖峰是周期性噪声，不是发散前兆

§3.2/§7.2 对 step 242 的 grad_norm=46.9 标记为"需关注后续是否持续发散"。续训数据给出了答案：

```
step 242:24.8  259:11.8  287:32.3  312:12.3  321:13.3  326:10.9  330:21.1  361:11.2  411:18.9  424:34.8
```

尖峰**每 10~40 步复现一次**，幅度 11~35，但每次都在 1~2 步内回落到 2-5 的正常水平，loss 和 bias_correction 都没有跟着跳变。**结论：这是训练中偶发极端 IS-ratio batch 引起的噪声尖峰，是 v3-no-ema 的正常噪声模式，不是 v5 那种"尖峰后持续恶化"的发散前兆。** 可以把 §3.2/§7.2 中"需要密切关注后续"的措辞更新为"已验证为良性周期性噪声"。

### 11.4 需要更新的认知：ppl_ratio 不再是"四实验最稳定"

§3.2/§4.1/§7.2 曾把 v3-no-ema 的 ppl_ratio 描述为"全实验最稳定，稳定在 1.1-1.2，max 仅 3.6"。**这个结论只对前 242 步成立，续训后已被推翻**：

| 区间 | ppl_ratio mean | ppl_ratio max | training_ppl max | rollout_ppl max |
|---|---|---|---|---|
| step 1-242（原文档结论） | ≈1.15 | 3.6 | 6.6 | 5.7 |
| step 242-425（本次更新） | **170.8** | **1271** | **3930** | **619** |

即便如此，`vopd_loss`（0.049-0.068）和 `mopd_bias_correction`（0.009-0.017，远低于 §9.5 的 0.05 危险阈值）**全程稳定**，`student_on_teacher_topk_mass`（0.963-0.977）和 `teacher_topk_mass`（0.980-0.987）也没有像 v5 那样暴跌。说明 ppl_ratio/training_ppl 的剧烈波动是**个别 batch 的极端离群点**（IS 权重被 `clip(max=2.0)` 截断后不进入 loss，§Q8 已解释这个机制），而非整体分布漂移——诊断结论"真实收敛"不变，但"ppl_ratio 稳定"这一支撑论据需要弱化为"loss 和 bias_correction 稳定，ppl_ratio 存在离群点噪声但不影响训练"。

### 11.5 rollout 文本质量：温和漂移，非崩塌

跑 `detect_rollout_degradation.py --sample-every 20` 覆盖 step 1→532：

| step | word_rep | char_rep | gibberish |
|---|---|---|---|
| 1 | 0.8% | 0.0% | 0.0% |
| 121 | 1.6% | 0.4% | 0.4% |
| 241 | 1.6% | 0.5% | 0.1% |
| 361 | 1.2% | 0.4% | 0.5% |
| 421 | 2.3% | 0.8% | 0.1% |
| 481 | 1.4% | 0.9% | 0.3% |
| 532 | 2.5% | 0.8% | 0.8% |

三项指标都从初始的 0-0.8% 缓慢爬升到 1-2.5% 区间，**量级上仍是 v5 发散阈值（word_rep 9.8%/char_rep 6.1%/gibberish 4.3%）的 1/3~1/10**，且没有单调加速的趋势（421→481 略降，532 略升，属正常波动）。抽样发现个别输出中出现零星乱码词插入（如法语网速描述中夹带"抢劫Bank""可爱Bank"），但摘要整体结构完整，非成段崩塌。**与 §7.2 诊断一致：真实收敛、无退化，continue 监控即可，暂不需要按 §9.5 判据降 LR 或回退 checkpoint。**

### 11.6 后续建议

1. 继续跑完剩余 ~969 步（~49h），无需人工干预。
2. 若 `mopd_bias_correction` 突破 0.05 或 `student_topk_mass` 跌破 0.90（§9.5 判据），立即检查最近 checkpoint 是否需要回退。
3. 由于 loss 已在 step 240 附近进入 plateau，后续 1000+ 步大概率只是精修，可考虑在 step 750（半程）时用当前 checkpoint 跑一次人工摘要质量抽检，判断是否值得跑满 2 epoch 或提前用 750 步 ckpt 收尾。
