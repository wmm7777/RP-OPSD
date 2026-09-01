# RP-OPSD 训练失败总结

> **训练 run**：`rp_opsd_v2`（tmux 会话，2026-08-31 09:26 启动）
> **模型**：Qwen3.5-9B（裸 base 起步，无 SFT warm-start）
> **方法**：RP-OPSD（V-OPD 自蒸馏 + EMA teacher + image-swap 特权）
> **脚本**：`scripts/run_rp_opsd_v2.sh`
> **上游/官方 canonical**：`scripts/run_rp_opsd.sh` + `config/best.yaml`（作者 sansanyuchen/RP-OPSD，verl 版）
> **结论**：**失败，已于 step ~471 终止**。loss 平滑下降但输出质量持续退化。
> **根因**：**配置漏传导致走错 loss 分支**——本地实际跑的是默认 `generalized_jsd`（forward JSD + student 选 support + tail bucket），而非论文的 `mopd_topk_reverse_kl`（bias-corrected reverse-KL + teacher 选 support + no tail）。叠加任务偏差（开放式摘要 vs 官方 MCQ）与步数过长放大。

---

## 一、现象

### 1.1 评测结果（step450 checkpoint）

| 语种 | Summary MOS 均分 | badcase 率（任一维度<3） | 重复提及次数 |
|------|-----------------|--------------------------|--------------|
| en | **2.84** | **67.5%**（139/206） | 376/1030 维度 |
| fr | 3.11 | 高 | 264/960 维度 |
| ru | 待评 | — | — |
| zh | 待评 | — | — |

参考基线：SFT-9B epoch2.0 四语种均 **3.95-3.96**。step450 en=2.84 远低于 SFT 最优，badcase 率 67.5% 说明过半样本不达标。

### 1.2 评测中的退化特征

评测理由反复出现：
- **重复生成**：将同一条信息"各自重复三次"，属于对信息频率的错误描述
- **语言混乱**：中文摘要里混入希伯来语（לבחור מנוע）、阿拉伯语（المج木雕）、英语片段
- **instruction leaking**：输出里出现"I have summarized the available information…"这类自我解释的元话语
- **长度膨胀**：摘要越写越长，大量冗余重复

### 1.3 训练 rollout 退化趋势（on-policy 生成，随训练步数）

| step | 重复率 | 平均字符数 | 说明 |
|------|--------|-----------|------|
| 123（最早） | 7% | 1486 | 早期输出基本正常（如拨号图标描述合理） |
| 175 | 7% | 1724 | 退化尚未启动 |
| 300 | 13% | 1950 | 退化加速 |
| 450 | **23%** | 2257 | 严重退化 |
| 498 | 17% | 2067 | — |

**退化随步数单调加速**：重复率 7%→23%，字符数膨胀 50%。前 175 步稳定在 5-8%，175 步后正反馈启动，一路恶化。

### 1.4 Loss 曲线（具有欺骗性）

| step | vopd_loss | jsd_token | corr_kl | grad_norm |
|------|-----------|----------|---------|-----------|
| 20 | 0.104 | 0.103 | 0.144 | 5.09 |
| 200 | 0.038 | 0.037 | 0.045 | 5.00 |
| 471 | 0.029 | 0.028 | 0.028 | 3.23 |

vopd_loss 从 0.123 平滑下降到 0.026，grad_norm 稳定无爆炸。**单看 loss 完全像健康收敛，但实际输出在崩坏**——这是本次失败最关键的教训。

---

## 二、问题定位分析

### 2.1 代码来源：本地 vs 官方 canonical

本地 `RP-OPSD/` 是作者 sansanyuchen/RP-OPSD（verl 版）的 clone，**不是** NJUNLP/RP-OPSD（纯文本 src/ 版，带 gate-gated dual-KL + p_ref）。两者的 verl 实现即本仓库上游。官方 canonical recipe 落在两个文件：
- `config/best.yaml` — canonical 设置汇总
- `scripts/run_rp_opsd.sh` — 训练启动脚本（`run_rp_opsd.bak.sh` 是底层 hydra 调用）

官方任务 = **Dataset4.0 = 5205 MCQ + 90 openqa**（视觉理解多选题），benchmark 全是 vstar/hrbench/mmvp/mmstar/pope；`max_response_length=1024`，`total_steps=55`，`warmup=10`。本地任务 = flash_note 通知摘要（开放式长文本生成），`max_response=2048`，`1502 步`，`warmup=75`。

### 2.2 根因：漏传 4 个 distillation 配置，走错 loss 分支（真 bug）

`verl/workers/config/actor.py` 与 `verl/trainer/config/actor/actor.yaml` 的 self_distillation 默认值 vs 官方 canonical vs 本地 v2 实际：

| 配置项 | verl 默认 | 官方 canonical | 本地 v2（未传→默认） |
|--------|----------|---------------|-------------------|
| `distillation_objective` | `generalized_jsd` | `mopd_topk_reverse_kl` | ❌ **generalized_jsd** |
| `distillation_topk_source` | `student` | `teacher` | ❌ **student** |
| `distillation_add_tail` | `True` | `False` | ❌ **True** |
| `full_logit_distillation` | `True` | `True` | ✅ True |

本地 `run_rp_opsd_v2.sh` 第 141-156 行只传了 `distillation_topk`、`is_clip`、`teacher_*`、`alpha`、`dont_reprompt_on_self_success`，**漏传了上面 4 项中的前 3 项**，于是静默走默认值。`actor.py:161-171` 有 validation：`mopd_topk_reverse_kl` 要求 `topk_source=teacher`+`add_tail=False`+`alpha=1.0`，但本地因 objective 走默认 `generalized_jsd`，这些约束**不触发、不报错**——悄悄进了错分支。

**运行时铁证（tensorboard metric）**：解析全部 9B event 文件，`self_distillation/` 下只有 `raw_jsd_token_mean`、`weighted_jsd_token_mean`、`raw_distillation_token_mean`、`num_distill_tokens` 等，**完全没有** `mopd_reverse_kl_term_mean`、`mopd_bias_correction_mean`、`teacher_topk_mass_mean`、`student_on_teacher_topk_mass_mean`。core_algos.py:1147-1151 这几个 metric 只在 `mopd_topk_reverse_kl` 分支（第 1133-1146 行）输出——它们缺席即证明训练实际走了 `generalized_jsd` 分支（第 1153+ 行）。这不是脚本静态推断，是训练产物的铁证。

`verl/trainer/ppo/core_algos.py:1085` `compute_self_distillation_loss` 的分支证实：
- `objective==mopd_topk_reverse_kl`（第 1133-1146 行）：`reverse_kl_term = p_s·log(p_s/p_t)`，`bias_correction = p_t − p_s`，`loss = Σ_topK(p_t)[p_s·log(p_s/p_t) − p_s + p_t]` —— 正是 issue 里作者贴的 MOPD Eq.(5)。
- `objective==generalized_jsd + add_tail=True`（第 1153-1155 行 + 1165+ 行）：student/teacher 各加 tail bucket 归一化，走 alpha-interpolated forward KL/JSD —— **本地实际走的分支**。

issue 作者原话："Some internal configuration names such as vopd and mopd_topk_reverse_kl are retained for compatibility, but the configuration activated by scripts/train.sh is the RP-OPSD recipe reported in the paper." —— `loss_mode=vopd` 只是开启自蒸馏的兼容开关，真正激活论文方法的是 `distillation_objective=mopd_topk_reverse_kl`。

### 2.3 退化机理：generalized_jsd 正反馈 vs mopd_reverse_kl 抗退化

本地实际跑的（generalized_jsd + topk_source=student + add_tail=True）退化正反馈链：

1. `topk_source=student` → **student 自己选 top-k 蒸馏 support**。student 退化（重复 token）时，它选的 top-k 就是退化 support。
2. `generalized_jsd`（forward JSD）→ mode-covering，student 倾向覆盖 teacher 在退化 support 上的分布。
3. EMA teacher（半衰期 ~14 步）在 student 选的退化 support 上 logits 跟着退化。
4. → student 选退化 support → teacher 在退化 support 上退化 → forward JSD 让 student 匹配退化 teacher → 正反馈 → 重复/语言混杂。loss 下降因为两者**一起坍缩到低熵重复分布**，一致性在升。

论文 mopd_topk_reverse_kl + topk_source=teacher + add_tail=False 抗退化：
1. `topk_source=teacher` → **EMA teacher 选 support**。teacher 慢变，top-k 由早期多样分布决定，student 退化改变不了蒸馏 support。
2. `reverse-KL`（mode-seeking）→ student 必须在 teacher 每个高概率 mode 上分配概率，漏掉任何 mode 被惩罚，无法坍缩到单一重复 mode。
3. `add_tail=False + bias_correction(p_t−p_s)` → 保留 teacher top-k 上原始 full-softmax 概率，不归一化、不补 tail。
4. teacher 多样 support + reverse-KL mode-seeking = 双重抗退化。

forward JSD（mode-covering，易坍缩低熵）与 reverse-KL（mode-seeking，强制覆盖多样 mode）的数学性质差异，是本地退化而论文不退化的根本机制。

### 2.4 放大因素：任务偏差 + 步数过长 + 裸 base 起步

即便走对 loss 分支，以下偏差仍会放大退化风险（在走错分支下雪上加霜）：

- **任务偏差**：官方是 MCQ/VQA（5205 选择题 + 90 openqa），答案空间小、有客观对错，退化易暴露且 teacher top-k 天然多样；本地是开放式摘要，输出空间巨大、无客观对错，重复/语言混杂是 self-reinforcing 的退化模式。论文方法在 MCQ 上验证，不能无改动迁移到开放式长文本生成。
- **步数过长**：官方 `total_steps=55`（~4 个 EMA 半衰期），teacher 还没充分跟随 student 漂远就结束；本地 `1502 步`（~107 个半衰期），EMA teacher 完全贴着 student，任何退化被无限放大。
- **裸 base 起步无 warm-start**：`MODEL_PATH=Qwen3.5-9B` 是未 SFT 的 base，初始 rollout 质量不稳，退化起点高（官方 canonical 也用 base，但 55 步 + MCQ 任务下退化来不及启动）。
- **max_response 1024→2048**：给退化更多生成空间，重复/膨胀有地可施。
- **image-swap 特权**：teacher 看 bbox 特权图、student 看全图，视角差异在走错 loss（无质量约束）时放大 rollout 方差，更易踩中退化模式。

### 2.5 为什么 loss 下降 ≠ 质量提升

| loss 衡量的 | 实际发生的 |
|------------|-----------|
| student 与 teacher 的分布一致性 | 两者一起坍缩到退化分布（重复、混乱） |
| 收敛方向 | 收敛到一个**退化解**，不是好的摘要分布 |
| grad_norm 稳定 | 优化过程稳定，但优化目标是错的 |

### 2.6 `training_ppl` 尖峰的误读

训练中 `rollout_corr/training_ppl` 间歇飙到 1000-15452（前半段 mean=1174、后半段 mean=874）。这**不是**"师生分歧的正常副产物"，而是退化输出（重复 token、混乱语言）在分布上概率异常的真实信号。前半段尖峰频率高、后半段中位数从 524 降到 127 看似"收敛"，实则是 student 和 teacher **一起退化到低熵重复分布**后分歧反而变小——退化的稳态。

### 2.7 方法论教训

1. **诊断训练健康度不能只看 loss**。本次 loss、grad_norm、lr warmup、蒸馏 mask 覆盖率（100%）、teacher_always_on（100%）、aborted_ratio（0）全部"正常"，但输出退化。**必须以 rollout 样本质量 / 评测 MOS 为准**，loss 只是必要非充分条件。
2. **移植方法必须逐项核对 canonical 配置**。`loss_mode=vopd` 只是兼容开关，真正激活论文方法的是 `distillation_objective=mopd_topk_reverse_kl` + `topk_source=teacher` + `add_tail=False`。漏传这几项 verl 不报错（走默认 `generalized_jsd`），静默走错分支——这是最隐蔽的 bug 类型。
3. **跨任务迁移要改的不只是数据**。MCQ→开放式摘要的任务性质差异（有/无客观对错、答案空间大小、退化模式是否 self-reinforcing）直接影响方法是否成立，不能只换 parquet 就跑。

### 3.1 立即措施

- **已终止训练**：step450 退化率 23% 且加速，继续跑无意义。
- **当前 run 可用 checkpoint**：仅 **step150**（退化率 8%，ru=3.28/zh=3.91），但也不理想，仅作应急。step300+ 全部废弃。

### 3.2 修复方案（治本程度递增）

| 方案 | 做法 | 治本程度 | 成本 |
|------|------|----------|------|
| **D. 补全 distillation 配置（首选）** | `run_rp_opsd_v2.sh` 补 3 行：`distillation_objective=mopd_topk_reverse_kl` + `distillation_topk_source=teacher` + `distillation_add_tail=False`（`full_logit_distillation=True` 本已默认） | **治本**，走对论文 loss 分支 | 极低，改 3 行配置 |
| **A. SFT warm-start** | student 初始权重从裸 base 换成 SFT-9B epoch2.0（MOS 3.95），`MODEL_PATH` 指向 SFT ckpt | 治标，延后退化、降低初始方差 | 低，改 `MODEL_PATH` |
| **B. 固定外部 teacher** | `teacher_regularization=none`，用 SFT-9B/27B 当不退化 teacher | 治本，断开 EMA 正反馈链 | 中，需部署 teacher |
| **C. 加质量约束** | v5 evaluator 给 rollout 打分过滤退化 rollout；或转 C-GRPO | 治本，需开发 | 高 |

### 3.3 推荐组合：D + A

**首选 D 单独跑一轮**（补 3 行配置），验证 `mopd_topk_reverse_kl` 在开放式摘要上是否原生抗退化。在 `run_rp_opsd_v2.sh` 第 156 行后补：

```bash
    actor_rollout_ref.actor.self_distillation.distillation_objective=mopd_topk_reverse_kl \
    actor_rollout_ref.actor.self_distillation.topk_source=teacher \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=False \
```

冒烟验证：`run_rp_opsd.sh --smoke` 的诊断脚本会 grep `mopd_bias_correction_mean` 和 `teacher_topk_mass_mean`，本地跑起来后在 train.log 确认这俩 metric 非 0 且有限，即证明已走 reverse-KL 分支（而非 `generalized_jsd` 无此 metric）。

若 D 单独仍退化（开放式摘要 + 1502 步可能仍压不住），叠加 **A（SFT-9B warm-start）** 降初始方差。D+A 仍是 EMA teacher + reward-free，不依赖外部服务，成本最低。

若 D+A 仍不够，再上 B（固定 teacher）或 C（C-GRPO）。B 的 SFT-9B 同词表白盒蒸馏方案见朴素版 OPD 文档。

### 3.4 备选：C-GRPO 路线

若固定 teacher 仍不够（固定 teacher 质量本身有上限），转 C-GRPO（`project_cgrpo_swift_landing`）：用 v5 evaluator 当 reward，单阶段 GRPO + MBR 蒸馏，有显式质量约束，从机制上杜绝退化正反馈。warm-start 同样用 SFT-9B epoch2.0。

---

## 四、关键数据索引

| 项 | 路径 |
|----|------|
| 训练脚本 | `scripts/run_rp_opsd_v2.sh` |
| 训练输出 | `outputs/flashnote_train_v2/` |
| rollout 样本 | `outputs/flashnote_train_v2/rollouts/{step}.jsonl` |
| tensorboard | `tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B/events.out.tfevents.1788139796.*` |
| step450 评测 | `eval_results/eval_res_0901/rp_opsd_v2_summary_9b_step450/` |
| SFT 最优 ckpt | `outputs/flashnote_sft_ori/v3-20260830-211922/checkpoint-*` |
| SFT 评测参考 | `eval_results/eval_res_0830/sft_summary_9b_epoch2.0/`（MOS 3.95） |
| 朴素 OPD 方案文档 | `docs/`（去特权 + 外部固定 teacher） |

---

## 五、结论

RP-OPSD v2 训练失败的**直接根因是配置漏传走错 loss 分支**：`run_rp_opsd_v2.sh` 未传 `distillation_objective`/`distillation_topk_source`/`distillation_add_tail`，verl 静默走默认 `generalized_jsd`（forward JSD + student 选 support + tail bucket），而非论文的 `mopd_topk_reverse_kl`（bias-corrected reverse-KL + teacher 选 support + no tail）。forward JSD 的 mode-covering 性质 + student 自选 support + EMA teacher 跟随退化，构成正反馈：student 退化→选退化 support→teacher 在退化 support 上退化→student 匹配退化 teacher→重复/语言混杂，loss 因两者一起坍缩到低熵而下降。放大因素：开放式摘要（无客观对错、退化 self-reinforcing）vs 官方 MCQ、1502 步 vs 官方 55 步、裸 base 无 warm-start。

loss 健康是假象，必须以 rollout 质量和评测 MOS 为准。**首选修复是补 3 行 distillation 配置走对论文 loss 分支**（方案 D），成本极低；若开放式摘要上仍退化再叠加 SFT warm-start（A）或固定 teacher（B）。本案例的最大教训：移植方法必须逐项核对 canonical 配置，`loss_mode=vopd` 只是兼容开关，漏传 objective 不报错但静默走错分支——这类"默认值掩盖"是比 loss 曲线更隐蔽的 bug。
