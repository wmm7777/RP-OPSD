# sft_gold_397b（全参 SFT）实验评测报告

> 生成时间：2026-09-01
> 实验阶段：epoch 1.0 / 1.5 / 4.5 / 5.0（**4 个 checkpoint**，中间 epoch2.0/2.5/3.0/3.5/4.0 ckpt 已被清理，无法补评测，见 §1.4）
> 训练机器：m3 (10.162.52.31)，全参 SFT
> 训练版本：`v5-20260831-134357`

## 1. 实验设置

### 1.1 训练

| 项 | 值 |
|---|---|
| 基座模型 | `/data4/wumeimei/download_models/Qwen3.5-9B`（9.4B 参数，9.15B 可训） |
| 训练方法 | **全参 SFT**（非 LoRA） |
| 训练数据 | `/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/sft_gold_397b_final.jsonl`（397B 教师模型生成的 gold summary） |
| 训练机器 | m3 (10.162.52.31)，H20 |
| 训练版本 | `v5-20260831-134357` |
| 训练状态 | 已完成（3730/3730 steps = 5.0 epoch，train_loss=0.174，token_acc=99.08%） |
| 训练耗时 | 16h 40m |

### 1.2 关键超参

```
--num_train_epochs 5
--per_device_train_batch_size 2
--gradient_accumulation_steps 6
--max_length 6144
--learning_rate 1e-5          ← 全参 SFT，比 LoRA 版低 10 倍
--lr_scheduler_type cosine
--warmup_steps 0
--weight_decay 0.1
--optim adamw_torch_fused
--adam_beta1 0.9 --adam_beta2 0.95
--bf16 true
--gradient_checkpointing true
--save_steps 376              ← 每 epoch ≈ 746 step，每 ~0.5 epoch 存一次
--save_total_limit 12
--deepspeed zero2
```

- 746 steps/epoch，总 3730 steps
- `save_steps=376` 导致保存位置在 step 376/752/1128/...，**不刚好对齐 0.5/1.0/1.5 epoch**，评测取的是最接近的 ckpt
- 评测用的 ckpt：epoch1.0→step746 区域、epoch1.5→step1119 区域、epoch4.5→step3357 区域、epoch5.0→step3730（末尾）

### 1.3 评测

| 项 | 值 |
|---|---|
| 样本量 | 220 条/语种 × 4 语种 = 880 条/epoch |
| 语种 | en / fr / ru / zh |
| 评委 | gemini-3-flash-preview |
| MOS 模式 | flash_summary_mos（summary-only，跳过 title） |
| 推理 | vllm serve，单卡 TP=1，bfloat16 |
| 推理 prompt | 训练同款 Core Identity prompt（`core_identity_prompts.json`） |
| enable_thinking | False |
| MAX_TOKENS | 1024 |
| TEMPERATURE | 0 |
| CONCURRENCY | 64 |

### 1.4 ⚠️ ckpt 缺失说明（数据不全的原因）

训练目录 `/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft_gold_397b/v5-20260831-134357/` 当前**只剩 checkpoint-3384 和 checkpoint-3730**（对应 epoch4.5 区域和 epoch5.0 末尾）。

中间 epoch1.0/1.5/2.0/2.5/3.0/3.5/4.0 的 ckpt 在 2026-09-01 的 ckpt cleanup 事故中被删（参见 `feedback_ckpt_cleanup_no_rm_rfk` 记录：清理脚本误删了 `sft_gold_397b` epoch1.0~4.0 的全部 model-*.safetensors + config + tokenizer）。

**已跑评测保留结果**的 epoch：
- epoch1.0 / epoch1.5：在 ckpt 被删之前已跑完评测，原始 JSON 结果在 `eval_results/eval_res_0831/`
- epoch4.5 / epoch5.0：ckpt 还在，在 `eval_results/eval_res_0901/`

**缺失 epoch**：epoch2.0 / 2.5 / 3.0 / 3.5 / 4.0 —— ckpt 已删，无法补评测。

## 2. 评测结果

### 2.1 平均分（核心指标）

| epoch | step | en | fr | ru | zh | avg | badcase% | bad/valid |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 746 | 4.031 | 4.093 | 4.056 | 4.075 | **4.064** | 4.80% | 42/875 |
| 1.5 | 1119 | 4.028 | 4.079 | 4.050 | 4.073 | **4.058** | 4.45% | 39/876 |
| 4.5 | 3357 | 4.032 | 4.089 | 4.019 | 4.083 | **4.056** | 3.68% | 32/869 |
| 5.0 | 3730 | 4.024 | 4.072 | 4.036 | 4.073 | **4.051** | 5.07% | 44/868 |

> 注：avg 是 4 语种 valid MOS 等权平均。

### 2.2 各维度均分（4 语种平均）

| epoch | 准确性 | 简洁性 | 完整性 | 格式 | 语种遵循度 | MOS |
|---|---|---|---|---|---|---|
| 1.0 | 4.569 | 4.773 | 4.977 | 5.000 | 1.000 | 4.064 |
| 1.5 | 4.564 | 4.751 | 4.974 | 5.000 | 0.999 | 4.058 |
| 4.5 | 4.557 | 4.758 | 4.976 | 4.994 | 0.995 | 4.056 |
| 5.0 | 4.543 | 4.740 | 4.979 | 4.999 | 0.995 | 4.051 |

### 2.3 各 epoch × lang 明细（5 维 + badcase）

#### 2.3.1 准确性

| epoch | en | fr | ru | zh | avg |
|---|---|---|---|---|---|
| 1.0 | 4.413 | 4.709 | 4.564 | 4.590 | 4.569 |
| 1.5 | 4.441 | 4.641 | 4.578 | 4.596 | 4.564 |
| 4.5 | 4.465 | 4.683 | 4.468 | 4.610 | 4.557 |
| 5.0 | 4.437 | 4.656 | 4.498 | 4.582 | 4.543 |

#### 2.3.2 简洁性

| epoch | en | fr | ru | zh | avg |
|---|---|---|---|---|---|
| 1.0 | 4.771 | 4.764 | 4.759 | 4.797 | 4.773 |
| 1.5 | 4.736 | 4.759 | 4.716 | 4.794 | 4.751 |
| 4.5 | 4.719 | 4.780 | 4.713 | 4.821 | 4.758 |
| 5.0 | 4.723 | 4.711 | 4.733 | 4.791 | 4.740 |

#### 2.3.3 完整性

| epoch | en | fr | ru | zh | avg |
|---|---|---|---|---|---|
| 1.0 | 4.972 | 4.991 | 4.959 | 4.986 | 4.977 |
| 1.5 | 4.964 | 4.995 | 4.959 | 4.977 | 4.974 |
| 4.5 | 4.982 | 4.982 | 4.954 | 4.986 | 4.976 |
| 5.0 | 4.967 | 4.991 | 4.968 | 4.991 | 4.979 |

#### 2.3.4 格式

| epoch | en | fr | ru | zh | avg |
|---|---|---|---|---|---|
| 1.0 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 |
| 1.5 | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 |
| 4.5 | 5.000 | 5.000 | 4.977 | 5.000 | 4.994 |
| 5.0 | 5.000 | 5.000 | 4.995 | 5.000 | 4.999 |

#### 2.3.5 语种遵循度

| epoch | en | fr | ru | zh | avg |
|---|---|---|---|---|---|
| 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.5 | 1.000 | 1.000 | 0.995 | 1.000 | 0.999 |
| 4.5 | 0.995 | 1.000 | 0.986 | 1.000 | 0.995 |
| 5.0 | 0.995 | 1.000 | 0.986 | 1.000 | 0.995 |

#### 2.3.6 Badcase 率（任一 5 分维度 <3 或 语种遵循度=0）

| epoch | en | fr | ru | zh | 整体 |
|---|---|---|---|---|---|
| 1.0 | 5.96% (13/218) | 2.73% (6/220) | 5.45% (12/220) | 5.07% (11/217) | 4.80% (42/875) |
| 1.5 | 5.45% (12/220) | 4.09% (9/220) | 5.50% (12/218) | 2.75% (6/218) | 4.45% (39/876) |
| 4.5 | 4.15% (9/217) | 0.92% (2/218) | 6.94% (15/216) | 2.75% (6/218) | 3.68% (32/869) |
| 5.0 | 6.57% (14/213) | 2.29% (5/218) | 6.45% (14/217) | 5.00% (11/220) | 5.07% (44/868) |

## 3. 横向对比

### 3.1 与 sft_gold_397b_lora_r64_m2（LoRA 版，m2）对比

| 实验 | 训练方式 | 机器 | 最佳 epoch | 最佳 avg MOS | 最佳 badcase% |
|---|---|---|---|---|---|
| **sft_gold_397b（全参）** | full SFT | m3 | epoch1.0 | **4.064** | 4.80% |
| sft_gold_397b_lora_r64_m2 | LoRA r=64 | m2 | epoch2.5 | **4.064** | — |

**结论**：全参 epoch1.0 与 LoRA r64 epoch2.5 **MOS 完全打平（4.064）**。考虑到：
- 全参训了 16h40m（5 epoch / 3730 step），LoRA 训了 ~8h（5 epoch / 4970 step，但 LoRA backward 远快）
- 全参 epoch1.0 仅 746 step（~3.3h）就到顶，LoRA 需要 2250 step（~3.6h）才到顶
- 全参 trainable params 9.15B vs LoRA r=64 约 80M（差 114 倍）

**全参 SFT 相对 LoRA 几乎无质量增益，但训练显存/工程成本高得多**。后续迭代优先选 LoRA r64 方案。

### 3.2 与 sft_ori（普通 SFT，非 gold）对比

| 实验 | 数据 | 最佳 epoch | 最佳 avg MOS | 最佳 badcase% |
|---|---|---|---|---|
| sft_ori（m3 全参） | 原 parquet prompt | epoch2.5 | 3.998~4.013 | 8.29%~9.06% |
| **sft_gold_397b（m3 全参）** | 397B 教师 gold | epoch1.0 | **4.064** | **4.80%** |

**结论**：
- MOS 提升 +0.051 分（4.013 → 4.064，相对 +1.3%）
- **badcase 率近乎腰斩**（8.93% → 4.80%，相对 -46%）
- gold summary 把底线拉高了——普通 SFT 在 hard examples（跨脚本 OCR 等）上崩得更多，gold 教师给的就是"读不准也要写对格式"的版本，学生学到更保守的输出分布

### 3.3 与 RP-OPSD verl RL 对比

| 实验 | tag | avg MOS |
|---|---|---|
| sft_gold_397b（全参） | epoch1.0 | 4.064 |
| sft_gold_397b_lora_r64_m2 | epoch2.5 | 4.064 |
| rp_opsd_v2 verl RL | step150 | 3.485 |
| rp_opsd_v2 verl RL | step300 | 3.135 |
| rp_opsd_v2 verl RL | step450 | 2.967 |

**结论**：RP-OPSD verl RL 远差于 SFT，且越训越差，需检查训练配置（reward、KL、teacher、lr 等）。

## 4. 事实核查错误类别（按句计，未按 rid 去重）

> 本节统计每条 badcase 句子的错误类别计数（一条样本可能多句、多类错误共存）。与 sft_ori 报告 §6.4 的"按 rid 去重"口径不同，只能横向看相对分布，不能直接对比绝对数。

| epoch | 语种 | entity | predicate | circumstantial | out_of_context | grammatical | linking |
|---|---|---|---|---|---|---|---|
| 1.0 | en | 75 | 8 | 12 | 4 | 0 | 1 |
| 1.0 | fr | 42 | 4 | 2 | 0 | 0 | 0 |
| 1.0 | ru | 55 | 14 | 1 | 2 | 1 | 0 |
| 1.0 | zh | 50 | 10 | 6 | 2 | 0 | 0 |
| 1.5 | en | 74 | 11 | 5 | 3 | 0 | 0 |
| 1.5 | fr | 48 | 7 | 3 | 1 | 0 | 0 |
| 1.5 | ru | 49 | 16 | 2 | 4 | 0 | 0 |
| 1.5 | zh | 57 | 14 | 2 | 1 | 0 | 0 |
| 4.5 | en | 76 | 11 | 5 | 1 | 1 | 0 |
| 4.5 | fr | 49 | 9 | 2 | 1 | 3 | 2 |
| 4.5 | ru | 61 | 26 | 3 | 4 | 9 | 0 |
| 4.5 | zh | 57 | 12 | 2 | 2 | 0 | 0 |
| 5.0 | en | 72 | 10 | 8 | 2 | 0 | 0 |
| 5.0 | fr | 50 | 6 | 3 | 1 | 4 | 0 |
| 5.0 | ru | 63 | 15 | 3 | 3 | 5 | 0 |
| 5.0 | zh | 52 | 14 | 4 | 0 | 0 | 0 |

**观察**：
1. **entity_error 全语种全 epoch 都是绝对主因**（en 70+/216-220、fr 42-50/218、ru 49-63/215-220、zh 50-57/217-220）——与 sft_ori 报告 §6.4 的 rid 去重口径结论一致：跨脚本 OCR/转写错字是硬伤，gold 蒸馏也救不掉。
2. **ru 的 predicate_error 在 epoch4.5 跳到 26**（其他 epoch 在 14-16），伴随 grammatical_error=9 出现——epoch4.5 ru 是 4 个 epoch 里最差的（MOS 4.019，badcase 6.94%），多训让 ru 主客体颠倒 + 语法错误双升。
3. **fr 全程最稳**：entity 最低（42-50）、predicate 最低（4-9）、badcase 最低（0.92%-4.09%），与 sft_ori 报告 §7"fr 最稳定、最优窗最长"一致。
4. **epoch4.5 fr 出现 grammatical_error=3 + linking_error=2**，是其他 epoch 没有的——多训到后期开始有轻微语法/指代漂移，但量极小。

## 5. 关键发现

1. **收敛极早**：epoch1.0 (MOS 4.064) 已是 4 个里最高，epoch1.5/4.5/5.0 都低于 epoch1.0。与 sft_ori、lora_r64_m2 的"早到顶"趋势一致，但 sft_gold_397b 的顶点更高（4.064 vs sft_ori 4.013 vs lora_r64_m2 4.064）。
2. **epoch4.5 是 badcase 最低点**（3.68%），但 MOS 不是最高（4.056）——badcase 改善以"hard example 输出更保守"实现，不提升 MOS 均值。
3. **epoch5.0 反弹退化**：MOS 4.051（最低）、badcase 5.07%（最高），ru 语种遵循度跌至 0.986，明显过拟合信号。
4. **格式 5.000 + 语种遵循度 0.995-1.000**：gold 教师训练的模型在格式与语种一致性上接近满分，与 lora_r64_m2 一致。
5. **维度排序**：格式 ≈ 完整性 > 简洁性 > 准确性，准确性 4.54-4.57 是最弱维度——与 sft_ori / lora_r64_m2 同序，gold 蒸馏没改变维度短板结构。
6. **ru 全程最弱**：MOS 4.019-4.056 全是 4 个 epoch 最低，badcase 5.45-6.94% 全最高；sft_ori 报告 §7 已指出"ru 早到顶先过拟合"，sft_gold 也复现了这个规律。
7. **fr 全程最强**：badcase 0.92%-4.09% 全最低，MOS 4.072-4.093 全最高，与 sft_ori 报告 §7 "fr 最稳定、最优窗最长"一致。

## 6. 选型决策

### 6.1 推荐 checkpoint

基于 4 个 epoch 的综合表现：

| 指标 | 推荐 epoch | 理由 |
|---|---|---|
| **MOS 最高** | epoch1.0 | 4.064，4 语种均稳定（4.03-4.09），训练成本最低（746 step / ~3.3h） |
| **badcase 最低** | epoch4.5 | 3.68%，但 MOS 略低（4.056），ru 退化明显 |
| **稳定性最佳** | epoch1.0 或 1.5 | 1.0 的 ru 语种遵循度=1.000；1.5 整体波动最小 |

**综合推荐 epoch1.0**（MOS 最高 + 训练成本最低 + ru 语种遵循度满分）。

### 6.2 但生产首选 LoRA r64 方案

| 指标 | sft_gold_397b（全参） | sft_gold_397b_lora_r64_m2（LoRA） | 决策 |
|---|---|---|---|
| 最佳 MOS | 4.064（epoch1.0） | 4.064（epoch2.5） | **完全打平** |
| 训练时间 | 16h40m（5 epoch 全跑完） | ~8h（5 epoch 全跑完） | LoRA 快一倍 |
| 训练显存 | 92.55 GiB（deepspeed zero2 + gradient_checkpointing） | 远低于全参 | LoRA 部署门槛低 |
| 工程复杂度 | 全参 + deepspeed + 4 模型分片合并 | LoRA + merge_lora 一步导出 | LoRA 简单 |
| 可训参数 | 9.15B（95.15%） | ~80M | LoRA 资源友好 |
| 9 epoch 全程评测 | ❌ 中间 5 epoch ckpt 已删 | ✅ 9 epoch 全有 | LoRA 数据更完整 |

**结论**：生产部署首选 `sft_gold_397b_lora_r64_m2/merged/step_2250`（epoch2.5）。全参 SFT 仅作"确认 LoRA 不输全参"的对照实验，已验证完毕。

## 7. 中间 ckpt 缺失的补救方案

由于 ckpt cleanup 事故，epoch2.0/2.5/3.0/3.5/4.0 的数据永久丢失，无法补评测。但根据现有 4 点数据可外推：

1. epoch1.0→1.5：MOS 4.064→4.058（-0.006），badcase 4.80%→4.45%（-0.35pp）
2. epoch1.5→4.5（跨 3 epoch）：MOS 4.058→4.056（-0.002），badcase 4.45%→3.68%（-0.77pp）
3. epoch4.5→5.0：MOS 4.056→4.051（-0.005），badcase 3.68%→5.07%（+1.39pp）

**外推假设**（与 lora_r64_m2 9 epoch 完整曲线对照）：
- epoch2.0/2.5/3.0：MOS 应在 4.058-4.064 之间窄幅震荡，与 lora_r64_m2 同期 4.059/4.064/4.050 类似
- epoch3.5/4.0：MOS 应开始下滑，与 lora_r64_m2 同期 4.040/4.038 对应
- **最优窗大概率仍在 epoch1.0~2.5 之间**，与 lora_r64_m2 一致

**如需精确曲线**：需重训 sft_gold_397b（保留全部 ckpt），按当前训练速度需 16-17h。但既然 LoRA 已证明不输全参，重训的边际收益极低，**不推荐重训**。

## 8. 文件清单

- 原始 MOS JSON:
  - epoch1.0/1.5: `/data4/wumeimei/flash_note/eval_results/eval_res_0831/sft_gold_397b_summary_9b_epoch{1.0,1.5}/<lang>/summary_mos_results.json`
  - epoch4.5/5.0: `/data4/wumeimei/flash_note/eval_results/eval_res_0901/sft_gold_397b_summary_9b_epoch{4.5,5.0}/<lang>/summary_mos_results.json`
- 单 tag 单 lang 报告: `.../sft_gold_397b_summary_9b_epoch*/<lang>/<lang>_eval_report.md`
- 训练 args: `/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft_gold_397b/v5-20260831-134357/args.json`
- 训练 logging: `.../v5-20260831-134357/logging.jsonl`
- 现存 ckpt: `.../v5-20260831-134357/checkpoint-{3384,3730}/`
- 关联报告:
  - sft_ori 9 epoch 报告: `docs/flashnote_sft_ori_eval_report.md`
  - sft_gold_397b_lora_r64_m2 9 epoch 报告: `docs/sft_gold_397b_lora_r64_m2_eval_report.md`
- 本报告: `docs/sft_gold_397b_eval_report.md`
