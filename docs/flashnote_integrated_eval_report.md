# flash_note 全实验整合评测报告

> **整合自三份原始报告**（口径已统一，可直接横向比较）：
> 1. `flashnote_sft_ori_eval_report.md` — sft_ori（原始数据 full SFT，9 epoch）
> 2. `sft_gold_397b_lora_r64_m2_eval_report.md` — sft_gold_397b（397B 教师 gold 数据 SFT，LoRA r64 + 全参对照）
> 3. `loss_analysis.md` — RP-OPSD 四实验 loss 分析 + §13 口径统一对比 + §14 论文修正
>
> **生成日期**：2026-09-08

---

## 0. 一句话结论

**统一评测口径后，SFT > RP-OPSD；gold 数据 SFT > 原始数据 SFT；LoRA 与全参打平。** flash_note 摘要任务的瓶颈是**视觉感知（entity_error）**而非语言建模，因此"注入知识"的 SFT 天花板高于"对齐分布"的 OPD。OPD 的正确用法是 SFT 后接精修，而非从 base 独立起步——但 §14 进一步指出，选对特权信号（VAD 擦除对比）的 9B 自蒸馏也能 +6.07，问题不在自蒸馏天花板而在特权信号选错。

---

## 1. 横向排名（4 维均分，口径已统一）

| 实验 | 训练方式 | 学生起点 | 教师 | 准确性 | 简洁性 | 完整性 | 格式 | **4维均分** | badcase% | 诊断 |
|------|---------|---------|------|--------|--------|--------|------|-----------|----------|------|
| **Qwen3-397B**（教师，gold 来源） | —（教师基座） | — | — | 4.76 | 4.82 | 4.99 | 5.00 | **4.89** | 1.95% | 上限，gold 数据来源 |
| sft_gold_397b ep1.0 | gold 数据 full SFT | base 9B | 397B 离线 gold | 4.569 | 4.773 | 4.977 | 5.000 | **4.830** | 4.8% | ep1.0 到顶，首选 |
| sft_gold_397b_lora_r64 ep2.5 | gold 数据 LoRA r64 | base 9B | 397B 离线 gold | 4.591 | 4.751 | 4.982 | 5.000 | **≈4.83** | 3.8% | 与全参打平，生产首选 |
| sft_ori ep1.5 | 原始数据 full SFT | base 9B | — | 4.546 | 4.792 | 4.754 | 4.969 | **4.765** | 8.3% | 收敛极早，ep≥3.5 过拟合 |
| v3-no-ema step300 | 9B←9B 自蒸馏 RKL | base 9B | 冻结 base 9B | 4.306 | 4.571 | 4.937 | 4.967 | **4.695** | 8.9% | 真收敛 plateau 0.046 |
| v3-no-ema step150 | 同上 | base 9B | 冻结 base 9B | 4.265 | 4.555 | 4.934 | 4.961 | **4.679** | 8.9% | 同上早期点 |
| **base 9B** | —（基座，未训练） | base 9B | — | 4.13 | 4.61 | 4.92 | 4.96 | **4.65** | 13.0% | 对照下限 |
| RP-OPSD v3sft step493 | 9B←9B 自蒸馏 RKL | SFT 9B | EMA(rate=0.05) | — | — | — | — | — | — | **假收敛**（EMA 退化） |
| RP-OPSD v4-fixed step216 | 9B←9B 自蒸馏 RKL | SFT 9B | 冻结 SFT 9B | — | — | — | — | — | — | 真收敛但震荡 |
| RP-OPSD v5-27B step67 | 27B→9B 跨尺寸 RKL | base 9B | 冻结 27B | — | — | — | — | — | — | **延迟发散**（step45 转折） |
| rp_opsd_v2（verl RL） | verl RL | — | — | — | — | — | — | 3.485→2.967 | — | 越训越差，配置失败 |

**base 9B(4.65) < v3-no-ema(4.695) < sft_ori(4.765) < sft_gold(4.830) < 397B 教师(4.89)**。SFT 学生（4.830）距 397B 教师（4.89）天花板差 0.06，主要在准确性（4.569 vs 4.76）。OPD（v3-no-ema）相对 base 仅 +0.045 均分、badcase 13%→8.9%，增益微弱；SFT（gold）相对 base +0.18 均分、badcase 13%→4.8%，增益显著——但学生始终够不到教师上限，因视觉感知硬伤（详见 §4）。

> v3sft/v4/v5 因发散或震荡无可信 MOS；v3sft 表面 loss 最低（0.007）但为 EMA 假收敛，未取评测分。

---

## 2. SFT 两条基线对比（sft_ori vs sft_gold_397b）

### 2.1 gold 数据显著优于原始数据，增益来自准确性

| 指标 | sft_ori ep1.5（最优） | sft_gold ep1.0（最优） | 差值 |
|------|---------------------|----------------------|------|
| 4维均分 | 4.765 | 4.830 | **+0.065** |
| 准确性 | 4.546 | 4.569 | +0.023（+0.263 vs v3-no-ema） |
| badcase% | 8.3% | 4.8% | **-3.5pp** |
| 5项均分（报告原口径） | 4.012 | 4.064 | +0.052 |

gold 数据把 badcase 率从 ~8% 压到 ~4%，且收敛更早（ep1.0 到顶 vs sft_ori ep1.5）。**gold 的价值 = 用 397B 教师正确 summary 直接注入学生不具备的知识（正确数值、正确图标语义）**，这正是 OPD token 级蒸馏做不到的。

### 2.2 LoRA r64 与全参 SFT 完全打平，首选 LoRA

| 实验 | 训练方式 | 最优 epoch | 5项均分 | badcase% |
|------|---------|-----------|---------|----------|
| sft_gold_397b_lora_r64 | LoRA r=64, α=128 | ep2.5（step2250） | 4.064 | 3.80% |
| sft_gold_397b（全参） | full SFT | ep1.0 | 4.064 | 4.80% |

两者最优均分完全相同（4.064），LoRA 训练成本仅 r=64（80M 可训参数 vs 全参 9.15B，差 114 倍），且 badcase 率更低。**后续迭代优先选 LoRA r64 方案**，最终 ckpt = `outputs/flashnote_sft_gold_397b_lora_r64_m2/merged/step_2250`。

### 2.3 SFT 收敛特征共性

- **收敛极早**：sft_ori ep1.0 已达 3.998（5项），sft_gold ep1.0 到顶；epoch≥3.5 后轻度过拟合（badcase 从 ~4-8% 升至 ~11%）。
- **维度排序**：格式 ≈ 完整性 > 简洁性 > 准确性，语种遵循度全程接近 1。**准确性是唯一短板和主要 badcase 来源**。
- **zh 唯一持续走强**（gold 报告）：4.060→4.081→4.097（ep1→2.5→5）；en/fr/ru 在 ep3.0+ 下滑。
- **稳定性好**：99%+ valid，无 timeout。

---

## 3. RP-OPSD 四实验诊断速查

四实验唯一变量是教师来源与更新方式，其余超参一致（MOPD top-k reverse-KL，α=1.0，topk=100，LR=2e-6，IS clip=2.0）。

| 指标 | v3sft (EMA) | v3-no-ema (冻base) | v4-fixed (冻SFT) | v5-27B (冻27B) |
|------|-------------|---------------------|-------------------|----------------|
| vopd_loss 最终 | 0.007（假） | 0.046（真 plateau） | 0.126（震荡） | 0.263（反弹） |
| bias_correction | 稳定 0.003 | 稳定 0.016 | 稳定 0.010 | **爆炸 0.012→0.204** |
| student_topk_mass | 0.841（双降退化） | 0.966（稳） | 0.985（稳） | **0.603（暴跌）** |
| ppl_ratio | 🔴 90万 | 1.15（含离群点噪声） | 🔴 104万 | 195（爆炸） |
| rollout_ppl | 🔴 148 | 5.7（含离群尖峰） | 4.9 | 🔴 **915** |
| response_length | 503（膨胀） | 307 | 243 | 294 |
| **诊断** | **假收敛** | ✅ 真收敛 | 真收敛但震荡 | **延迟发散** |

### 3.1 三种失败/成功模式

1. **v3sft 假收敛**：EMA 教师跟随学生退化 → 师生同步变平 → 反向 KL 自然趋零。铁证：topk_mass 双降至 0.84 且 gap 恒定 0.003，ppl_ratio 剧烈波动。loss→0.007 不代表学到东西。
2. **v3-no-ema 真收敛**：冻结教师 + base 学生，ppl_ratio 基本稳定（含离群点但 IS clip 截断不影响 loss），topk_mass 稳定，bias_correction 稳定 0.016。1 epoch（751 步）已收敛，第 2 epoch 边际收益递减。**但被 idleprotect 误判 SIGTERM 杀于 step 1055（epoch1 70%）**，ckpt-1050 可用。
3. **v5-27B 延迟发散**：前 45 步健康（loss 0.50→0.22），step 46 起急剧恶化。失败模式 = **支撑域外 mode collapse**：学生 40% 概率质量移到教师 top-100 外，bias_correction 爆炸 17 倍。reverse_kl_term 持续下降是假象（概率移出求和范围导致总和变小）。rollout 文本三阶段退化：健康→短语循环(9.8%)→乱码(4.3%)。

### 3.2 v5 发散根因（五层因果链，loss_analysis §Q12）

27B→9B 跨尺寸蒸馏 + base 学生起点 → 初始 raw_jsd 0.494（接近 ln2 上界，9B←9B 仅 0.122 精度噪声）→ top-k 表面重叠但本质不同 → 反向 KL mode-seeking 推学生离开自然分布 → top-k 截断让泄漏概率不被 reverse_kl 惩罚（假收敛温床）→ rollout 退化 → ppl_ratio 爆炸 → IS 失稳 → 恶性正反馈。

**核心教训**：跨尺寸蒸馏不能用纯反向 KL + top-k 截断。同尺寸蒸馏（9B←9B）无此问题，因师生同分布，反向 KL 只做尖锐化不做模式转移。

---

## 4. SFT vs RP-OPSD：为何 OPD 比不上 SFT（loss_analysis §13）

统一口径后 v3-no-ema（4.695）< sft_ori（4.765）< sft_gold（4.830）。三个根因：

**根因 A（被 §14 部分修正）**：9B←9B 自蒸馏，教师能力上限 = base 9B。OPD 只能 sharpen 已有分布。SFT 用 gold 直接注入知识。**但 §14 引入 VAD 论文后修正：9B 自蒸馏天花板不低（VAD +6.07），问题在特权信号选错而非自蒸馏本身。**

**根因 B（视觉感知硬伤，最关键）**：v3-no-ema badcase 73% 是 entity_error（24.7万→2.47万差一个数量级、4.5G→5G、图标状态认错、挖掘机"装"实为"卸"）。这是视觉编码器感知上限，token 级蒸馏无法教学生"看清图里的数字"。SFT 用 gold 强制告诉正确答案。**铁证：step150→step300 badcase 数量完全相同（76→76），OPD 训练 150 步一个 badcase 都没修复。**

**根因 C（on-policy 强化错误）**：学生看错图 → rollout 生成错误描述 → 教师（EMA=学生慢变版）若也认错 → 蒸馏强化错误。

### 4.1 与 sft_ori §8 跨脚本感知天花板结论的交叉印证

sft_ori 报告 §8 用 9 张顽固跨脚本 badcase（阿姆哈拉/阿拉伯/乌尔都/孟加拉/西里尔）跨三档模型实测：
- ai-gold 教师能读对（9 张中 7 满分）
- sft_ori 9B 读不对
- sft_gold_397b 蒸馏后 9B 仍读不对（0/9 真正读对）

**蒸馏救不了感知**：蒸馏后只产生两种结局——退化为"安全但丢信息"（干脆不读脚本）或编出另一个错读。根因：SFT/蒸馏只教输出分布，不教感知；感知发生在视觉塔，输出对齐梯度不足以重塑视觉表征。30B-A3B 同族放大也无效（错法与 9B 完全一致）。

sft_ori §10 分辨率 A/B 进一步精确化盲点本质：放大 2x/4x 对"会-但-字小"的拉丁/西里尔脚本有效（mHx 4x 从漏名变全对），但对阿姆哈拉/孟加拉/乌尔都放大无效（模型对这些字形无预训练表征）。**盲点本质 = 视觉塔对罕见脚本字形缺乏预训练表征，不在分辨率。**

---

## 5. 优化方向排序（loss_analysis §14 最终修正版）

§13 原结论"9B 自蒸馏无能力注入"被 VAD 论文证伪后，方向重排：

| 优先级 | 方向 | 攻击点 | 依据 | 成本/风险 |
|--------|------|--------|------|----------|
| **1（新首选）** | **VAD 擦除对比特权** | entity_error 根因 | 9B frozen 自身教师同范式 +6.07，零外部依赖；已 clone `VAD_Multimodal_OPD`（verl 同构） | 低 |
| **2** | **OPD-V 模态平衡 trust region** | 视觉信号被文本梯度淹没 | 9B +7.88，同裁剪特权但加 trust region；已 clone `OPD-V` | 低，可叠加 VAD |
| 3 | SFT→OPD 两段式（保底） | exposure bias | sft_gold ckpt warm-start 后 JSD α=0.5 精修；师生分布接近不发散 | 低，复用已有 ckpt |
| 4 | 27B+JSD+SA-OPD 虚假信号过滤 | 教师盲图 drift | §9 推荐 v6 配置（见下）；27B 真实质量存疑需先验 | 中高 |
| 5 | C-GRPO reward-based | reward 约束 | 用 v5 evaluator（GPT-5 2-agent）当 reward 绕过教师 logit | 高，SSM 未验证 |
| 6（治本） | 视觉理解根因 | 罕见脚本感知 | 更高分辨率/OCR 辅助/更强视觉编码器；前置 OCR 立竿见影 | 极高 |

### 5.1 若走 27B 教师路线：v6 推荐配置（loss_analysis §9）

v5 发散后，v6 = "三改 + 一前置"：

| 改动 | v5（发散） | v6（推荐） | 针对根因 |
|------|-----------|-----------|---------|
| OBJECTIVE | mopd_topk_reverse_kl | generalized_jsd | 换掉 mode-seeking RKL；JSD 梯度有界 ≤log2 |
| ALPHA | 1.0 | 0.5 | 广义 JSD = 0.5×FKL + 0.5×RKL，等价 swift GKD beta=0.5 |
| ADD_TAIL | False | False（走 renorm） | renorm 使泄漏概率被 inflate，形状差异仍被 JSD 捕获 |
| LR | 2e-6 | 5e-7 | 减缓推力 |
| LR_WARMUP | 75 | 200 | IS 权重充分稳定 |
| 学生起点 | base 9B | **SFT 9B** | 缩小初始 raw_jsd（0.494→预计 0.15-0.20） |

**实证支撑**：dialog_title 任务 OPD（27B→4B，跨度更大）用 swift GKD beta=0.5（=JSD）稳定收敛。唯一关键变量是 loss 类型（JSD vs RKL）。代码约束：只改 ALPHA 不行（mopd 分支 assert alpha==1.0），必须换 objective。

### 5.2 OPD 的价值定位（核心判断）

**OPD 在 flash_note 上不是"替代 SFT"而是"精修 SFT"**。SFT 注入知识（解决 entity_error 靠 gold），OPD 对齐分布（解决 exposure bias）。正确用法是 SFT 先行、OPD 后接，而非 OPD 独立从 base 起步。v3-no-ema 从裸 base 起步是 OPD 比不上 SFT 的直接原因。

但 §14 进一步指出：换 VAD 擦除对比特权 + OPD-V trust region，9B 自蒸馏即可大幅提升准确性，**不必依赖 27B 教师或 SFT warm-start**——这是比两段式更治本的路线。

---

## 6. 训练健康度监控判据（loss_analysis §9.5）

训练时盯以下信号，任一触发即应暂停降 LR 或回退 ckpt：

| 信号 | 阈值 | 含义 |
|------|------|------|
| bias_correction > 0.05 | （若仍存在） | 学生开始脱离教师支撑域 |
| student_topk_mass < 0.90 | | 概率泄漏超 10% |
| rollout_ppl > 50 | | rollout 生成低质量 token |
| raw_jsd 连续 10 步不降 | | 训练停滞 |
| rollout word_rep > 3% | | 短语循环苗头（跑 detect 脚本） |

**bias_correction 是最早暴露发散的信号**：v5 在 step 45 时 bias 已升（0.012→0.016），而 loss 反弹到 step 46 才出现——bias 比 loss 早 1 步。

**诊断口诀（reverse_kl_term + bias_correction 联合判读）**：
- reverse_kl 下降 + bias 稳定 → 真收敛
- reverse_kl 下降 + bias 上升 → 假收敛（支撑域外 mode collapse）
- reverse_kl 上升 → 学生在 top-k 内远离教师

**永远不要单独看 reverse_kl_term**（top-k 截断会让它"骗人"：学生概率移出求和范围导致总和虚降）。

---

## 7. 关键行动项

1. **SFT 上线**：用 sft_gold_397b_lora_r64 ep2.5（step2250），4维均分 4.83，badcase 3.8%。已结案。
2. **OPD 主推**：VAD 擦除对比特权（9B 自蒸馏，零外部依赖，预期 +6.07 量级）。已 clone `VAD_Multimodal_OPD`（verl 同构），可直接复用 RP-OPSD 框架。
3. **OPD 叠加**：OPD-V 模态平衡 trust region 保护视觉信号（预期 +7.88 量级）。
4. **保底路线**：SFT→OPD 两段式（sft_gold ckpt warm-start + JSD α=0.5）。
5. **27B 路线**：先验 27B 在 flash_note 上真实摘要质量（memory 记录 27B 3.893 < 9B 4.034 方向反了，需排查 27B 服务解码退化），确认 27B>9B 后再开 v6（JSD 配置见 §5.1）。
6. **前置 OCR**（生产侧立竿见影）：PaddleOCR 支持 amh/ben/urd/ara/rus，绕过 Qwen 视觉塔盲区，把脚本文字抽出注入 prompt。对"会-但-字小"的拉丁/西里尔脚本，LANCZOS 2x 预处理零成本有效。
7. **idleprotect 坑**：重启任何训练前先 `tmux ls | grep idleprotect` 杀保护会话（v3-no-ema 即被其误判 SIGTERM 杀于 step 1055）。

---

## 8. 文件索引

| 内容 | 路径 |
|------|------|
| sft_ori 原始评测（9 epoch + badcase 深度分析 + 跨脚本 + 分辨率 A/B） | `docs/flashnote_sft_ori_eval_report.md` |
| sft_gold_397b_lora_r64 评测（LoRA vs 全参） | `docs/sft_gold_397b_lora_r64_m2_eval_report.md` |
| RP-OPSD loss 分析（四实验 + FAQ + 107 tag 释义 + 修复方案 + 预检清单 + §13/§14） | `docs/loss_analysis.md` |
| sft_ori 最佳 ckpt | sft_ori ep2.5（5项 4.013 / 4维 4.765） |
| sft_gold 最佳 ckpt | `outputs/flashnote_sft_gold_397b_lora_r64_m2/merged/step_2250` |
| VAD 仓库 | `/data4/wumeimei/flash_note/VAD_Multimodal_OPD` |
| OPD-V 仓库 | `/data4/wumeimei/flash_note/OPD-V` |
| SA-OPD 仓库 | `/data4/wumeimei/flash_note/SA-OPD` |
| rollout 退化检测脚本 | `RP-OPSD/scripts/detect_rollout_degradation.py` |
| base 9B 评测报告（eval_res_0904，唯一基座评测） | `eval_results/eval_res_0904/qwen35_9b_base_summary/{en,fr,ru,zh}/{lang}_eval_report.md` |
| 397B 教师评测报告（eval_res_0906，gold 数据来源） | `eval_results/eval_res_0906/qwen397b_summary/{en,fr,ru,zh}/{lang}_eval_report.md` |
| base 9B 部署脚本（注释标"基座，未 SFT/未 OPSD"） | `infer/start_qwen35_9b_base_m2.sh` |
| 分辨率 A/B 脚本 | `infer/resolution_ab_test.py` |
| image OPD 调研报告 | `explore_tasks/image_OPD调研报告.md` |
