# 跨模型分辨率特权 On-Policy Distillation

该补充实验用于验证分辨率特权的 On-Policy Distillation 能否在不同模型规模之间
迁移能力。实验使用固定的 Qwen3.5-9B Teacher 监督 Qwen3.5-4B Student，并沿用
RP-OPSD 的 Dataset4.0 和高低分辨率视图；与主实验不同的是，这里用更大的固定
Teacher 替代 EMA Self-Teacher。

## 训练流程

### Student Rollout

Qwen3.5-4B Student 接收宽高各缩小至原始尺寸 1/2 的图像，即原图 1/4 的像素量。
对于每个问题，Student 在线采样 8 条回答，采样参数为 `temperature=1.0`、
`top_p=1.0`、`top_k=-1`，最大回答长度为 1,024 Tokens。

### Teacher Scoring

固定的 Qwen3.5-9B Teacher 接收同一问题及对应的原始分辨率图像。Teacher 不单独
生成答案，而是在 Student 已生成的每一条轨迹上，计算每个回答位置的条件 Token
分布。

### Token 级目标

每个位置选取 Teacher 概率最高的 100 个 Token，并在该 Teacher Top-100 支持集
上计算带截断偏差校正的反向 KL：

$$
\mathcal{L}_{\mathrm{distill}}
=
\sum_{v \in \operatorname{TopK}(p_t)}
\left[
p_s(v)\log\frac{p_s(v)}{p_t(v)} - p_s(v) + p_t(v)
\right].
$$

本实验使用 `alpha=1.0`，不添加 Tail Bucket，也不在 Top-100 支持集上重新归一化。
优化目标只包含该 Token 级蒸馏损失，不额外叠加 GRPO Policy Loss 或 SFT Loss。

### 参数更新

训练时只更新 Qwen3.5-4B Student，Qwen3.5-9B Teacher 全程固定。Student 的视觉
编码器没有冻结，因此其 Vision Tower 和语言模型均参与优化。

### 训练设置

| 设置 | 数值 |
|---|---|
| 数据集 | Dataset4.0，共 5,295 条 |
| Student | Qwen3.5-4B |
| Teacher | Qwen3.5-9B，全程固定 |
| Student 视图 | 宽高各缩小至 1/2 |
| Teacher 视图 | 原始分辨率 |
| 每个问题的 Rollout 数 | 8 |
| Batch Size | 96 |
| 学习率 | 2e-6 |
| Warmup | 10 Steps |
| 训练长度 | 1 Epoch，共 55 Steps |
| 蒸馏支持集 | Teacher Top-100 Tokens |

整体流程为：

```text
半分辨率图像 -> 4B Student 采样回答
                         |
                         | 同一条回答轨迹
                         v
原始分辨率图像 -> 固定 9B Teacher 计算逐 Token 分布
                         |
                         v
              Top-100 偏差校正反向 KL
                         |
                         v
                   更新 4B Student
```

该方法与答案级蒸馏的关键区别在于：9B Teacher 不提供单独生成的目标答案；4B
Student 在自己的 On-Policy 输出轨迹上，学习 Teacher 基于更清晰图像给出的逐
Token 预测分布。

## 实验结果

所有模型均使用原始分辨率图像进行评测。

| Benchmark | Qwen3.5-4B Base | 9B Teacher → 4B Student | 提升 |
|---|---:|---:|---:|
| V\*Bench | 84.29 | **90.58** | **+6.29** |
| HR-Bench 4K | **84.38** | 84.25 | -0.13 |
| HR-Bench 8K | 80.13 | **83.00** | **+2.87** |
| VisualProbe | 43.22 | **49.90** | **+6.68** |
| MMVP | 76.67 | **77.67** | **+1.00** |
| MMStar | 78.53 | **80.07** | **+1.54** |
| POPE | 88.28 | **89.31** | **+1.03** |
| **7 项平均** | **76.50** | **79.25** | **+2.75** |

跨模型蒸馏后的 Student 在 7 项 Benchmark 中有 6 项提升，其中 VisualProbe
（+6.68）和 V\*Bench（+6.29）的提升最明显；HR-Bench 4K 基本持平（-0.13）。
这表明分辨率特权的 On-Policy 监督可以把更大固定 Teacher 的细粒度视觉能力迁移
到更小的 Student。

这是一个包含 7 项 Benchmark 的补充实验，其平均分不能与论文主表中的 9 项平均分
直接比较。本次更新仅记录实验流程与结果，不改变仓库中标准的 Qwen3.5-9B 训练配置。
