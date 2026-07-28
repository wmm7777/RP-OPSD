# RP-OPSD

**面向多模态大语言模型的分辨率特权在线策略自蒸馏**

[English](README.md) · **论文：** [arXiv:2607.24447](https://arxiv.org/abs/2607.24447) · **数据集：** coming soon

RP-OPSD 将同一图像在不同分辨率下的能力差异作为特权信息：低分辨率 Student
生成 on-policy 轨迹，原始分辨率 Teacher 提供稠密的 Token 级监督。该方法只需
图像—问题对，不依赖外部 Teacher、额外生成的推理轨迹或区域标注。

本仓库提供论文中 Qwen3.5-9B 实验所使用的完整实现、训练、Checkpoint 合并、
验证与评测流程。训练和评测源码已直接包含在仓库中，配置环境和运行时均不会
克隆其他代码仓库。

## 方法

![RP-OPSD 方法概览](assets/rp_opsd_overview.png)

Student 接收宽高各缩小一半的图像，即原图四分之一的像素；EMA Teacher 接收
原始图像，并通过带偏差校正的 Teacher Top-100 反向 KL 目标，在 Student 采样
的前缀上提供监督。

| 设置 | 数值 |
|---|---|
| Student / Teacher 视图 | 半分辨率 / 原始分辨率 |
| Rollout | 8 个回答，temperature 1.0 |
| 蒸馏目标 | 带偏差校正的 Teacher Top-100 反向 KL |
| Teacher 更新 | EMA，更新率 0.05 |
| 优化配置 | Batch 96，学习率 2e-6，55 steps |
| 论文实验硬件 | 8 × NVIDIA H20 |

## 实验结果

在 Qwen3.5-9B 上，RP-OPSD 在原始分辨率评测下达到 **80.43** 的平均分，相比
Base 提高 **4.16 个百分点 / 5.45% 相对提升**。在半分辨率评测下，平均分提高
**6.09 个百分点**；相比 OPSD，训练速度提升至 **1.78×**（7.83 h vs. 13.93 h）。

| 方法 | V\* | HR-4K | HR-8K | MME-RW EN | MME-RW CN | VisualProbe | MMVP | MMStar | POPE | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 84.82 | 84.75 | 81.50 | 71.40 | 67.67 | 41.85 | 83.00 | 82.07 | 89.36 | 76.27 |
| SFT | 91.10 | **87.88** | 83.62 | 73.25 | 71.54 | 51.25 | 83.67 | 78.93 | 89.74 | 79.00 |
| GRPO | 88.48 | 84.50 | 81.50 | 75.72 | 71.81 | 50.38 | **84.33** | 80.73 | 89.09 | 78.50 |
| OPSD | 91.10 | 86.88 | 84.25 | **78.12** | **74.43** | 49.97 | 83.00 | 81.40 | 89.08 | 79.80 |
| Vision-OPD | 85.86 | 86.62 | **85.12** | 73.40 | 70.46 | 56.84 | 81.33 | 81.53 | **89.79** | 78.99 |
| **RP-OPSD** | **91.10** | 86.50 | 84.12 | 76.91 | 72.84 | **56.97** | 83.33 | **82.67** | 89.43 | **80.43** |

所有结果均使用原始分辨率图像，`Avg.` 为九项指标的无权平均。机器可读的参考
结果与评测协议见 [expected_metrics.json](expected_metrics.json) 和
[provenance/evaluation_protocol.json](provenance/evaluation_protocol.json)。

## 复现

### 环境要求

- Linux 与 Python 3.12
- 论文配置需要 8 张 NVIDIA GPU
- 本地 Qwen3.5-9B Base Checkpoint

固定环境使用 PyTorch 2.10.0、Transformers 5.5.0、vLLM 0.18.0 和 Ray 2.53.0。

> **数据状态：** Dataset4.0 及其准备说明将单独发布。当前公开包已包含数据校验
> 与物化流程。

### 快速开始

```bash
# 1. 创建固定环境并安装仓库内置的 RP-OPSD 源码。
./run.sh prepare-env

# 可选：校验原生源码 Manifest。
./run.sh verify --check-source

# 2. 运行单步 Smoke Test。
./run.sh smoke --model-path <Qwen3.5-9B> --asset-root <dataset-assets> \
  --output-dir outputs/smoke

# 3. 运行 55-step 训练。
./run.sh train --model-path <Qwen3.5-9B> --asset-root <dataset-assets> \
  --output-dir outputs/train

# 4. 合并最终 FSDP Checkpoint。
./run.sh merge --checkpoint-dir outputs/train/checkpoints/global_step_55 \
  --output-dir outputs/merged

# 5. 评测合并后的模型。
./run.sh eval --model-path outputs/merged --judge-model-path <Qwen3.5-9B> \
  --eval-data-dir <evaluation-data> --output-dir outputs/eval --prepare-data
```

评测使用原始分辨率图像、贪心解码、关闭 Thinking，并固定 seed 42。标准评测集
包括 V\*Bench、HR-Bench 4K/8K、MME-RealWorld EN/CN、VisualProbe、MMVP、
MMStar 和 POPE。

环境变量驱动的完整流程见 `./run.sh --help`。训练前可使用
`./run.sh verify --help` 检查代码包与输入。

## 目录结构

| 路径 | 内容 |
|---|---|
| `config/` | 标准实验配置 |
| `environment/` | 固定依赖版本 |
| `verl/` | 包含 RP-OPSD 目标的内置训练运行时 |
| `eval/` | 标准评测数据准备、推理、Judge 与评分实现 |
| `chat_templates/` | Qwen3.5 多模态 Chat Template |
| `provenance/` | 源码哈希与评测协议 |
| `scripts/` | 环境、训练、合并、验证与评测工具 |
| `tests/` | 软件包与发布安全测试 |

## 引用

```bibtex
@misc{zhu2026rpopsdresolutionprivilegedonpolicyselfdistillation,
      title={RP-OPSD: Resolution-Privileged On-Policy Self-Distillation for Multimodal Large Language Models},
      author={Qihui Zhu and Yuchen Wang and Zijian Wen and Tao Zhang and Mengjie Zhang and Yang Liu and Shuangwu Chen and Siying Wu and Jian Yang and Xiaofeng Jiang},
      year={2026},
      eprint={2607.24447},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.24447},
}
```

## 致谢

内置训练运行时包含基于 [Vision-OPD](https://github.com/VisionOPD/Vision-OPD) 固定提交
[`c2e345f`](https://github.com/VisionOPD/Vision-OPD/commit/c2e345fcab10c806ba83e2ec6e1e246d73e7aba2)。

## 许可证

代码使用 [Apache License 2.0](LICENSE) 发布。上游归属信息见
[UPSTREAM_LICENSE](UPSTREAM_LICENSE) 和 [NOTICE.md](NOTICE.md)。
