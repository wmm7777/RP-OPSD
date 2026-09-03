# 闪记 Summary 分辨率特权自蒸馏训练方案（RP-OPSD 复现）

> 编制人：meimei.wu　创建日期：2026-08-29
> 复现对象：[RP-OPSD](../README_zh.md)（Resolution-Privileged On-Policy Self-Distillation，[arXiv:2607.24447](https://arxiv.org/abs/2607.24447)）
> 复现目标：在 **flash_note summary** 任务（en / fr / ru / zh 四门语种）上跑通 RP-OPSD 自蒸馏训练，复用本仓库（`RP-OPSD/`）自带的 verl 训练代码
> 基础文档：[跨模型蒸馏实验](cross_model_distillation_zh.md)（同仓库补充实验，结构参照）
> 配套文档：[flash_note_opsd_训练方案.md](../../flash_note_opsd_训练方案.md)（注：那是另一套 27B 外部教师 + ms-swift GKD 方案，与本方案无关）

---

## 1. 方法概述

**RP-OPSD** 把同一图像在不同分辨率下的能力差异当作特权信息：低分辨率 Student 生成 on-policy 轨迹，原始分辨率 Teacher 在相同轨迹上提供稠密的 Token 级监督。只需图像—指令对，不依赖外部教师、额外生成的推理轨迹或区域标注，**reward-free**（无 gold 答案、无 reward model）。

### 1.1 自蒸馏（关键：无外部教师）

与 flash_note OPSD（27B 外部教师 + ms-swift GKD）根本不同，RP-OPSD 是**自蒸馏**：

- **Teacher = Student = Qwen3.5-9B**（同一个基模型）；
- Teacher 是 Student 的 **EMA 影子权重**（`teacher_regularization=ema`，`teacher_update_rate=0.05`），由 verl trainer 内部维护，**不需要部署任何外部教师服务**；
- 唯一的"特权"是**分辨率**：Student 看 1/2 物理分辨率图，Teacher(EMA) 看原始分辨率图；
- 师生 prompt 完全相同（无答案特权、无 ref）。

### 1.2 分辨率不对称特权

```
Student(Qwen3.5-9B)  看 【1/2 物理分辨率图 + 指令】→ 在线采样 8 条轨迹
Teacher(EMA 影子)    看 【原图 + 同一指令】       → 对同一轨迹逐 Token 打分
                         ↓
         Top-100 偏差校正反向 KL（reward-free）
                         ↓
                   更新 Student（EMA 慢速跟踪）
```

Student 在更苛刻的低分辨率条件下被纠正，迁回高分辨率推理时形成**超调增益**。

### 1.3 与论文/仓库的关系

| 维度 | 论文 RP-OPSD（主实验）| 本方案（flash_note summary）|
|------|------------------|---------------------------|
| 任务 | 视觉 QA（V\*Bench / HR-Bench / …，MCQ+OpenQA）| 图片 → 多语种摘要（en/fr/ru/zh）|
| 模型 | Qwen3.5-9B（自蒸馏）| Qwen3.5-9B（同，复现）|
| Teacher | EMA 影子（ρ=0.05）| EMA 影子（ρ=0.05，同）|
| Student 视图 | 物理半分辨率（宽高各 1/2）| 物理半分辨率（同）|
| 数据 | Dataset4.0，5,295 条 | summary，72,166 条（每门 2.2W，4 语种）|
| 评测 | 9 项视觉 Benchmark | 4 语种 summary MOS（LLM-as-judge）|
| 代码 | 本仓库 verl | **复用本仓库 verl**（仅换数据+评测）|

> 跨模型实验（[cross_model_distillation_zh.md](cross_model_distillation_zh.md)）是同仓库的补充实验（固定 9B Teacher → 4B Student，替代 EMA）。本方案先复现主实验（9B 自蒸馏）；如需跨模型变体，见 §10。

---

## 2. 训练流程（复用仓库 verl 代码）

仓库训练入口链路：`run.sh train` → `scripts/train.sh` → `scripts/run_rp_opsd.sh` → `verl.trainer.main_ppo --config-name vopd`。本方案**不改训练代码**，只换数据 parquet 与评测。

### 2.1 Student Rollout

Qwen3.5-9B Student 接收**物理半分辨率图**（宽高各缩至 1/2，像素量 1/4）。对每个问题在线采样 8 条回答，`temperature=1.0`、`top_p=1.0`、`top_k=-1`，最大回答长度 1024 Tokens。

### 2.2 Teacher Scoring（EMA，原图）

EMA Teacher（Student 的慢速影子权重）接收同一问题及**原始分辨率图像**。Teacher 不单独生成答案，而是在 Student 已生成的每条轨迹上，计算每个回答位置的条件 Token 分布。

### 2.3 Token 级目标

每个位置选取 Teacher 概率最高的 100 个 Token，在 Teacher Top-100 支持集上计算带截断偏差校正的反向 KL：

$$
\mathcal{L}_{\mathrm{distill}}
=
\sum_{v \in \operatorname{TopK}(p_t)}
\left[
p_s(v)\log\frac{p_s(v)}{p_t(v)} - p_s(v) + p_t(v)
\right].
$$

配置（`config/best.yaml` / `config/best.env`）：

| 项 | 值 |
|----|----|
| `distillation_objective` | `mopd_topk_reverse_kl`（bias-corrected reverse KL）|
| `distillation_topk` | 100 |
| `distillation_topk_source` | teacher |
| `distillation_add_tail` | False（不加 Tail Bucket）|
| `alpha` | 0.5（作者默认，复现不动）|
| `is_clip` | 2.0（importance sampling 截断）|
| `reward_model.enable` | False（reward-free）|
| `teacher_regularization` | ema，`teacher_update_rate=0.05` |

### 2.4 参数更新

只更新 Qwen3.5-9B Student（全参数 FSDP，非 LoRA），EMA Teacher 慢速跟踪 Student 权重。Student 视觉编码器**参与训练**（论文 RP-OPSD 不 freeze_vit，与 flash_note OPSD 的 `freeze_vit=true` 不同）。

---

## 3. 数据准备

### 3.1 源数据

| 项 | 值 |
|----|----|
| 源文件 | `/data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl` |
| 源条数 | 96,348 |
| 源语种分布 | en 40,000 / fr 28,182 / ru 14,773 / zh 13,393 |
| **采样后条数** | **72,166**（每门封顶 2.2W：en 22,000 / fr 22,000 / ru 14,773 / zh 13,393）|
| 唯一图数 | 69,305（72166 行中有 2861 行图复用）|
| 原图目录(teacher) | `/data4/wumeimei/flash_note/train/<lang>_image/*.jpg` |
| 半分辨率图(student) | `/data4/wumeimei/flash_note/train/<lang>_image_lr/*.jpg`（已生成，共 69,305 张）|
| 提示词 | 4 门语种各一份全文翻译（非英文换词），见 `convert_flashnote_summary.py` 的 `PROMPT_TEMPLATE_<lang>` |
| 转换产物 | `/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/train.parquet`（4.9 MB）|

源 jsonl 每条结构：`messages`（1 条 user，content 含 `<image>` 占位符 + summary 指令）/ `images`（原图绝对路径）/ `teacher_prompt`（本方案不用，自蒸馏无答案特权）。

### 3.2 Parquet schema（仓库期望）

verl 的 `RLHFDataset`（`verl/utils/dataset/rl_dataset.py`）从 parquet 读以下列：

| 列名 | 类型 | 内容 |
|------|------|------|
| `prompt` | `list[dict]` | messages，如 `[{"role":"user","content":"<image>\n<summary 指令>"}]`，content 为含 `<image>` 占位符的字符串 |
| `images` | `list[dict]` | Student 用图，`[{"image": "<半分辨率图路径>"}]` |
| `teacher_images` | `list[dict]` | Teacher 用图，`[{"image": "<原图路径>"}]`（`teacher_image_key=teacher_images`，见 train.sh）|
| `extra_info` | `dict` | `{"index": i, "task_family": "summary_<lang>"}` 等 |

> `images` / `teacher_images` 路径在 portable release 中为相对路径（`assets/student/...`、`assets/teacher/...`），由 `materialize_data.py` 物化为绝对路径。本方案可直接写绝对路径（跳过 portable release 流程，见 §3.4）。

#### 3.2.1 完整样例（English）

每行一条样本，4 列结构如下（以 en 为例）：

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "<image>\nCore Identity\nYou are a professional image-text comprehension and summary expert. You specialize in extracting valid key information from images and generating accurate, concise, well-structured English summaries.\nMandatory Rules (Highest Priority)\nAll outputs must strictly comply with the following hard constraints. Any violation makes the output invalid:\n1. Language Rule: Output only English, no other languages are allowed.\n2. Authenticity Rule: Only use recognizable content from the image. No subjective speculation, no external knowledge, no fabricated information.\n3. Length Rule: The total summary length must not exceed 500 English words.\n4. Quality Rule: Extract core information only. Do not copy original content mechanically, and avoid redundant or empty descriptions.\nTask & Output Standards\nGenerate a standard English summary based on the visible content of the image. Adapt your output structure according to the information density of the image:\nMode 1: Information-Dense Images\nApplicable to images containing multiple key points, data, steps, opinions or rules. Use structured Markdown format:\n1. Opening: 1–2 concise sentences to summarize the core theme of the image (required).\n2. Main content: List key information in ordered or unordered lists.\n3. List specification: Each list item must start with a short bold subheading, followed by a concise and objective explanation based strictly on the image content.\nMode 2: Simple & Low-Density Images\nApplicable to images with a single topic or simple content. Use plain paragraph natural language without redundant Markdown formatting, and accurately output the only core information.\nSupplementary Rules\n1. If the image contains extremely limited information, output the most accurate and concise summary available.\n2. All Markdown symbols are only for layout and do not count towards the word limit.\n3. Keep all content logical, objective and refined."
    }
  ],
  "images": [
    {"image": "/data4/wumeimei/flash_note/train/en_image_lr/b7f39493c2148c7a32c4226c57cc9231187d7c530bb8a42d098295b7f79d138c.jpg"}
  ],
  "teacher_images": [
    {"image": "/data4/wumeimei/flash_note/train/en_image/b7f39493c2148c7a32c4226c57cc9231187d7c530bb8a42d098295b7f79d138c.jpg"}
  ],
  "extra_info": {"index": 0, "task_family": "summary_en"}
}
```

#### 3.2.2 完整样例（中文）

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "<image>\n核心身份\n你是专业的图文理解与摘要撰写专家，擅长从图片中提取有效关键信息，生成精准、简洁、结构清晰的中文摘要。\n强制规则（最高优先级）\n所有输出必须严格遵守以下硬性要求，违反任意一条即视为输出无效：\n1. 语言规则：仅输出中文内容，禁止出现其他语种文字。\n2. 内容真实规则：仅使用图片中可识别的内容，禁止主观推测、引用外部知识、编造虚假信息。\n3. 字数规则：摘要总字数不得超过500个汉字。\n4. 质量规则：仅提炼核心有效信息，禁止机械照搬原图内容，杜绝冗余、空洞描述。\n任务与输出规范\n根据图片可视内容生成标准摘要，依据图片信息密度自适应匹配输出结构：\n模式一：高信息密度图片\n适用于包含大量要点、数据、步骤、观点、规则的图片，采用Markdown结构化格式输出：\n1. 开篇：用1-2句精简语句概括图片核心主题（必填）。\n2. 主体内容：以有序或无序列表展示关键信息。\n3. 列表规范：每条内容必须以简短加粗小标题开头，后跟严格贴合原图、客观简洁的文字说明。\n模式二：简单低信息密度图片\n适用于单一主题、内容简单的图片，采用纯段落自然语言输出，无需多余Markdown格式，精准输出唯一核心信息即可。\n补充规则\n1. 若图片有效信息极少，输出最精准、最简洁的核心摘要即可。\n2. Markdown符号仅用于排版，不计入字数统计。\n3. 输出内容全程逻辑清晰、客观中立、精炼无冗余。"
    }
  ],
  "images": [
    {"image": "/data4/wumeimei/flash_note/train/zh_image_lr/321a73feb0b6917de96e148bd9b279826431d2e5e1bf2c0d843e1f52b90f5df0.jpg"}
  ],
  "teacher_images": [
    {"image": "/data4/wumeimei/flash_note/train/zh_image/321a73feb0b6917de96e148bd9b279826431d2e5e1bf2c0d843e1f52b90f5df0.jpg"}
  ],
  "extra_info": {"index": 1, "task_family": "summary_zh"}
}
```

> 样例要点（reward-free 自蒸馏，无 gold/assistant）：
> - `prompt` 只有 **1 条 user message**，`content` = `<image>\n` + 语种任务 prompt（Core Identity 风格），**不含** assistant、不含 gold summary、不含 `teacher_prompt` 字段内容、不含 `【Reference Summary】` 等生成标记
> - `images` 指向 Student 用的**半分辨率图**，`teacher_images` 指向 Teacher 用的**原图**（同一张，不同分辨率）
> - `extra_info.task_family` = `summary_<lang>`，用于训练时分语种统计

#### 3.2.3 复核脚本（训练前必跑）

```python
import pandas as pd
df = pd.read_parquet(".runtime/flashnote_summary/train.parquet")
assert (df['prompt'].apply(len) == 1).all(), "prompt 应只有 1 条 message"
assert df['prompt'].apply(lambda x: x[0]['role']).eq('user').all(), "role 必须是 user"
for kw in ["Reference Summary", "Role Setting", "参考摘要", "【参考摘要】"]:
    leak = df['prompt'].apply(lambda x: kw in x[0]['content']).sum()
    assert leak == 0, f"prompt 含泄漏关键词 '{kw}': {leak} 行"
print(f"✅ {len(df)} 条 RP-OPSD parquet 格式检查通过")
```

### 3.3 生成半分辨率图

RP-OPSD 的 Student 视图是**物理半分辨率**（`student_view: physical half width and height`，宽高各 1/2，像素量 1/4 → 视觉 token ~1/4 → rollout 加速 1.78×），不是 pixelation 模糊。

脚本：`/data4/wumeimei/flash_note/train/gen_lr_images.py`（多进程，支持 `--from-parquet` 只处理 parquet 里实际用到的图，避免对全量 8.6 万张 en 图无谓下采样）。

```bash
# 只对采样后 parquet 里 teacher_images 列的图生成 LR（去重，69,305 张）
python /data4/wumeimei/flash_note/train/gen_lr_images.py \
  --from-parquet .runtime/flashnote_summary/train.parquet
# 或整目录全量生成（备选）
python /data4/wumeimei/flash_note/train/gen_lr_images.py --langs en fr ru zh
```

### 3.4 转换脚本

`scripts/convert_flashnote_summary.py`：读 summary jsonl → 生成 `train.parquet`，列如 §3.2。每条 `prompt` **按语种套用对应翻译模板**（`PROMPT_TEMPLATE_en/_fr/_ru/_zh`，不从 jsonl 透传英文模板），`images` 指向半分辨率图，`teacher_images` 指向原图，`extra_info` 填 index + `task_family=summary_<lang>`（lang 从图片路径 `/(en|fr|ru|zh)_image/` 检测）。每门语种封顶 `--max-per-lang 22000`。

```bash
python scripts/convert_flashnote_summary.py \
  --src /data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl \
  --out .runtime/flashnote_summary/train.parquet \
  --max-per-lang 22000            # 每门封顶 2.2W
python scripts/convert_flashnote_summary.py --out ... --check-lr   # 校验 LR 图齐
python scripts/convert_flashnote_summary.py --show-prompt --lang zh  # 看模板
```

> 绕过 `prepare_data.py`（Dataset4.0 校验/归档，强校验 5295 行）与 `materialize_data.py`（portable→绝对路径），直接产绝对路径 parquet 喂给 `run_rp_opsd.sh` 的 `data.train_files`。

---

## 4. 训练设置

复用 `config/best.env` 口径，仅按 flash_note summary 调整数据量与响应长度：

| 设置 | 论文值 | 本方案 | 备注 |
|------|--------|--------|------|
| 模型 | Qwen3.5-9B | Qwen3.5-9B | 本地 `/data4/wumeimei/download_models/Qwen3.5-9B` |
| 数据 | Dataset4.0 5,295 条 | summary 72,166 条 | 4 语种，每门封顶 2.2W |
| Student 视图 | 物理半分辨率 | 物理半分辨率 | 同 |
| Teacher 视图 | 原始分辨率 | 原始分辨率 | 同 |
| Rollout 数 | 8 | 8 | temp 1.0 / top_p 1.0 / top_k -1 |
| 最大回答长度 | 1024 | 1024 | 摘要 ≤500 单位，1024 token 兜底防截断 |
| 最大 prompt 长度 | 8192 | 5120 | 数据集实测文本 token mean=395 + 图像 ~1280 ≈ 1700，6k 口径留 4x 余量 |
| Batch Size | 96 | 96 | |
| 学习率 | 2e-6 | 2e-6 | 全参数 |
| Warmup | 10 Steps | 10 Steps | |
| 蒸馏目标 | Top-100 reverse KL | Top-100 reverse KL | alpha=0.5，无 tail，is_clip 2.0 |
| Teacher | EMA ρ=0.05 | EMA ρ=0.05 | 自蒸馏，无外部教师 |
| 训练长度 | 55 Steps（1 epoch）| 见 §4.1 | |
| save_freq | 55 | 按 §4.1 | |
| GPU | 8 × H20 | 8 × H20 | `trainer.n_gpus_per_node=8` |

### 4.1 训练长度与时间估算

- 论文：5,295 条 / batch 96 = **55 步 = 1 epoch**，8×H20 耗时 **7.83h**（≈8.5 min/step）。
- 本方案：72,166 条 / batch 96 = **~752 步 = 1 epoch**。按 8.5 min/step 计，1 epoch ≈ **106h（≈4.4 天）**，代价大。
- **建议分阶段**：
  1. **Smoke**（1 步）：`./run.sh smoke --model-path ... --asset-root ... --output-dir outputs/smoke`，验证诊断指标（`self_distillation/teacher_image_swap_fraction=1.0`、`actor/vopd_loss>0`、`self_distillation/num_distill_tokens>0`，指标定义见 §5.4.5）；
  2. **55 步验证**（取 5,280 条子集，1 epoch）：验证机制在 summary 任务上成立（loss 收敛、不 drift）；
  3. **全量训练**（~752 步 / 1 epoch，或按算力截断到如 300–500 步）：机制成立后再上。

> 步数通过 `--steps N` 传给 `train.sh`（透传 `trainer.total_training_steps`）。子集采样可在转换脚本里按 lang 分层抽 5,280 条。

---

## 5. 部署与启动

### 5.1 无外部教师部署

RP-OPSD 自蒸馏，**不需要部署教师服务**。verl trainer 内部维护 EMA 影子权重，8 卡全部用于 Student 训练（actor + rollout + ref 共置）。

### 5.2 环境

```bash
# 仓库固定环境（Python 3.12 + PyTorch 2.10 + Transformers 5.5 + vLLM 0.18 + Ray 2.53）
./run.sh prepare-env
# 或指定已有 python
./run.sh prepare-env --python /path/to/python3.12 --env-dir .runtime/venv
```

> ⚠️ 仓库要求 `flash-attn` + `causal-conv1d` 内核（`--skip-kernels` 可跳过但训练需装）。本地若已有 swift 环境的内核 wheel 可复用：`FLASH_ATTN_WHEEL=... CAUSAL_CONV_WHEEL=... ./run.sh prepare-env`。

### 5.3 启动

```bash
# 1) 转换 summary → parquet（每门封顶 2.2W，套用翻译模板）
python scripts/convert_flashnote_summary.py \
  --out .runtime/flashnote_summary/train.parquet --max-per-lang 22000

# 2) 只对 parquet 里用到的图生成半分辨率 LR（仅一次，69,305 张）
python /data4/wumeimei/flash_note/train/gen_lr_images.py \
  --from-parquet .runtime/flashnote_summary/train.parquet

# 3) 校验 LR 图齐（应为 0 缺失）
python scripts/convert_flashnote_summary.py --out .runtime/flashnote_summary/train.parquet --check-lr

# 4) Smoke（1 步验证）
./run.sh smoke --model-path /data4/wumeimei/download_models/Qwen3.5-9B \
  --asset-root <含 assets/student + assets/teacher 的根> \
  --output-dir outputs/flashnote_smoke
# 注：smoke/asset-root 校验针对 Dataset4.0，若用绝对路径 parquet 需绕过 train.sh 的
#     materialize/verify 步骤，直接调 run_rp_opsd.sh（见 §5.4）

# 4) 训练
./run.sh train --model-path /data4/wumeimei/download_models/Qwen3.5-9B \
  --output-dir outputs/flashnote_rp_opsd --steps 55
```

### 5.4 绕过 Dataset4.0 校验直接训练

`train.sh` 内置 `materialize_data.py` + `verify.py`（强校验 5295 行 + asset 归档），针对 Dataset4.0。flash_note 数据需绕过，直接调底层 `run_rp_opsd.sh`：

```bash
source .runtime/venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_V1=1 RP_OPSD_LAUNCHER_CHECKPOINT_DIR=outputs/flashnote_rp_opsd/checkpoints
RP_OPSD_LAUNCHER_ROLLOUT_DIR=outputs/flashnote_rp_opsd/rollouts

bash scripts/run_rp_opsd.sh \
  "data.train_files=[\".runtime/flashnote_summary/train.parquet\"]" \
  "data.val_files=[]" \
  "data.image_key=images" \
  "actor_rollout_ref.model.path=/data4/wumeimei/download_models/Qwen3.5-9B" \
  "actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images" \
  "trainer.total_training_steps=55" \
  "trainer.save_freq=55" \
  # ... 其余沿用 run_rp_opsd.sh 默认（已在脚本里设好 vopd/topk100/ema0.05/alpha/batch96/lr2e-6）
```

> `run_rp_opsd.sh` 默认 `MODEL_PATH=Qwen/Qwen3.5-4B`、`teacher_image_key=bbox_images`、`ALPHA=0.5`，需在命令行覆盖为 9B + `teacher_images` + `alpha=1.0`（对齐 best.env）。或直接改脚本默认值。

#### 5.4.1 实测启动命令（2026-08-29 跑通，v2 单文件入口）

8 卡 H20 + Qwen3.5-9B + verl_opd_flashnote 环境的实测启动模板。`scripts/run_rp_opsd_v2.sh` 是单文件入口，所有变量在脚本顶部统一配置，不嵌套任何其它脚本，改参数只改顶部：

```bash
tmux new -d -s rp_opsd_v3 "bash /data4/wumeimei/flash_note/RP-OPSD/scripts/run_rp_opsd_v2.sh"
```

脚本顶部关键字段（5k 口径 + mopd 三件套，详见 §5.4.2 / §5.4.6）：

```bash
# 长度字段（5k 口径，2026-09-01 从 6k 降，压低 actor forward/backward 峰值显存）
MAX_PROMPT_LENGTH=3072
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))   # 5120
PPO_MAX_TOKEN_LEN_PER_GPU=$MAX_MODEL_LEN                      # 必须等于 MAX_MODEL_LEN

# 路径
OUTPUT_DIR="$PROJECT_ROOT/outputs/flashnote_train_v4"
TRAINER_DEFAULT_LOCAL_DIR="$OUTPUT_DIR/checkpoints"
TRAINER_ROLLOUT_DATA_DIR="$OUTPUT_DIR/rollouts"

# 蒸馏目标（官方 canonical 必传，漏传则 verl 默认 generalized_jsd 致退化，详见 §5.4.6）
DISTILLATION_OBJECTIVE="mopd_topk_reverse_kl"   # 论文 Eq.(5) bias-corrected reverse-KL
DISTILLATION_TOPK_SOURCE="teacher"               # teacher 选 top-k support
DISTILLATION_ADD_TAIL=False                       # no tail bucket, no renormalization
```

日志自动 `tee` 到 `$OUTPUT_DIR/logs/train.log`。旧 `run_rp_opsd.sh` / `run_rp_opsd.bak.sh` 已废弃，新增需求一律在 `run_rp_opsd_v2.sh` 顶部改。

> 旧入口 `run_rp_opsd.sh` 内部仍调用 `.bak.sh` 且硬设 `use_dynamic_bsz=False` + 55 step smoke 语义，与 2 epoch 长训练不一致，不再推荐。

#### 5.4.2 关键踩坑（必读，否则 OOM）

| 参数 | 作者默认 | ❌ 错误覆盖 | 后果 |
|---|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.7 | 0.5 | CUDA OOM @ `dp_actor.py:703` 的 `torch.logsumexp(logits, dim=-1)` |
| `actor_rollout_ref.model.use_remove_padding` | True | False | 同上，关掉 padding 优化让 logits 矩阵更大 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | ≥ `max_prompt_length + max_response_length` | < `max_model_len` | `AssertionError: max_token_len must be greater than the sequence length` @ `seqlen_balancing.py:382`，进程在 step 1 的 `update_actor` 阶段直接死，8 卡 vLLM engine core 全部退出 |
| `self_distillation.distillation_objective` / `topk_source` / `add_tail` | `mopd_topk_reverse_kl` / `teacher` / `False` | **漏传→verl 默认 `generalized_jsd`/`student`/`True`** | ⚠️ **静默退化无报错**：verl 校验只在 objective==mopd 时触发（`actor.py:149-171`），漏传不进校验分支→走 generalized_jsd（forward JSD + student 选 support + tail bucket）→ EMA teacher 正反馈退化，详见 §5.4.6 |

**9B 模型必须保留作者默认 0.7 + True**。实测峰值显存 `perf/max_memory_allocated_gb=125.3`（卡 97GB ×8 卡 + FSDP offload 兜底）。降到 0.5 + 关 remove_padding 时 actor 占 85.97GB，logsumexp 那步分不出 9.47GB 直接 OOM。

**长度字段统一口径（6k，2026-08-29 数据集实测后统一定稿）**：

```
data.max_prompt_length = 5120          ┐
                                       ├─ 6144 = max_model_len = max_num_batched_tokens
data.max_response_length = 1024        ┘    = ppo_max_token_len_per_gpu (use_dynamic_bsz=True 时必须 ≥)
```

数据集实测：72,166 条 summary 样本，prompt 文本 token mean=395（固定模板）+ 图像 token ~1280（student 半分辨率单图）≈ 1700，p99 同 mean。6144 远覆盖且留 4x 余量，不再用旧的 8192+1024=9216 口径。

**改长度字段的链式约束**（改一个就要全链复查，否则又会 OOM 或 assert）：
- `data.max_prompt_length` + `data.max_response_length` = `actor_rollout_ref.rollout.max_model_len` = `actor_rollout_ref.rollout.max_num_batched_tokens`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` 必须 ≥ 上式结果（`use_dynamic_bsz=True` 时）；`use_dynamic_bsz=False` 时断言不触发但仍建议设为相等
- `actor_rollout_ref.rollout.response_length` = `data.max_response_length`
- 所有上述字段在 `scripts/run_rp_opsd_v2.sh` 顶部「长度字段」段统一配置（见 §5.4.1），改只改那一处；`config/best.env` 同步保持一致

`distillation_topk=100`（top100）在 `run_rp_opsd_v2.sh` 顶部 `DISTILLATION_TOPK=100` 配置，跟 OOM 无关——logsumexp 在 topk 之前算全词表，top100 只影响 loss 层。

#### 5.4.3 实测速度与 ckpt 体积

- **单 step 实测 3.9 min**（H20，比 doc §4.1 估算 8.5 min/step 快一倍）
- 全集 72,166 条 / batch 96 = 752 step/epoch
- 2 epoch = 1504 step ≈ **4.07 天**
- 单个 ckpt = **106GB**：8 × `model_*.pt` (35.2GB) + 8 × `optim_*.pt` (70.4GB) + 元数据
  - `model_*.pt` 4.4GB/rank × 8 = actor FSDP shards
  - `optim_*.pt` 8.8GB/rank × 8 = optimizer state（占 2/3 体积）
- 10 个 ckpt × 106GB = **1.06TB**，本机单盘装不下（/data4 716GB、/data3 510GB、/data2 无写入权限）

**减小 ckpt 体积**：覆盖 `actor_rollout_ref.actor.checkpoint.save_contents=['model']` 只存模型权重不存 optimizer state → 单 ckpt 35GB，10 个 352GB 可入 /data4。代价：resume 时 optimizer state 会重置（lr 重新 warmup），不能完美续训。

#### 5.4.4 训练完成后的 ray teardown 噪声

训练正常完成、ckpt 已落盘后，ray 在 teardown 阶段会抛：

```
RuntimeError: DataLoader worker (pid=XXX) is killed by signal: Killed.
```

这是 ray 清理 dataloader worker 时的已知噪声，**不影响训练结果**。看到这个不算训练崩，看 `Training Progress: 100%` 和 `global_step_N/` 目录存在即可确认成功。

#### 5.4.5 过程指标监控与趋势分析

训练日志每 step 打一行 metrics（`TaskRunner pid=...` 开头），关键字段分 6 类。**注意：旧文档提到的 `teacher_topk_mass` / `distillation_loss` 已不在当前 verl 输出里**（源码 `core_algos.py:1148` 有定义但本配置 `distillation_add_tail=False` 不触发），实际看下面的指标。

**① 蒸馏是否生效（最关键，=0 即说明 RP-OPSD 机制没起作用）**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `self_distillation/num_distill_tokens` | 每 batch 蒸馏覆盖的 token 数 | >0 且稳定（实测 ~2600） | =0 蒸馏完全失效 |
| `self_distillation/self_distillation_mask.mean()` | 蒸馏 mask 覆盖率 | =1.0 | <1 说明部分 token 无 teacher 监督 |
| `self_distillation/empty_target_batch` | 空 target batch 占比 | =0.0 | >0 说明 teacher 没出分 |
| `self_distillation/teacher_always_on_fraction` | teacher 始终在线比例 | =1.0 | <1 说明 teacher 间歇掉线 |
| `self_distillation/teacher_image_swap_fraction` | teacher 用原图比例 | =1.0 | <1 说明分辨率特权没生效 |
| `self_distillation/policy_fallback_fraction` | 退化到 policy gradient 的比例 | =0.0 | >0 说明蒸馏目标缺失、退化成纯 PG |
| `self_distillation/grpo_fallback_count` | GRPO 退化次数 | =0.0 | >0 说明 advantage 异常 |

**② 师生分布对齐（RP-OPSD 的核心目标）**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `actor/vopd_loss` | vopd 蒸馏损失（bias-corrected reverse KL） | 缓慢下降（0.14 左右起步） | 突然飙升 = 师生分布炸了 |
| `rollout_corr/kl` | 师生 KL 散度 | 0.2~0.5，缓慢下降 | >1 师生差太大；趋 0 没学到东西 |
| `rollout_corr/k3_kl` | 3 阶 KL（更敏感） | 跟 `kl` 同趋势 | 与 `kl` 背离 = 高阶矩不匹配 |
| `rollout_corr/ppl_ratio` | training_ppl / rollout_ppl | ~1.0（师生分布接近） | >10 师生分布严重不一致 |
| `rollout_corr/log_ppl_diff` | log ppl 均值差 | <0.5 且稳定 | 持续上涨 = 师生发散 |
| `rollout_corr/chi2_token` | token 级 chi2 | <5 | 突然飙大 = 师生 token 分布炸 |

**③ 重要性采样健康（on-policy 校正）**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `rollout_corr/rollout_is_mean` | IS 均值 | ~1.0 | 偏离 >0.5 说明 off-policy 严重 |
| `rollout_corr/rollout_is_std` | IS 标准差 | <1.0 | >2 说明 IS 分布太散 |
| `rollout_corr/rollout_is_max` | IS 最大值 | <5 | >10 个别样本 off-policy 失控 |
| `rollout_corr/rollout_is_ratio_fraction_high` | IS>2 的比例 | <0.1 | >0.3 大量样本需要重采样 |
| `rollout_corr/rollout_is_eff_sample_size` | 有效样本数 | 接近实际 batch | <batch/2 说明有效信息极少 |

**④ 训练稳定性**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `actor/grad_norm` | 梯度范数 | 10~30 稳定 | 突然 >100 = 训练发散 |
| `actor/lr` | 学习率 | warmup 0→2e-6（前 10 step），之后稳 2e-6 | 没涨 = warmup 异常 |
| `rollout_corr/training_ppl` | 训练 ppl | 稳定或缓降 | 暴涨 = student 崩了 |
| `rollout_corr/rollout_ppl` | rollout ppl | 跟 training_ppl 接近 | 偏离 = 师生分布漂移 |

**⑤ 生成质量（summary 任务特有）**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `response_length/mean` | 平均生成长度 | 稳定（~270 token） | 突降到 <50 = mode collapse；飙到 2048 = 失控 |
| `response_length/clip_ratio` | 达到 2048 截断的比例 | <0.05 | >0.2 大量样本被截断 |
| `response/aborted_ratio` | 生成中断率 | =0.0 | >0 说明 rollout 出错 |

**⑥ 效率与显存**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `perf/max_memory_allocated_gb` | 峰值显存 | <125GB（9B + 8卡） | 接近 145GB reserved 上限要小心 |
| `timing_s/step` | 单步耗时 | 稳定 ~210s（3.5min） | 突然翻倍 = IO/显存瓶颈 |
| `perf/mfu/actor` | actor MFU | 0.2~0.3 | <0.1 说明计算效率低 |
| `perf/throughput` | 吞吐（tokens/s） | >300 | 持续下降 = 卡顿 |

**趋势判读口诀**

- **看趋势不看绝对值**：单 step 值意义有限，看连续 10+ step 的趋势。tensorboard 日志在**仓库根** `tensorboard_log/RP-OPSD/<experiment_name>/`（verl `tracking.py:264` 相对 cwd 写仓库根，**非** `output_dir` 下；按 experiment 分子目录，如 `RP-OPSD-Qwen3.5-9B/`=v3、`-v3sft/`、`-v4-m1-4gpu/`）。训练 venv 未装 tensorboard，用 swift 环境启动：`/data1/meimei.wu/miniforge3/envs/swift/bin/tensorboard --logdir tensorboard_log/RP-OPSD --port 6007 --bind_all`。
- **先看①**：① 任一 = 0 → 蒸馏机制没起，后续指标都没意义，停下来 debug。
- **②+④ 联看**：`vopd_loss` 下降 + `kl` 缓降 + `grad_norm` 稳 = 健康训练；`vopd_loss` 降但 `kl` 涨 = student 在 collapse，蒸馏目标被钻空子。
- **③ 是 on-policy 健康度**：`rollout_is_mean` 偏离 1 + `rollout_is_max` 飙大 = on-policy 假设破坏，要降 `is_clip` 或缩短 rollout-update 间隔。
- **⑤ mode collapse 信号**：`response_length` 突降 + `clip_ratio=0` + `kl→0` = student 坍缩到固定输出，停训。
- **⑥ 显存**：`max_memory_allocated_gb` 持续上涨 = 显存泄漏，会 OOM。

**监控命令**

```bash
# 看最新 step 全量 metrics
grep "step:" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/logs/train.log | tail -1

# 只看关键 6 指标的趋势（最近 20 step）
grep "step:" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/logs/train.log | tail -20 | \
  grep -oE "step:[0-9]+|actor/vopd_loss:np.float64\([0-9.eE+-]+\)|rollout_corr/kl:np.float64\([0-9.eE+-]+\)|actor/grad_norm:np.float64\([0-9.eE+-]+\)|rollout_corr/ppl_ratio:np.float64\([0-9.eE+-]+\)|response_length/mean:[0-9.]+|self_distillation/num_distill_tokens:np.float64\([0-9.eE+-]+\)"

# 训练进度
grep "Training Progress" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/logs/train.log | tail -1

# 报错检查
grep -iE "out of memory|cuda.*error|Traceback|RuntimeError.*killed" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/logs/train.log | tail -5
```

#### 5.4.5.1 v3 实测健康分析（2026-09-02，step428）

v3（`flashnote_train_v4`，base 9B + mopd 修复后）运行至 step428/1502（2 epoch 计划，完成约 28%）。用 `scripts/analyze_loss.py` 解析 train.log 全量 step 行，六类指标实测如下。

**进度**：step 151–428（解析 278 步，前/中/后三段 + 末 5 步均）

| 类别 | 关键指标 | 实测（前段均→末5步均） | 判读 |
|---|---|---|---|
| ①蒸馏生效 | num_distill_tokens | 1663→1938，全程 >0 | ✅ 缓升，监督覆盖扩大 |
| | teacher_image_swap_fraction | =1.0 全程 | ✅ 分辨率特权生效 |
| | policy_fallback / grpo_fallback | =0 / =0 全程 | ✅ 未退化到 PG |
| ②师生对齐 | **actor/vopd_loss** | 0.0758→0.0099（**-87%**） | ✅ 平滑单调收敛 |
| | rollout_corr/kl | 0.057→0.064（区间 0.05–0.13） | ✅ 缓降 |
| | mopd_* 四指标 | 全程存在 | ✅ 走对 mopd 分支（§5.4.6 铁证） |
| | tkmass 缺口(teacher-student) | 0.020→0.009 | ✅ 师生分布在靠近 |
| ③IS | is_mean / is_std / is_max | 1.0 / 0 / 1.0 | ✅ 完全 on-policy |
| ④稳定性 | actor/grad_norm | max 25.3，末 0.7 | ✅ 远 <100 发散线 |
| | actor/lr | 稳 2e-6 | ✅ warmup 已过 |
| ⑤生成质量 | response_length/mean | 304→383（+26%，0.32 tok/step） | ⚠️ 温和上涨（未失控） |
| | response_length/clip_ratio | 0.0075→0.022 | ✅ <0.05 健康线 |
| ⑥效率显存 | perf/max_memory_allocated_gb | 112GB 稳定 | ✅ <125 |
| | timing_s/step | 195→200 | ✅ 3.3min/step |

**弱信号（未失控，需持续盯）**：

1. **resp_len 温和上涨**（304→383，斜率 0.32 tok/step）：对比 v2 退化是 265→737（+177%），v3 仅 +26% 且 max=422 未失控、clip_ratio<0.03。是轻度 drift 倾向，**非 mode collapse**（collapse = 长度突降 + clip=0 + kl→0，v3 三项相反）。配合 vopd 持续下降，属于"对齐中生成变长"而非坍缩。
2. **chi2_token 偶发尖峰**：中位 p50=0.42 / p90=1.16 健康，但有 7 个 step >10，**最新 step428=147 为全程最高**。偶发尖峰是个别 batch 师生 token 分布瞬时偏离；若后续 step 持续 >10 = 师生分布炸的前兆，停训排查。
3. **rollout_ppl 上涨 18→162**：配合 resp_len 涨，student 生成更长更多样；但 ppl_ratio 从 340 降到 122（师生/训练-采样分布对齐中，趋势向好）。摘要开放式生成 ppl 绝对值参考意义有限，看 ppl_ratio 下降这个好信号。

**结论**：训练健康，继续。vopd_loss 持续收敛（-87%）+ 蒸馏全程生效 + mopd 分支正确 + grad/显存稳定。EMA teacher 的 `teacher_topk_mass_mean` 从 0.979 降到 0.828（分布变平、监督能力衰减），印证 §12.2 结论——mopd 推迟退化但不根治，根治需 v4 固定 teacher。

**继续训期间监控判据**：

| 信号 | 阈值 | 动作 |
|---|---|---|
| resp_len 加速上涨 | 破 500 | 考虑早停 |
| clip_ratio | 破 0.1 | 考虑早停 |
| step428 后 chi2_token | 持续 >10 | 停训排查师生分布 |
| actor/vopd_loss | 反弹上涨 | 蒸馏失效，停训 |
| actor/grad_norm | 突 >100 | 训练发散，停训 |

**分析脚本**（`scripts/`）：

```bash
python scripts/analyze_loss.py    # 六类指标分段趋势 + 关键诊断（① 蒸馏/分支/vopd/resp_len/kl/grad/显存/速度）
python scripts/detail_metrics.py  # 可疑指标逐 step 序列 + chi2/resp_len 尖峰定位 + 线性斜率
```

#### 5.4.6 ⚠️ 配置漏传事故复盘（2026-09-01）

**症状**：v2 运行（`flashnote_train_v2`）在 step175 后 `response_length/mean` 从 ~265 持续暴涨到 737，`max` 长期撞 2048 上限——典型正反馈退化。用户质疑"原始论文 teacher 也是 EMA，为啥我的 summary 场景就不行了"。

**根因**：`run_rp_opsd_v2.sh` **漏传 3 行蒸馏配置**，verl 静默走了错误分支：

| 配置 | 官方 canonical（best.yaml:40-43）| verl 默认（actor.py:88-90）| v2 实际 |
|---|---|---|---|
| `distillation_objective` | `mopd_topk_reverse_kl` | `generalized_jsd` | ❌ 漏传→默认 |
| `distillation_topk_source` | `teacher` | `student` | ❌ 漏传→默认 |
| `distillation_add_tail` | `false` | `True` | ❌ 漏传→默认 |

致命点：verl 配置校验（`actor.py:149-171`）**仅在 `objective==mopd` 时才触发断言**。漏传时 `objective=generalized_jsd` 不进校验分支 → **不报错，静默走错**。所以"clone 官方仓库"≠"用官方配置"——`run_rp_opsd.sh` 第 155-159 行传了这 3 行，但自写的 `run_rp_opsd_v2.sh` 丢了。

**TB 运行时铁证**（区分走哪个分支的唯一可靠方法）：
- `core_algos.py:1133-1146` 的 `mopd_topk_reverse_kl` 分支输出 4 个专属指标（:1147-1151）：`mopd_reverse_kl_term_mean` / `mopd_bias_correction_mean` / `teacher_topk_mass_mean` / `student_on_teacher_topk_mass_mean`
- `generalized_jsd` 分支（:1153-1155）只输出 `raw_jsd_token_mean`，无 `mopd_*`
- 旧 v2 运行 TB 只有 `raw_jsd_token_mean` → 确认走错分支；v3 补传后 TB 出现 4 个 `mopd_*` → 确认走对

**退化机制**（为什么 generalized_jsd 在 summary 场景退化，而论文 VQA 不退化）：
- `generalized_jsd` = forward JSD + **student 选 top-k support** + tail bucket。student 退化时自己选的 support 也跟着偏，support 集自适应退化 → 正反馈
- `mopd`（teacher 选 support）= teacher 固定选 support，student 退化改不了 support → 抗退化。但 EMA teacher 紧跟 student + 开放式摘要无客观锚点 → mopd 也只是**推迟**退化（step4→step27），非根治
- 论文 VQA 任务有隐式锚点（MCQ 正确答案语义），且任务短答；summary 是开放式长生成，drift 空间大

**修复**：在 `run_rp_opsd_v2.sh` 顶部加 3 变量（:81-83）+ 命令行 3 override（:147-149），详见 §5.4.1。v3（`flashnote_train_v4`）从 0 开始训练，step18 mean 266 稳定（未涨），mopd 生效。

**附坑：idle_protect cron 抢 GPU 致 vllm init failed**：
- 现象：v3 启动报 `ValueError: Free memory on device cuda:0 (31.71/95.07 GiB) < desired GPU memory utilization (0.7, 66.55 GiB)` → `RuntimeError: Engine core initialization failed`
- 根因：v3 崩溃后 GPU 空，`idle_protect.sh` cron（每 20min）检测到空闲 → 回填 gemma filler 占满 8 卡（~63GB/卡）→ v3 vllm 抢不到显存
- 修复：启动训练前先 `tmux kill-session -t idleprotect_g{0..7}` + kill filler 进程，**立刻**启动训练（cron 回填窗口 ~20min，vllm init ~2-3min 占满 GPU 后 idle_protect 不再回填）。详见记忆 `feedback_idleprotect_check_before_launch`

### 5.5 合并 FSDP 权重

训练产 FSDP 分片，推理前需合并：

```bash
./run.sh merge --output-dir outputs/flashnote_rp_opsd
# 产 HF 格式权重到 outputs/flashnote_rp_opsd/checkpoints/global_step_N/merged
```

---

## 6. 评测方案

### 6.1 评测口径

所有模型均用**原始分辨率图像**评测（高分辨率推理，验证超调增益）。

- **任务**：summary，4 语种 en / fr / ru / zh
- **方式**：LLM-as-judge MOS 打分（gemini-3-flash-preview），复用 flash_note 现有评测
- **脚本**：`/data4/wumeimei/flash_note/auto_eval/evaluators/run_qwen38_27b_recipe_eval.sh`
- **测试集**：`/data4/wumeimei/flash_note/test_data/{en,fr,ru,zh}_image/` + excel

### 6.2 对比基线

| 模型 | 说明 |
|------|------|
| Qwen3.5-9B Base | 无训练基线 |
| RP-OPSD（本方案）| 自蒸馏后，原图评测 |
| SFT 对照 | 半分辨率图 + gold summary 监督，2 epoch，原图评测（见 §11）|

> 仓库自带 9 项视觉 Benchmark 评测（`run.sh eval`，用 Qwen3.5-9B 当 judge），但那是 Dataset4.0 评测，与 flash_note summary 无关。flash_note summary 评测用本节 MOS 口径。

### 6.3 流程

```bash
# 1) 合并 LoRA... 不，是合并 FSDP 权重（§5.5）
# 2) 推理（原始分辨率图，4 语种 summary），产物入 infer_res_rp_opsd/
# 3) 4 语种 MOS 评测
INFER_DIR=/data4/wumeimei/flash_note/infer/infer_res_rp_opsd \
LABEL=rp_opsd_summary_9b \
bash /data4/wumeimei/flash_note/auto_eval/evaluators/run_qwen38_27b_recipe_eval.sh
```

### 6.4 预期

论文主实验：9B 自蒸馏 9 项 Benchmark 平均 76.27 → 80.43（+4.16 / +5.45%）。本方案预期 RP-OPSD 在 4 语种 summary MOS 上**不低于 Base**（分辨率特权超调增益）。重点关注含表格/小字/图表的视觉细节密集样本是否因高分辨率超调显著提升。

---

## 7. 风险与注意事项

1. **reward-free 适配生成任务**：RP-OPSD 论文数据是 MCQ+OpenQA（有"正确答案"语义但不用），flash_note summary 是开放生成。自蒸馏 KL 只对齐师生分布，不依赖答案正确性，理论可迁移；但生成任务 on-policy 轨迹更长（512–1024 token vs QA 短答），训练更重，需关注 drift（监控 `actor/vopd_loss` 趋势 + `rollout_corr/kl` + `self_distillation/num_distill_tokens`，指标定义见 §5.4.5）。
2. **无 freeze_vit**：RP-OPSD 视觉塔参与训练（全参数），与 flash_note OPSD 的 `freeze_vit=true` 不同。全参数 + 8 卡 H20 显存需确认（仓库用 FSDP param/optimizer offload 兜底）。
3. **训练时长**：96K 数据 1 epoch ≈ 6 天，建议先子集 55 步验证机制再全量（§4.1）。
4. **数据转换正确性**：`prompt` 的 content 必须含与 `images` 数量匹配的 `<image>` 占位符（每条 1 图 1 占位符），否则 `_build_messages` 的 `assert image_offset == len(images)` 崩。
5. **teacher_image_key 覆盖**：`run_rp_opsd.sh` 默认 `bbox_images`，flash_note 数据列名是 `teacher_images`，必须在命令行覆盖（§5.4）。
6. **模型 ID**：`run_rp_opsd.sh` 默认 `Qwen/Qwen3.5-4B`，需覆盖为本地 9B 绝对路径。
7. **环境内核**：`flash-attn` + `causal-conv1d` 必装；若复用 swift 环境内核需版本对齐（仓库固定 PyTorch 2.10 / vLLM 0.18，与 swift 环境可能不一致，建议新建 venv）。

---

## 8. 落地步骤速查

```bash
# 0. 准备环境
./run.sh prepare-env
# 1. 转换 summary jsonl → parquet（每门封顶 2.2W，套翻译模板）
python scripts/convert_flashnote_summary.py --out .runtime/flashnote_summary/train.parquet --max-per-lang 22000
# 2. 只对 parquet 用到的图生成半分辨率 LR（物理 1/2，非 pixelation）
python /data4/wumeimei/flash_note/train/gen_lr_images.py --from-parquet .runtime/flashnote_summary/train.parquet
# 3. --check-lr 校验 LR 齐（0 缺失）
# 4. Smoke 1 步验证（诊断 teacher_image_swap=1.0 / vopd_loss>0 / num_distill_tokens>0，见 §5.4.5）
# 5. 55 步子集验证机制
# 6. 合并 FSDP → 推理 → 4 语种 summary MOS 评测 vs Base
```

---

## 9. 路径索引

| 项 | 路径 |
|----|------|
| 本方案 | `RP-OPSD/docs/flash_note_RP_OPSD.md` |
| 仓库 README | `RP-OPSD/README_zh.md` |
| 跨模型实验（结构参照）| `RP-OPSD/docs/cross_model_distillation_zh.md` |
| 训练入口 | `RP-OPSD/run.sh` → `scripts/train.sh` |
| 底层训练脚本 | `RP-OPSD/scripts/run_rp_opsd.sh` |
| 训练配置 | `RP-OPSD/config/best.yaml` / `config/best.env` |
| verl 数据加载 | `RP-OPSD/verl/utils/dataset/rl_dataset.py`（`_build_messages`）|
| chat template | `RP-OPSD/chat_templates/perception_chat_template_qwen35.jinja` |
| 数据转换 | `RP-OPSD/scripts/convert_flashnote_summary.py` |
| 降质图生成 | `/data4/wumeimei/flash_note/train/gen_lr_images.py` |
| summary parquet（产物）| `RP-OPSD/.runtime/flashnote_summary/train.parquet` |
| summary 源数据 | `/data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl` |
| 半分辨率图(student) | `/data4/wumeimei/flash_note/train/<lang>_image_lr/` |
| 原图目录(teacher) | `/data4/wumeimei/flash_note/train/<lang>_image/` |
| 本地模型 | `/data4/wumeimei/download_models/Qwen3.5-9B` |
| 评测脚本 | `/data4/wumeimei/flash_note/auto_eval/evaluators/run_qwen38_27b_recipe_eval.sh` |
| 测试集 | `/data4/wumeimei/flash_note/test_data/{en,fr,ru,zh}_image/` |
| tensorboard 日志 | `RP-OPSD/tensorboard_log/RP-OPSD/<experiment_name>/`（仓库根，非 output_dir） |
| loss 分析脚本 | `RP-OPSD/scripts/analyze_loss.py`（文本log分段趋势）、`scripts/detail_metrics.py`（逐step+尖峰）、`scripts/analyze_tfevents.py`（tfevents解析，文本log不在本机时用）|

---

## 10. 扩展：跨模型变体

如需复现 [跨模型实验](cross_model_distillation_zh.md)（固定 9B Teacher → 4B Student，替代 EMA 自蒸馏），改 `run_rp_opsd.sh`：

```bash
actor_rollout_ref.model.path=/data4/wumeimei/download_models/Qwen3.5-4B   # Student 4B
actor_rollout_ref.actor.self_distillation.teacher_model_source=legacy     # 固定外部 teacher
actor_rollout_ref.actor.self_distillation.teacher_regularization=none     # 关 EMA
# teacher 用固定 9B 权重（需额外部署/加载 9B 作为 teacher，非自蒸馏）
```

> 跨模型变体不再是自蒸馏，需加载两个模型（4B student + 9B fixed teacher），显存与实现复杂度上升。建议先跑通 §1–8 的自蒸馏主实验再考虑。

---

## 11. SFT 对照实验（隔离训练范式）

### 11.1 目的

控制变量，隔离"分辨率特权自蒸馏"本身的增益。除训练范式外其余全部对齐：

- 同模型：Qwen3.5-9B
- 同数据：72k 采样（seed42，每门 2.2W，与 RP-OPSD **同一批**数据，见 §11.3）
- 同训练视图：SFT 用**原图**训练（与评测同口径）；RP-OPSD student 用半分辨率
- 同评测：原始分辨率图，4 语种 summary MOS
- 同 epoch：2

唯一变量 = 训练范式：
- **RP-OPSD**：reward-free 自蒸馏（on-policy 轨迹 + Teacher Top-100 reverse KL）
- **SFT**：gold summary 监督（reward-dependent，标准 teacher forcing）

### 11.2 对照矩阵

| 实验 | 训练图 | 监督信号 | 训练范式 | 评测图 | 角色 |
|------|--------|----------|----------|--------|------|
| Base | — | — | — | 原图 | 无训练基线 |
| RP-OPSD | 半分辨率 | 无（reward-free 自蒸馏）| on-policy distill | 原图 | 本方案 |
| **SFT-HR** | **原图** | gold summary | SFT | 原图 | **主对照**（隔离范式）|
| SFT-LR（可选）| 半分辨率 | gold summary | SFT | 原图 | 与 RP-OPSD 同视图的对照 |

> SFT-HR 用原图训练（与评测同口径），作为主对照隔离训练范式 → 直接回答"分辨率特权自蒸馏是否优于 gold 监督 SFT"。
> SFT-LR 与 RP-OPSD 唯一差训练范式、训练图也同为半分辨率，是最严格同视图对照，可选。

### 11.3 数据（与 RP-OPSD 同批）

`scripts/convert_flashnote_sft.py`：从源 jsonl 取 gold summary 作 SFT 监督。

- **gold summary 自探测**：优先 `messages[assistant].content` → 顶层 `summary/response/answer/gold/target/output` → `teacher_prompt`（str 或 dict 的 `summary/content/...`）。运行时打印命中样例，便于核对字段。
- **4 语种 prompt 模板**：复用 `convert_flashnote_summary.py` 的 `PROMPT_TEMPLATE_<lang>`，与 RP-OPSD 同 prompt。
- **采样一致性**：seed42、每门封顶 2.2W、同遍历顺序、`rng.sample` 同位置选择 → 与 RP-OPSD parquet 用的是**同一批 72k 数据**（只差监督信号：SFT 有 gold，RP-OPSD 无）。
- **训练图**：SFT 默认**原图**训练（与评测同口径，主对照用此）；`--lr` 可切半分辨率做 SFT-LR 同视图对照。
- 输出 swift SFT jsonl：`messages=[{user:<image>+语种模板},{assistant:gold}]`、`images=[图]`。

```bash
# SFT-HR 主对照（原图训练，与评测同口径）
python scripts/convert_flashnote_sft.py --out .runtime/flashnote_summary/sft_train_hr.jsonl --hr --show-sample
# SFT-LR 同视图对照（半分辨率，与 RP-OPSD student 同视图）
python scripts/convert_flashnote_sft.py --out .runtime/flashnote_summary/sft_train.jsonl
```

### 11.4 数据格式规范与完整样例（必读）

> ⚠️ **血泪教训（2026-08-30 触发）**：首批 `sft_train.jsonl` 全 72,166 条 assistant response 都把 `teacher_prompt`（"Role Setting & Task Description" 模板）拼在真摘要前面当 label，模型被训练成"复读 prompt 模板 + 真摘要"，推理时浪费 2/3 token 且 train/inference prompt 不一致。根因是 `convert_flashnote_sft.py` 的 gold summary 探测命中 `teacher_prompt` 字段时，把整段 teacher_prompt（含 prompt 模板）当成 gold 写进了 assistant。
>
> 修复：`sft_train_clean.jsonl` 已清洗（split `【Reference Summary】` 取后半），重训用此文件。详见 `.runtime/flashnote_summary/DATA_FORMAT_SPEC.md`。

#### 11.4.1 格式规范

每条 SFT 样本为单行 JSON，结构如下：

| 字段 | 类型 | 内容要求 |
|------|------|----------|
| `messages` | `list[dict]` | 长度=2，`[0]=user`（任务 prompt）、`[1]=assistant`（**仅纯摘要**）|
| `messages[0].role` | str | `"user"` |
| `messages[0].content` | str | `<image>\n` + 语种 prompt 模板（Core Identity 风格任务定义）|
| `messages[1].role` | str | `"assistant"` |
| `messages[1].content` | str | **仅目标摘要**，不含 `<image>`/prompt 模板/`【Reference Summary】` 等任何前缀 |
| `images` | `list[str]` | 长度=1，**原图**绝对路径（SFT 用原图训练，与评测同口径；非 RP-OPSD 的半分辨率）|

**红线（训练前必跑复核脚本 assert）**：
1. `assistant.content` 不以 `<image>` 开头
2. `assistant.content` 前 200 字符不含 `Role Setting` / `Core Identity` / `Task Description` 等 prompt 模板关键词
3. `assistant.content` 不含 `【Reference Summary】` 生成标记
4. `user.content` 和 `assistant.content` 不能共用同一套 prompt 文本（user 是"任务要求"，assistant 是"答案"）
5. 推理脚本 user prompt 必须和训练 user prompt 一致（都用 Core Identity 同一版）

#### 11.4.2 完整样例 1（English）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nCore Identity\nYou are a professional image-text comprehension and summary expert. You specialize in extracting valid key information from images and generating accurate, concise, well-structured English summaries.\nMandatory Rules (Highest Priority)\nAll outputs must strictly comply with the following hard constraints. Any violation makes the output invalid:\n1. Language Rule: Output only English, no other languages are allowed.\n2. Authenticity Rule: Only use recognizable content from the image. No subjective speculation, no external knowledge, no fabricated information.\n3. Length Rule: The total summary length must not exceed 500 English words.\n4. Quality Rule: Extract core information only. Do not copy original content mechanically, and avoid redundant or empty descriptions.\nTask & Output Standards\nGenerate a standard English summary based on the visible content of the image. Adapt your output structure according to the information density of the image:\nMode 1: Information-Dense Images\nApplicable to images containing multiple key points, data, steps, opinions or rules. Use structured Markdown format:\n1. Opening: 1–2 concise sentences to summarize the core theme of the image (required).\n2. Main content: List key information in ordered or unordered lists.\n3. List specification: Each list item must start with a short bold subheading, followed by a concise and objective explanation based strictly on the image content.\nMode 2: Simple & Low-Density Images\nApplicable to images with a single topic or simple content. Use plain paragraph natural language without redundant Markdown formatting, and accurately output the only core information.\nSupplementary Rules\n1. If the image contains extremely limited information, output the most accurate and concise summary available.\n2. All Markdown symbols are only for layout and do not count towards the word limit.\n3. Keep all content logical, objective and refined."
    },
    {
      "role": "assistant",
      "content": "A person wearing a hijab with a purple flower adornment stands on a sandy beach. Behind them are large trees, including coconut palms, with other individuals and a boat visible in the background under an overcast sky."
    }
  ],
  "images": [
    "/data4/wumeimei/flash_note/train/en_image/b7f39493c2148c7a32c4226c57cc9231187d7c530bb8a42d098295b7f79d138c.jpg"
  ]
}
```

#### 11.4.3 完整样例 2（中文）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\n核心身份\n你是专业的图文理解与摘要撰写专家，擅长从图片中提取有效关键信息，生成精准、简洁、结构清晰的中文摘要。\n强制规则（最高优先级）\n所有输出必须严格遵守以下硬性要求，违反任意一条即视为输出无效：\n1. 语言规则：仅输出中文内容，禁止出现其他语种文字。\n2. 内容真实规则：仅使用图片中可识别的内容，禁止主观推测、引用外部知识、编造虚假信息。\n3. 字数规则：摘要总字数不得超过500个汉字。\n4. 质量规则：仅提炼核心有效信息，禁止机械照搬原图内容，杜绝冗余、空洞描述。\n任务与输出规范\n根据图片可视内容生成标准摘要，依据图片信息密度自适应匹配输出结构：\n模式一：高信息密度图片\n适用于包含大量要点、数据、步骤、观点、规则的图片，采用Markdown结构化格式输出：\n1. 开篇：用1-2句精简语句概括图片核心主题（必填）。\n2. 主体内容：以有序或无序列表展示关键信息。\n3. 列表规范：每条内容必须以简短加粗小标题开头，后跟严格贴合原图、客观简洁的文字说明。\n模式二：简单低信息密度图片\n适用于单一主题、内容简单的图片，采用纯段落自然语言输出，无需多余Markdown格式，精准输出唯一核心信息即可。\n补充规则\n1. 若图片有效信息极少，输出最精准、最简洁的核心摘要即可。\n2. Markdown符号仅用于排版，不计入字数统计。\n3. 输出内容全程逻辑清晰、客观中立、精炼无冗余。"
    },
    {
      "role": "assistant",
      "content": "成功不仅仅是靠运气，更取决于你所做的选择和养成的习惯。"
    }
  ],
  "images": [
    "/data4/wumeimei/flash_note/train/zh_image/321a73feb0b6917de96e148bd9b279826431d2e5e1bf2c0d843e1f52b90f5df0.jpg"
  ]
}
```

> 样例要点：`user.content` 以 `<image>\n` 开头接任务 prompt；`assistant.content` **只有纯摘要**，无任何 prompt 回显、无 `<image>`、无 `【Reference Summary】` 标记。训练前用 `.runtime/flashnote_summary/DATA_FORMAT_SPEC.md` 里的复核脚本全量 assert。

### 11.5 训练

`scripts/run_flashnote_sft.sh`：swift 全参数 SFT，2 epoch，每 150 步存一次，H20 8 卡，DeepSpeed zero2。

- effective batch = per_device(2) × grad_accum(6) × 8 卡 = 96（与 RP-OPSD batch96 对齐）
- max_length 9216 = 8192 + 1024（与 RP-OPSD 同）
- lr 1e-5（SFT 全参数惯例；RP-OPSD 用 2e-6 是 RL 口径，范式不同，lr 不强行对齐）
- 环境：swift env（SFT 不依赖 flash_attn/causal_conv1d/vllm rollout）

```bash
bash scripts/run_flashnote_sft.sh
```

> 框架差异说明：RP-OPSD 用仓库 verl（FSDP），SFT 对照用 swift（DeepSpeed zero2）。两者均为全参数标准实现，框架差异是本对照的已知混淆因素，但非主导；若需进一步消歧，可用 verl 的 SFT trainer 同框架跑（待仓库 SFT 支持确认）。

### 11.6 评测

同 §6：合并权重 → 原图 4 语种 summary 推理 → MOS 评测，与 Base / RP-OPSD 同口径对比。

---

## 12. 方案版本矩阵（v2 / v3 / v3-no-ema / v3-SFT / v4 / v5）

变体共享 mopd + 分辨率特权 + 5k 口径 + alpha1.0，只换 student 起点 或 teacher 更新策略。目的是隔离退化因素（EMA 正反馈 vs student 起点质量 vs teacher 模型能力）。

| 版本 | Student 起点 | Teacher | 目的 | 运行机器 | 状态 |
|---|---|---|---|---|---|
| **v2**（base） | base Qwen3.5-9B | EMA ρ=0.05 | 基线复现 | m4 | ❌ 退化（漏传 mopd，详见 §5.4.6） |
| **v3**（base+mopd） | base Qwen3.5-9B | EMA ρ=0.05 | mopd 修复基线 | m4 | ⚠️ step451 仍在跑但已退化（mixed% 78.6%@step441，见 §5.4.5.1 + rollout 分析）|
| **v3-no-ema**（关 EMA） | base Qwen3.5-9B | **固定不更新**（rate=0，ema 模式但冻结，`teacher_model_source=legacy`） | 隔离 EMA 正反馈因素，与 v3 base 同起点对照 | **m3** | ✅ 已启动（2026-09-02，`scripts/run_rp_opsd_v3_no_ema.sh`），step221 健康 training_ppl=18 ✅（见 §12.3.1） |
| **v3-SFT**（SFT warm-start） | sft_gold_397b_lora_r64_m2 merged/step_2250（2.5epoch, MOS4.064）| EMA ρ=0.05 | SFT 拉齐基础再自蒸馏精修，起点更高防 drift | m3 | ⚠️ 中度退化（step459 实测，退化比 v3 更重，推翻预期，见 §12.1.1），2026-09-02 已停换成 v3-no-ema |
| **v4**（SFT warm-start + 固定 teacher） | sft_gold_397b_lora_r64_m2 merged/step_2250（2.5epoch, MOS4.064）| **固定不更新**（rate=0，`teacher_model_source=legacy`） | 组合 SFT 起点优势 + 消除 EMA 正反馈退化，根治方案 | m4（原计划）| ❌ 已停（2026-09-03 step214 时停掉，SFT 起点尖锐分布 + 固定 9B teacher 错配：training_ppl 从 step1 就 3.6e4，vopd_loss 0.13 收敛慢，换成 v5） |
| **v5**（9B student + **27B 外部 teacher**） | base Qwen3.5-9B | **外部 27B 固定 teacher**（Qwen3.8-27B，`teacher_model_source=fixed` + `teacher_model_path`，rate=0） | 用更强 27B teacher 提供更高质量蒸馏信号 + 仍固定不退化 | **m4** | ✅ 已启动（2026-09-03，`scripts/run_rp_opsd_v5_teacher27B.sh`），step1 健康 training_ppl=6.54 ✅（见 §12.4） |

> **v3-no-ema vs v4 vs v5**：三者都关 EMA（`teacher_regularization=ema` + `teacher_update_rate=0.0`，框架不支持 `none`，见 `verl/workers/config/actor.py:130`），但 student 起点和 teacher 来源不同：
> - **v3-no-ema**（m3）：base 9B student + legacy 固定 teacher（=student 初始副本）。自蒸馏对照，隔离 EMA 因素。
> - **v4**（m4，已停）：SFT warm-start student + legacy 固定 teacher（=student 初始 SFT 副本）。SFT 起点尖锐分布与固定 teacher 错配，step1 ppl=3.6e4。
> - **v5**（m4）：base 9B student + **外部 27B 固定 teacher**（`source=fixed` + `teacher_model_path=Qwen3.8-27B`）。27B 比 9B teacher 更强，logit 分布更平滑，student 容易学（step1 ppl=6.54，比 v3-no-ema 的 4.52 略高但合理）。

### 12.1 v3-SFT（SFT warm-start student）

**思路**：SFT 先用 397B 教师 gold 数据拉齐基础摘要能力（MOS 4.064），RP-OPSD 自蒸馏在此基础上用分辨率特权精修——student 已会摘要，自蒸馏只补"从 LR 也能产出 HR 质量"的视觉细节能力，起点更高、drift 空间更小。

**student 权重**：`/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft_gold_397b_lora_r64_m2/merged/step_2250/`（LoRA r64 2.5epoch merge 后的全参数 safetensors + config.json，可直接加载）。选 2.5epoch 因 eval report §3.1 确认 LoRA epoch2.5 = MOS 4.064（与全参 epoch1.0 打平，性价比最优）。

**配置改动**（相对 v3 base）：仅改 `run_rp_opsd_v2.sh` 顶部一行
```bash
MODEL_PATH="/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft_gold_397b_lora_r64_m2/merged/step_2250"
OUTPUT_DIR="/data3/wumeimei/flash_note/flashnote_train_v3sft"   # /data3（1.2T 可用，/data4 装不下）
```
teacher 仍 EMA ρ=0.05（teacher 从 SFT ckpt 初始化，慢速跟随 student）。mopd/分辨率特权/5k 口径/alpha1.0 全不变。

**注意**：SFT warm-start 提高起点但**不根治退化**——EMA teacher 仍无外部锚点。~~预期退化起始推迟、幅度减小~~（⚠️ 2026-09-02 实测**推翻**此预期，v3-SFT 退化反而比 v3 更早更重，见 §12.1.1）。

#### 12.1.1 实测健康分析（2026-09-02，step459）

v3-SFT 运行至 step459/1502（30%，仍在 m3 跑）。用 `scripts/analyze_tfevents.py` 解析 tfevents（文本 log 在 m3 本地，tfevents 经 sshfs cwd 落 m4 仓库根 `tensorboard_log/RP-OPSD/RP-OPSD-Qwen3.5-9B-v3sft/`）。

**核心发现：⚠️ 中度退化，形态与 v3 相反——"目标对齐但行为漂移"**

蒸馏目标指标全绿（teacher top-100 support 上对齐），但生成长度/分布失控（top-100 之外长尾漂移）——mopd 蒸馏被"钻空子"。

| 指标类别 | 蒸馏目标（健康↓） | 生成行为（失控↑） |
|---|---|---|
| vopd_loss | 0.214→0.009（-90%）✅ | — |
| kl | 0.179→0.014（-81%）✅ | — |
| chi2_token | 1.24→0.035（-86%，**无末段尖峰**）✅ | — |
| response_length/mean | — | 230→501（+125%）⚠️ |
| clip_ratio | — | 0.01→0.092（逼近 0.1 早停线）⚠️ |
| rollout_ppl | — | 4.9→117（指数涨）⚠️ |
| teacher_topk_mass | 0.997→0.857（teacher 分布变平） | — |

逐 step 拐点：resp_len 无平台期单调上涨，step 241（441）→ 271（500）加速，之后在 476–517 高位震荡（未继续冲高亦未回落）。clip_ratio 从 step 241 破 0.06、271 破 0.09 并持续高位。ro_ppl 从 step 241（10.9）起指数加速。

**vs v3 对比（同进度 ~30%）**

| 指标 | v3 (base+mopd) | v3-SFT (SFT warm-start) |
|---|---|---|
| vopd 末5均 | 0.0099 | 0.0079（更优） |
| kl 末5均 | 0.064 | 0.014（更优） |
| chi2 末5均 | 29.7（尖峰⚠️） | 0.052（干净✅） |
| **resp_len 末5均** | 383（+26%） | **501（+125%）⚠️** |
| **clip_ratio 末5均** | 0.022 | **0.092⚠️ 逼近0.1** |
| ro_ppl 末5均 | 162 | 102（但起点 4.9 → 涨 2444%） |
| grad_norm max | 25.3 | 171.3（均在 warmup 75 步内，非退化信号） |

**⚠️ 推翻 §12.1 预期的原因**

原预期"SFT warm-start 推迟退化起始、减小幅度"，实测相反。根因：
1. SFT student 起点高、生成更确定/更长，on-policy rollout 8 条轨迹偏长且相似 → 长度正反馈更强
2. SFT 学到的摘要长度先验在 LR 视图下被打破，student 用更长生成补偿低分辨率信息缺失
3. EMA teacher 从 SFT 初始化，师生初始几乎相同 → 早期梯度信号弱，长度漂移正反馈持续积累
4. teacher 跟 student 变平（tkmass 0.997→0.857），无法约束长尾

**退化形态对比**：v2（漏传 mopd）= 长度冲到 737 持续暴涨；v3-SFT（mopd 正确）= 长度到 500 后高位震荡。mopd 的 teacher 选 support 起了部分约束（未像 v2 发散），但未根治。

**结论与建议**：
- v3-SFT **考虑早停**（resp_len 破 500 + clip 逼近 0.1 + ro_ppl 指数涨，退化趋势明确）
- 蒸馏目标的"健康"是**虚假的**——top-100 对齐好但长尾失控，监控不能只看 vopd/kl，必看 `response_length/mean` + `clip_ratio`
- 根因仍是 EMA 无外部锚点 + 开放式摘要无长度约束，SFT 起点不解决反加剧长度漂移
- 进一步支持 v4 固定 teacher 路线

### 12.2 v4（SFT warm-start + 固定 teacher，机器4 m4）

**思路**：v3-SFT 已证明 SFT 起点 + EMA 仍退化（§12.1.1），根因是 EMA teacher 无外部锚点。v4 组合两条修复：SFT warm-start student（起点高）+ 固定 teacher 不随 student 更新（消除正反馈），是同时拿到 SFT 优势 + 根治退化的方案。cross_model_distillation_zh.md 已验证固定 teacher 不退化（7 benchmark 6 提升）。

**固定 teacher 两种实现**（框架 `actor.py:72-73,183` 支持）：

| 方式 | 配置 | 显存 | 说明 |
|---|---|---|---|
| **A. rate=0**（v4 选用） | `teacher_regularization=ema` + `teacher_update_rate=0.0` | 不额外占（teacher 复用 student 初始副本） | teacher = student 初始 SFT 权重冻结，最小改动 |
| **B. fixed source** | `teacher_model_source=fixed` + `teacher_model_path=<path>` | +1 份 9B 权重 | teacher 用独立权重（如另载 base 9B），显存翻倍 |

v2/v3 base 配置基础上改（`run_rp_opsd_v4.sh` 顶部）：
```bash
MODEL_PATH="/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_sft_gold_397b_lora_r64_m2/merged/step_2250"   # ★ SFT warm-start student
TEACHER_REGULARIZATION="ema"        # 保持 ema 模式（框架不支持 none）
TEACHER_UPDATE_RATE=0.0             # ★ rate=0 → teacher 冻结在 SFT 初始值，等价固定
OUTPUT_DIR="/data3/wumeimei/flash_note/flashnote_train_v4_fixed_teacher"   # /data3（1.2T 可用）
```
命令行 override 对应 `actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0`。其余（mopd/分辨率特权/5k）全同 v2/v3 base。

**机器4 运行注意**（v4 改到 m4 本机跑）：
- m4 是本机，`/data4/wumeimei/flash_note/RP-OPSD` 直接访问，无 sshfs
- `OUTPUT_DIR=$PROJECT_ROOT/outputs/flashnote_train_v4_fixed_teacher`（m4 本地 `/data4`）
- `TMPDIR=/tmp/rp_opsd_v4` + `MASTER_PORT=29504`（避免与 v3 base 29500、v3-no-ema 29503 撞端口）
- 启动前先 `tmux ls | grep idleprotect` 并 kill 目标卡的保护任务（feedback_kill_idle_protect_before_train）
- 启动前先 `pkill -f run_rp_opsd_v2.sh` 停掉 v3 base 腾出 8 卡

**预期**：固定 teacher 根治 EMA 正反馈，`response_length/mean` 应稳定不涨。teacher = SFT warm-start 初始值（已会摘要，MOS 4.064），比 base 9B teacher 质量高，student 向高质量 teacher 对齐上限更高——优于 v4 原方案（base 9B teacher）。

### 12.3 长训练退化复盘（2026-09-02）

**动机**：v3 base 跑到 step 451 仍持续退化（mixed% 78.6%@step441，见 §5.4.5.1），但作者论文报告 9B 不退化。需定位是参数错还是训练量超作者验证范围。

#### 12.3.1 作者 9B 配置 vs 我们 v3 base 配置

| 参数 | 作者 9B（论文 §训练设置 + 附录 B 补充表 2/3）| 我们 v3 base | 差异 |
|---|---|---|---|
| 训练集 | 5.2K 样本（Vision-SR1 + VLM-CapCurriculum + ZwZ-RL-VQA + Vision-OPD）| 72k（flashnote 4 语种摘要）| 14× |
| epoch | 1 | 2 | 2× |
| **优化步数** | **55** | 1502 | **27×** |
| batch_size | 96 | 96 | ✅ |
| lr | 2e-6 | 2e-6 | ✅ |
| warmup | 10 | 75 | 7.5× |
| max_response_length | 1024 | 2048 | 2× |
| max_prompt_length | 8192 | 3072 | 短（实际只用 1700 token）|
| max_seq_length | 9216 | 5120 | 短 |
| rollout n | 8 | 8 | ✅ |
| EMA ρ | 0.05 | 0.05 | ✅ |
| distillation_topk | 100 | 100 | ✅ |
| distillation_objective | mopd_topk_reverse_kl | mopd_topk_reverse_kl | ✅ |
| topk_source | teacher | teacher | ✅ |
| 9B 训练时间 | 7.83h | ~80h | 10× |

**自蒸馏参数（5 项）作者默认 ≠ 我们**，但我们用的是 MOPD 论文最优组合（表 3 消融），不是错：

| 参数 | 作者默认（actor.py:80-90）| 我们 v3 | 备注 |
|---|---|---|---|
| distillation_objective | generalized_jsd | mopd_topk_reverse_kl | MOPD 论文最优（表 3）|
| distillation_topk | None（全分布）| 100 | MOPD 论文最优 |
| distillation_topk_source | student | teacher | MOPD 要求 |
| distillation_add_tail | True | False | MOPD 要求 |
| alpha | 0.0（forward KL）| 1.0（reverse KL）| MOPD 要求 |

#### 12.3.2 55 步以内 rollout 分析（v3 base + v3-SFT）

直接分析两个运行的前 55 步 rollout（mixed% = 输出含 2+ 语种 token 的比例）：

| step | v3 base mixed% | v3 base avg_len | v3-SFT mixed% | v3-SFT avg_len |
|---|---|---|---|---|
| 1 | 37.6 | 1040 | 32.8 | 854 |
| 11 | 38.0 | 1027 | 29.7 | 833 |
| 21 | 37.9 | 1118 | 31.4 | 823 |
| 31 | 34.8 | 1104 | 29.0 | 824 |
| 41 | 40.1 | 1106 | 34.5 | 862 |
| 51 | 37.2 | 1207 | 31.5 | 946 |

**观察**：
- 前 55 步两个版本 mixed% 都**持平震荡，无上涨趋势**——符合论文报告不退化
- base 9B 起步 mixed=37.6% **本身就高**，这不是退化，是模型在 LR（半分辨率）图像上的初始行为（base 9B 看模糊图倾向跨语种 token 补偿）
- v3-SFT 起步 mixed=32.8%，比 base 9B 低 5 个点——SFT 确实让模型更语种纯净
- avg_len 全程稳定 820-1210，无长度膨胀

#### 12.3.3 退化时间线

| step 区间 | v3 base mixed% | v3-SFT mixed% | 状态 |
|---|---|---|---|
| 1-55（作者验证范围）| 34-40% 持平 | 29-35% 持平 | ✅ 不退化 |
| 55-200（模糊地带）| 缓慢上升 | 缓慢上升 | ⚠️ 漂移累积 |
| 200+ | 46%@221 | — | ❌ 退化起始 |
| 440+ | 78.6%@441 | 50%@459 | ❌ 严重退化 |

#### 12.3.4 关键结论

1. **作者论文没错**——55 步内方法安全，论文报告真实
2. **作者只验证了短训练**——搜遍论文正文 + 附录 B/C + `cross_model_distillation_zh.md`，关键词 `long|extend|更多|更长|多.*epoch|prolong|over.?train|稳定|stabil|收敛|collapse|退化|degrad|drift` 全部 0 命中（除"1 Epoch 共 55 Steps"这一条）。补充表 3 训练计划列写死"1 epoch / 55 步"，检查点列"最终步"——作者设计上就只跑 55 步
3. **我们错在误把"作者验证过的方法"当成"长训练下也安全的方法"**——直接把训练量放大 27×（72k × 2epoch = 1502 步）没做渐进验证
4. **退化起始点在 step 200+**，完全在作者验证范围外
5. **EMA 在长训练下必然退化**——作者表 4 消融"EMA vs fixed teacher"在 55 步内做，EMA 略优（82.73 vs 82.22）；长训练下 EMA teacher 持续累积漂移，fixed teacher 是合理外推

#### 12.3.5 修复路径

**v3-no-ema（m3）+ v4（m4）**：都用固定 teacher（`teacher_update_rate=0.0`）消除 EMA 长训练正反馈。其他参数全同 v3 base（包括 topk=100、max_resp=2048、warmup=75、2 epoch）——**只改 teacher 更新策略一个变量**，隔离 EMA 因素。

**不改的参数**（作者消融已验证最优）：
- `distillation_topk=100`（表 3 消融最优，不要扩到 500）
- `mopd_topk_reverse_kl + topk_source=teacher + add_tail=False + alpha=1.0`（MOPD 论文最优组合）
- `batch 96, lr 2e-6, rollout n=8`（对齐作者）

**保留的变量**（不对齐作者，因我们任务不同）：
- `max_response_length=2048`（摘要需要更长输出，作者 max_resp=1024 是单语种 perception 任务）
- `max_prompt_length=3072`（实测 prompt ~1700 token，3072 是 1.8× 余量）
- `total_epochs=2`（72k 数据 1 epoch = 750 步，仍远超作者 55 步，但 2 epoch 是任务需要）
- `lr_warmup_steps=75`（1502 步 × 0.7% = 10.5 步对齐作者比例，但 75 步更稳）

#### 12.3.6 监控补强

之前 §5.4.5.1 把 v3 step428 判为"健康"是错的——只看 vopd/kl/chi2 三个 top-100 support 指标。语言混杂是**长尾失控**的具体表现，监控必加：

- `mixed_language_rate`（en/zh/other 语种检测）— 用 Unicode 范围扫描
- `response_length/mean` + `clip_ratio`（已有）
- `rollout_ppl`（已有，student 在自己策略下 token 概率，>100 就是漂移信号）

后续 v3-no-ema / v4 启动后必须监控这 4 个指标，不能只看 vopd/kl。

### 12.4 v5（base 9B student + 外部 27B 固定 teacher，机器4 m4）

**思路**：v4（SFT warm-start + legacy 固定 9B teacher）已证明"SFT 起点尖锐分布 + 固定同尺寸 teacher"是错配——step1 training_ppl 就 3.6e4，student 演化后偏离 teacher top-k 越来越远（见 §12.4.4 v4 实测）。v5 换思路：用**更强 27B 外部模型当固定 teacher**，提供更高质量蒸馏信号 + logit 分布更平滑（27B 比 9B 不会过度尖锐），student 容易学且不退化。

**配置**（`scripts/run_rp_opsd_v5_teacher27B.sh`，相对 v3 base 只改 4 行）：
```bash
MODEL_PATH="/data4/wumeimei/download_models/Qwen3.5-9B"             # student 回到 base 9B（同 v3，未训练）
TEACHER_MODEL_SOURCE="fixed"                                          # 从 legacy 改成 fixed（外部 teacher）
TEACHER_MODEL_PATH="/data4/wumeimei/download_models/Qwen3.8-27B"     # 外部 27B teacher 路径
TEACHER_UPDATE_RATE=0.0                                               # 固定不更新（rate=0）
# 其余 mopd / 5k 口径 / batch / lr / warmup 全同 v3 base
# OUTPUT_DIR=outputs/flashnote_train_v5_teacher27B
# TMPDIR=/dev/shm/rp_opsd_v5 (tmpfs 965G)
# MASTER_PORT=29505
```

**框架支持**（`verl/workers/config/actor.py:183`）：`teacher_model_source` 三选一 `legacy/current/fixed`，`fixed` 时需配 `teacher_model_path`，加载逻辑在 `verl/workers/fsdp_workers.py:904-919`（用 `ref.fsdp_config` 做 FSDP 加载，和 student/ref 共存但参数独立）。

**架构兼容性**：Qwen3.8-27B 和 Qwen3.5-9B 都是 `Qwen3_5ForConditionalGeneration` 架构、`model_type: qwen3_5`、`image_token_id: 248056`、`eos_token_id: 248044`，vocab 一致，reverse-KL top-k 可算（hidden_size 5120 vs 4096 不影响 logits 维度）。

#### 12.4.1 启动 + step1 实测（2026-09-03）

启动 2026-09-03 09:32，m4 本机 8 卡 H20。GPU 显存峰值 75GB/卡（27B teacher FSDP + 9B student FSDP + vllm 0.7 util + ref 9B + activations），余 22GB，安全。

**step1 指标**（对比 v3-no-ema / v4）：

| 指标 | v5 step1 | v3-no-ema step1 | v4 step1 | 判断 |
|---|---|---|---|---|
| training_ppl | **6.54** | 4.52 | 3.6e4 ⚠️ | v5 健康 ✅（27B teacher top-k 在 9B student 上 ppl 合理，比 v4 错配好 5500 倍）|
| rollout_ppl | 4.99 | 3.56 | 3.53 | 正常 |
| kl | 0.193 | 0.189 | 0.19 | 正常 |
| vopd_loss | 0.495 | 0.122 | 0.210 | v5 偏高但合理（27B vs 9B 差异比 9B vs 9B 大），预期会降 |
| rollout_is_mean | 1.0 | 1.0 | 1.0 | 完全 on-policy ✅ |
| 首步用时 | ~8 min | ~5 min | ~5 min | 27B teacher forward 慢，1502 步预计 ~200h（8 天）|

**判断**：v5 step1 健康，27B teacher 完全规避了 v4 的"SFT 起点尖锐分布 + 固定 9B teacher 错配"问题。

#### 12.4.2 v4 实测复盘（已停，2026-09-03）

v4 从 2026-09-02 22:57 启动跑到 2026-09-03 09:30（约 10.5 小时），step 214/1502。**没在发散**（vopd_loss 0.21→0.13 缓慢收敛、kl 0.19→0.11 下降、rollout_ppl 3-10 正常），但 **SFT warm-start + 固定 9B teacher 是结构性错配**：

- step1 training_ppl=3.6e4（vs v3-no-ema 4.5，差 8000 倍）——SFT 起点分布尖锐（某些 token 概率极大），第一次 update 后 student 在 teacher top-k 上概率暴跌
- teacher rate=0 永不更新 = teacher top-k 永远是 SFT 起点的尖锐分布，student 越演化偏离越大
- training_ppl 长期在 1e3~1e5 量级剧烈波动，vopd_loss 绝对值始终比 v3-no-ema 高一倍（0.13 vs 0.06），收敛慢

**结论**：固定 teacher 消除 EMA 正反馈退化的设计意图正确（v3-no-ema 已验证），但 v4 用 legacy source（=student 初始副本）+ SFT warm-start 是错配——SFT 起点越尖锐越不适合当固定 teacher。v5 改用外部 27B teacher 完全规避此问题。

#### 12.4.3 v5 设计预期

1. **27B teacher logit 分布更平滑**：比 9B teacher 不会过度尖锐，student 容易学（step1 ppl=6.54 已验证）
2. **固定不更新 = 无 EMA 正反馈退化**：同 v3-no-ema 的"固定 teacher 消除退化"机制
3. **更强蒸馏信号**：27B 本身比 9B 强，top-k 选出的 token 更高质量，reverse-KL 推 student 学到更好的分布
4. **预期 vopd_loss 收敛目标**：v3-no-ema 到 step221 已 0.06，v5 初始 0.495 偏高但应逐步降到 0.1 以下
5. **风险**：27B teacher forward 慢（1502 步 ~8 天），如时间不够可考虑 500 step 早停评测

### 12.5 完整 loss 趋势汇总（v3-no-ema / v3-SFT / v4 / v5，截至 2026-09-03）

> 数据来源：各任务 `train.log` 中 `step:N` 行，提取 `actor/vopd_loss` / `rollout_corr/training_ppl` / `rollout_corr/rollout_ppl` / `rollout_corr/kl` / `rollout_corr/chi2_token` 五个核心指标。
> 关键阅读：
> - `vopd_loss` = reverse-KL top-100 蒸馏损失，**越小越好**，但单独看会被 mode-seeking 蒙蔽
> - `training_ppl` = student 当前 batch 上对 teacher top-k token 的 ppl，**稳定在 10 以内为健康**；若指数级增长 = student 正在跑偏到 teacher support 外
> - `rollout_ppl` = rollout（student 自己生成）token 在 student 自身下的 ppl，**和 training_ppl 应同量级**；ro_ppl 指数级 > tr_ppl = rollout 乱码化
> - `kl` = rollout/student policy 与 ref 的 KL，**稳定低位即健康**；v3-SFT kl 看着低（0.01）但 ro_ppl 指数爆 = 退化信号被 kl 蒙蔽
> - `chi2_token` = token 级 chi2 统计量，**<2 健康**；突发尖峰（如 21.7）= batch 内有极端 outlier，通常下步自愈

#### 12.5.1 v3-no-ema（m3，base 9B student + 固定 9B legacy teacher，rate=0）— ✅ 健康

step 范围：1 → 235（仍在训）。健康判据：tr_ppl 稳定 4~9，ro_ppl 同量级，vopd_loss 0.12 → 0.066 单调收敛。

| step | vopd_loss | training_ppl | rollout_ppl | kl | chi2_token | 备注 |
|------|-----------|--------------|-------------|------|-------------|------|
| 1    | 0.1219 | 4.52  | 3.56  | 0.189 | 2.11 | 起点 |
| 25   | 0.0949 | 13.35 | 7.49  | 0.127 | 1.30 | warmup 期波动 |
| 50   | 0.0800 | 4.52  | 3.85  | 0.124 | 1.04 | 稳定 |
| 100  | 0.0778 | 5.37  | 4.30  | 0.126 | 0.72 | 稳定 |
| 150  | 0.0662 | 5.72  | 4.45  | 0.128 | 1.06 | 稳定 |
| 200  | 0.0671 | 5.28  | 4.44  | 0.121 | 0.69 | 稳定 |
| 225  | 0.0626 | 8.65  | 5.67  | 0.133 | 3.82 | 小尖峰，未崩 |
| 235  | 0.0658 | 5.49  | 4.54  | 0.123 | 0.59 | 已恢复 |

**健康结论**：vopd_loss 0.12→0.066（-46%），tr_ppl 全程 4~9 稳定，ro_ppl 同步稳定，无退化信号。step 225 小尖峰（tr_ppl=8.65, chi2=3.82）下步立即恢复，属正常波动。**作为 v5 的对照基线**。

#### 12.5.2 v3-SFT（m3，SFT warm-start 9B + EMA teacher rate=0.05）— ❌ 退化已停

step 范围：1 → 493（已停）。退化判据：tr_ppl 全程 1e3~1e5 大幅波动，ro_ppl 4.87 → 148.31 单调指数增长 = student rollout 逐步乱码化。

| step | vopd_loss | training_ppl | rollout_ppl | kl | chi2_token | 备注 |
|------|-----------|--------------|-------------|------|-------------|------|
| 1    | 0.2142 | 1.62e4 | 4.87   | 0.179 | 1.24 | SFT 起点尖锐 |
| 50   | 0.1436 | 1.18e4 | 1.84   | 0.081 | 0.83 | vopd 看似在降 |
| 100  | 0.0871 | 711    | 2.21   | 0.043 | 0.27 | vopd 继续降 |
| 200  | 0.0394 | 5.36e4 | 6.68   | 0.021 | 0.06 | vopd 已很低 |
| 300  | 0.0184 | 6.94e3 | 26.86  | 0.013 | 0.26 | ro_ppl 开始涨 |
| 400  | 0.0093 | 9.24e4 | 70.38  | 0.013 | 0.04 | ro_ppl 指数涨 |
| 484  | 0.0064 | 3.97e4 | 111.30 | 0.013 | 0.14 | 接近停训 |
| 493  | 0.0068 | 1.10e5 | 148.31 | 0.013 | 0.07 | 停训点 |

**退化结论**：vopd_loss 从 0.21 降到 0.007（看似收敛 97%）是**假象**——这是 mode-seeking collapse 的典型特征，student 把 teacher top-k 上的概率堆满但忽略了 top-k 外的支撑。真正的退化信号在 `rollout_ppl`：4.87 → 148.31（30 倍），且 `kl` 在 step 200 后稳定 0.013 不再下降 = student 在 rollout 支撑外乱跑。tr_ppl 大幅震荡（711 ~ 1.1e5）也说明 student 分布尖锐不稳定。**EMA 正反馈退化的标准案例**。

#### 12.5.3 v4（m4，SFT warm-start 9B + 固定 legacy 9B teacher rate=0）— ❌ 结构错配已停

step 范围：1 → 216（已停换成 v5）。错配判据：tr_ppl 从 step 1 起就 3.6e4，全程在 3e4 量级震荡，无收敛趋势。

| step | vopd_loss | training_ppl | rollout_ppl | kl | chi2_token | 备注 |
|------|-----------|--------------|-------------|------|-------------|------|
| 1    | 0.2099 | 3.64e4 | 3.53  | 0.191 | 1.47 | SFT 起点尖锐 |
| 25   | 0.1752 | 2.69e4 | 2.48  | 0.151 | 0.74 | warmup 期 |
| 50   | 0.1562 | 3.03e4 | 8.19  | 0.107 | 0.57 | tr_ppl 不降 |
| 100  | 0.1350 | 2.68e5 | 2.38  | 0.108 | 0.51 | tr_ppl 反弹 |
| 150  | 0.1239 | 73.82  | 3.27  | 0.100 | 0.43 | 单步尖峰 |
| 200  | 0.1228 | 3.65e4 | 9.52  | 0.105 | 0.65 | 回到 3e4 |
| 216  | 0.1256 | 2.24e4 | 4.88  | 0.108 | 5.69 | 停训点 |

**错配结论**：vopd_loss 0.21→0.126 只下降 40%（vs v3-no-ema 下降 46%），且 tr_ppl 全程 3e4 量级震荡完全没收敛。根因：**SFT 起点 student 分布过度尖锐 + 固定 9B teacher 能力不足以提供平滑矫正信号** = teacher 在 top-k 选出的 token 对 SFT student 没有信息增益。ro_ppl 稳定 3~10 说明 student 自身没崩，但 tr_ppl 不降说明蒸馏没生效。**这就是 v5 把 teacher 升级到 27B 的动机**。

#### 12.5.4 v5（m4，base 9B student + 外部固定 Qwen3.8-27B teacher rate=0）— ⏳ 初始化中

step 范围：0（wandb 重启后仍在初始化，截至 2026-09-03 10:02）。首批数据待补。

| step | vopd_loss | training_ppl | rollout_ppl | kl | chi2_token | 备注 |
|------|-----------|--------------|-------------|------|-------------|------|
| 1    | 0.4950 | 6.54  | -     | -    | -    | wandb 重启前首次记录 |
| 2+   | 待补   | 待补  | 待补  | 待补 | 待补 | 初始化中 |

**预期观察点**（待首 50 step 数据出来后填回）：
- step 1：vopd=0.495 偏高（27B teacher 与 9B student 分布差距大于 9B legacy），tr_ppl=6.54 与 v3-no-ema 同量级（4.52），无 SFT 尖锐问题
- step 25-50：vopd 应快速降到 0.1 量级（27B 信号强）
- step 100+：vopd 应稳定 0.05~0.08，tr_ppl 应稳定 4~10
- 若 step 200 ro_ppl 开始指数涨 = v5 也退化（不应该，因为 teacher 固定且更强）
- 若 step 500 tr_ppl 仍稳定 = v5 成功，可作最终方案候选

#### 12.5.5 四任务横向对比（step 1 / step 50 / 末步）

| 任务 | step1 tr_ppl | step1 vopd | 末步 | 末步 tr_ppl | 末步 ro_ppl | 末步 vopd | 健康判定 |
|------|-------------|------------|------|-------------|-------------|-----------|----------|
| v3-no-ema | 4.52 | 0.122 | 235  | 5.49   | 4.54   | 0.066 | ✅ 健康 |
| v3-SFT    | 1.62e4 | 0.214 | 493  | 1.10e5 | 148.31 | 0.007 | ❌ 退化 |
| v4        | 3.64e4 | 0.210 | 216  | 2.24e4 | 4.88   | 0.126 | ❌ 错配 |
| v5        | 6.54  | 0.495 | -    | -      | -      | -     | ⏳ 待补 |

**核心洞察**：
1. **起点 ppl 决定健康基线**：v3-no-ema / v5（base 9B）step1 tr_ppl=4.52/6.54 正常；v3-SFT / v4（SFT 起点）step1 tr_ppl=1.62e4/3.64e4 = SFT 分布尖锐导致 teacher top-k 外概率大，后续难收敛
2. **vopd_loss 下降不是健康判据**：v3-SFT vopd 从 0.21 降到 0.007 看似收敛最好，实则退化最严重；必须配合 `rollout_ppl` 看 rollout 是否乱码化
3. **固定 teacher（rate=0）+ base student 是稳定配方**：v3-no-ema 已证；v5 在此基础上把 teacher 从 9B 升到 27B，预期更稳更强
4. **EMA teacher 是退化根因**：v3-SFT 与 v3-no-ema 唯一区别是 rate（0.05 vs 0.0），结果 v3-SFT ro_ppl 30 倍增长；EMA 正反馈让 teacher 逐步追 student 的错误分布，最终一起偏
5. **SFT warm-start + 9B 固定 teacher 不适配**：v4 证明 SFT 起点需要更强 teacher（如 27B）才有足够信号矫正

#### 12.5.6 训练健康度巡检口径（建议）

日常巡检只看 3 个指标即可判断任务是否健康，不必看全表：

1. **training_ppl 趋势**：与前 50 step 均值比，若 >2x 且持续 3 step = 预警，>10x = 已退化
2. **rollout_ppl 趋势**：若 ro_ppl / tr_ppl 比 >3 = rollout 失控；指数增长 = 必停
3. **chi2_token 尖峰**：单步 >5 但下步回落 = 正常 outlier；连续 >5 = 分布异常需看样本

vopd_loss / kl 不作为巡检指标——两者都会在退化时给出误导信号（vopd 下降、kl 稳定低）。

