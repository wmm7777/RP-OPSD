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
tmux new -d -s flashnote_train_v2 "bash /data4/wumeimei/flash_note/RP-OPSD/scripts/run_rp_opsd_v2.sh"
```

脚本顶部关键字段（6k 口径，详见 §5.4.2）：

```bash
# 长度字段（6k 口径）
MAX_PROMPT_LENGTH=5120
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))   # 6144
PPO_MAX_TOKEN_LEN_PER_GPU=$MAX_MODEL_LEN                      # 必须等于 MAX_MODEL_LEN

# 路径
OUTPUT_DIR="$PROJECT_ROOT/outputs/flashnote_train_v2"
TRAINER_DEFAULT_LOCAL_DIR="$OUTPUT_DIR/checkpoints"
TRAINER_ROLLOUT_DATA_DIR="$OUTPUT_DIR/rollouts"
```

日志自动 `tee` 到 `$OUTPUT_DIR/logs/train.log`。旧 `run_rp_opsd.sh` / `run_rp_opsd.bak.sh` 已废弃，新增需求一律在 `run_rp_opsd_v2.sh` 顶部改。

> 旧入口 `run_rp_opsd.sh` 内部仍调用 `.bak.sh` 且硬设 `use_dynamic_bsz=False` + 55 step smoke 语义，与 2 epoch 长训练不一致，不再推荐。

#### 5.4.2 关键踩坑（必读，否则 OOM）

| 参数 | 作者默认 | ❌ 错误覆盖 | 后果 |
|---|---|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.7 | 0.5 | CUDA OOM @ `dp_actor.py:703` 的 `torch.logsumexp(logits, dim=-1)` |
| `actor_rollout_ref.model.use_remove_padding` | True | False | 同上，关掉 padding 优化让 logits 矩阵更大 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | ≥ `max_prompt_length + max_response_length` | < `max_model_len` | `AssertionError: max_token_len must be greater than the sequence length` @ `seqlen_balancing.py:382`，进程在 step 1 的 `update_actor` 阶段直接死，8 卡 vLLM engine core 全部退出 |

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
| `response_length/mean` | 平均生成长度 | 稳定（~270 token） | 突降到 <50 = mode collapse；飙到 1024 = 失控 |
| `response_length/clip_ratio` | 达到 1024 截断的比例 | <0.05 | >0.2 大量样本被截断 |
| `response/aborted_ratio` | 生成中断率 | =0.0 | >0 说明 rollout 出错 |

**⑥ 效率与显存**

| 指标 | 含义 | 健康值 | 异常信号 |
|---|---|---|---|
| `perf/max_memory_allocated_gb` | 峰值显存 | <125GB（9B + 8卡） | 接近 145GB reserved 上限要小心 |
| `timing_s/step` | 单步耗时 | 稳定 ~210s（3.5min） | 突然翻倍 = IO/显存瓶颈 |
| `perf/mfu/actor` | actor MFU | 0.2~0.3 | <0.1 说明计算效率低 |
| `perf/throughput` | 吞吐（tokens/s） | >300 | 持续下降 = 卡顿 |

**趋势判读口诀**

- **看趋势不看绝对值**：单 step 值意义有限，看连续 10+ step 的趋势。tensorboard 在 `outputs/flashnote_train_v1/` 下（如有 `tensorboard_log/`）。
- **先看①**：① 任一 = 0 → 蒸馏机制没起，后续指标都没意义，停下来 debug。
- **②+④ 联看**：`vopd_loss` 下降 + `kl` 缓降 + `grad_norm` 稳 = 健康训练；`vopd_loss` 降但 `kl` 涨 = student 在 collapse，蒸馏目标被钻空子。
- **③ 是 on-policy 健康度**：`rollout_is_mean` 偏离 1 + `rollout_is_max` 飙大 = on-policy 假设破坏，要降 `is_clip` 或缩短 rollout-update 间隔。
- **⑤ mode collapse 信号**：`response_length` 突降 + `clip_ratio=0` + `kl→0` = student 坍缩到固定输出，停训。
- **⑥ 显存**：`max_memory_allocated_gb` 持续上涨 = 显存泄漏，会 OOM。

**监控命令**

```bash
# 看最新 step 全量 metrics
grep "step:" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v1/logs/train.log | tail -1

# 只看关键 6 指标的趋势（最近 20 step）
grep "step:" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v1/logs/train.log | tail -20 | \
  grep -oE "step:[0-9]+|actor/vopd_loss:np.float64\([0-9.eE+-]+\)|rollout_corr/kl:np.float64\([0-9.eE+-]+\)|actor/grad_norm:np.float64\([0-9.eE+-]+\)|rollout_corr/ppl_ratio:np.float64\([0-9.eE+-]+\)|response_length/mean:[0-9.]+|self_distillation/num_distill_tokens:np.float64\([0-9.eE+-]+\)"

# 训练进度
grep "Training Progress" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v1/logs/train.log | tail -1

# 报错检查
grep -iE "out of memory|cuda.*error|Traceback|RuntimeError.*killed" /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v1/logs/train.log | tail -5
```

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
