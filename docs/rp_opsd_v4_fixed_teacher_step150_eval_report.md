# RP-OPSD v4 fixed_teacher 评测报告 — global_step_150

> 评测时间：2026-09-03
> 评测机器：m2 (10.162.52.29) GPU 0（vLLM 部署）+ m4 (本机) MOS 评测
> 训练来源：`/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4_fixed_teacher/`
> 评测 checkpoint：`global_step_150`（150/1502 step，约 0.1 epoch，save_freq=150 的首个保存点）

## 1. 实验设置

### 1.1 训练

| 项 | 值 |
|---|---|
| 训练项目 | flash_note RP-OPSD（强化偏好优化 + 自蒸馏） |
| 基座模型 | `/data4/wumeimei/download_models/Qwen3.5-9B`（多模态） |
| 训练框架 | verl（FSDP + ray + vllm rollout） |
| 训练机器 | m1 (10.162.52.30)，4 卡 H20（GPU 0,5,6,7） |
| 训练任务 | `rp_opsd_v4_m1_4gpu`（supervisor 托管，MAX_STEPS=1502，MAX_RETRIES=8） |
| 训练数据 | 72k 条 perception 数据集，2 epoch |
| 当前进度 | 已完成 step 150（首个 save_freq 触发点），训练仍在继续 |

### 1.2 关键超参

```
TRAINER_TOTAL_EPOCHS=2
TRAINER_SAVE_FREQ=150
TRAIN_BATCH_SIZE=96
ROLLOUT_N=8
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=5120
LR=2e-6
LR_WARMUP_STEPS=75
ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
TEACHER_MODEL_SOURCE="legacy"
TEACHER_REGULARIZATION="ema"
TEACHER_UPDATE_RATE=0.05
ROLLOUT_USE_REMOVE_PADDING=True
```

- 自蒸馏 + teacher EMA（rate=0.05，半衰期 ~14 步，75 步 warmup ≈ 5 个半衰期）
- FSDP param/optimizer 双 offload（CPU offload，省显存但 step 慢，~3.9 min/step）
- 2 epoch × 72k / 96 batch ≈ 1502 step，global_step_150 是约 10% 进度

### 1.3 Checkpoint 合并

verl 保存的 FSDP 8-shard 格式不能直接给 vLLM 用，需要合并为 HF 格式：

```bash
# 在 m4 本机执行（verl env + /data4 本地）
conda activate verl_opd_flashnote
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/actor \
  --target_dir /data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/merged_hf
```

**⚠️ 已知 bug：verl model_merger 的 key 前缀嵌套错误**

合并后的 `merged_hf/model.safetensors` key 结构错误：
- 期望：`model.visual.*` + `model.language_model.*` + `lm_head.weight`
- 实际：`model.language_model.visual.*`（多 1 层 `language_model.`）+ `model.language_model.language_model.language_model.*`（多 2 层 `language_model.`）

vLLM 加载会报 `ValueError: Following weights were not initialized from checkpoint: {'visual.blocks.*', ...}`。

修复脚本：`/data4/wumeimei/flash_note/RP-OPSD/scripts/fix_merged_ckpt_keys.py`

- 重映射：`model.language_model.visual.*` → `model.visual.*`
- 重映射：`model.language_model.language_model.language_model.*` → `model.language_model.*`
- 缺失的 `model.mtp.*`（多 token 预测模块）从 base 模型补齐（vLLM 不强制要求，但完整性更好）
- 输出目录：`merged_hf_fixed/`

后续 merge 后的 ckpt 都需要跑一次这个修复脚本。

## 2. 评测部署

### 2.1 vLLM 部署

| 项 | 值 |
|---|---|
| 部署机器 | m2 (10.162.52.29) GPU 0 |
| conda env | `verl_opd_flashnote`（vllm 0.18.0） |
| 模型路径 | `/data4/wumeimei/flash_note/RP-OPSD/outputs/.../global_step_150/merged_hf_fixed/` |
| served-model-name | `flashnote_v4_step150` |
| 端口 | 8000 |
| TP size | 1（单卡） |
| max-model-len | 6144 |
| gpu-memory-utilization | 0.3（~29GB，避开 rui.ni 的 TP=4 vllm 占用 30GB/卡） |
| max-num-seqs | 16 |
| reasoning-parser | `qwen3`（拆分 thinking 到 `reasoning` 字段） |
| trust-remote-code | True |

环境变量：
```bash
source /data1/meimei.wu/miniforge3/etc/profile.d/conda.sh && conda activate verl_opd_flashnote
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # 修复 sqlite3 CXXABI 缺失
export TMPDIR=/dev/shm/vllm_v4_150                          # tmpfs, 避免 NFS IPC bind 失败
export CUDA_VISIBLE_DEVICES=0
```

### 2.2 资源共存

m2 GPU 0-3 被 rui.ni 的 vllm（TP=4，~30GB/卡，100% util）占用。本评测部署挤进 GPU 0：
- rui.ni vllm: 30GB
- 本评测 vllm: 29GB（gpu_mem_util=0.3 × 97GB）
- 合计 59GB / 97GB，38GB headroom，无冲突

GPU 1/2/3 仍是 rui.ni 独占，GPU 4-7 是 sft_gold_397b_lora_r64_m2 训练占用。

### 2.3 推理客户端修复

`test_image_ts_qwen35_9b.py` 原逻辑只读 `message.content`，但 vLLM 0.18 + `reasoning-parser=qwen3`：
- `content`: null
- `reasoning`: 实际输出（vllm 0.18 字段名是 `reasoning`，非 `reasoning_content`）

已补 fallback：
```python
content = (msg.get("content") or "").strip()
if not content:
    content = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
```

## 3. 评测结果

### 3.1 推理

| 项 | 值 |
|---|---|
| 推理脚本 | `/data4/wumeimei/flash_note/infer/test_image_ts_qwen35_9b.py` |
| 样本量 | 220 条/语种 × 4 语种 = 880 条 |
| 语种 | en / fr / ru / zh |
| 模式 | summary-only（跳过 title，请求减半） |
| 并发 | 16 |
| MAX_TOKENS | 1024 |
| enable_thinking | False（对齐 minicpm 评测） |
| 推理耗时 | 514s（8.6 min），0 错误 |
| 平均速度 | 0.6s/条 |
| 推理产物 | `/data4/wumeimei/flash_note/infer/infer_res_0903/flashnote_<lang>_rp_opsd_summary_9b_step150.json` |

### 3.2 MOS 评分

评测模型：gemini-2.5-flash（Transsion 内部 aibotplatform 接入）
评测维度：准确性 / 简洁性 / 完整性 / 格式 / 语种遵循度（0-1）
并发：8

| lang | n (valid) | 准确性 | 简洁性 | 完整性 | 格式 | 语种遵循% |
|------|-----------|--------|--------|--------|------|----------|
| en | 215/220 | 4.48 | 4.72 | 4.99 | 5.00 | 100 |
| fr | 217/220 | 4.63 | 4.70 | 4.99 | 5.00 | 100 |
| ru | 218/220 | 4.55 | 4.76 | 4.99 | 5.00 | 100 |
| zh | 218/220 | 4.57 | 4.76 | 4.99 | 5.00 | 100 |
| **ALL** | **868/880** | **4.56** | **4.74** | **4.99** | **5.00** | **100** |

MOS 产物：`/data4/wumeimei/flash_note/eval_results/eval_res_0903/rp_opsd_summary_9b_step150/<lang>/`

### 3.3 准确性分布

| lang | 5分 | 4分 | 3分 | 2分 | mean | stdev |
|------|----|----|----|----|------|-------|
| en | 134 | 61 | 9 | 11 | 4.48 | 0.80 |
| fr | 162 | 37 | 11 | 7 | 4.63 | 0.73 |
| ru | 146 | 55 | 7 | 10 | 4.55 | 0.77 |
| zh | 156 | 42 | 9 | 11 | 4.57 | 0.80 |

- 62% 样本拿到 5 分（134~162 条/语种）
- 28% 样本拿到 4 分
- 6% 拿 3 分（主体忠实但有 1 处可商榷解读）
- 4% 拿 2 分（明确实体错误或幻觉）
- 无 1 分样本（无主体错误或整段幻觉）

### 3.3.1 各语种 Bad Case 率

> bad case 定义：准确性 ≤ 3 分（含 2 分"明确错误或幻觉" + 3 分"主体忠实但有 1 处可商榷解读"）。4 分及以上视为可接受。

| lang | valid | 2分 | 3分 | 2分率 | bad率(≤3) | 备注 |
|------|-------|----|----|-------|-----------|------|
| en | 215 | 11 | 9 | 5.12% | 9.30% | 2分率最高，OCR/界面识别错多 |
| fr | 217 | 7 | 11 | 3.23% | 8.29% | 2分率最低，但 3 分偏高（边缘 case 多） |
| ru | 218 | 10 | 7 | 4.59% | 7.80% | bad率最低，整体最稳 |
| zh | 218 | 11 | 9 | 5.05% | 9.17% | 2分率与 en 持平，社媒互动数据错多 |
| **ALL** | **868** | **39** | **36** | **4.49%** | **8.64%** | 全语种 bad率 8.64% |

**关键观察：**
- 全语种 bad率（≤3 分）= 8.64%，即 ~91% 样本准确性 ≥ 4 分
- 2 分率（明确错误/幻觉）4.49%，无 1 分样本
- **en/zh 是 bad率最高的语种（~9.3%）**，主要因为社媒互动数据图标归属错（en/zh 高发）+ 外文 OCR 错（en 高发）
- **ru bad率最低（7.8%）**，但 2 分绝对数仍 10 条（视觉细节幻觉如车型/动物识别错）
- **fr 2分率最低（3.23%）**，但 3 分最多（11 条），边缘可商榷解读比例高
- 2 分样本共 39 条（en 11 + fr 7 + ru 10 + zh 11），详见 §3.6

### 3.4 各维度分数分布（按语种）

#### en (valid=215, invalid=5)
- 准确性：{5: 134, 4: 61, 3: 9, 2: 11}
- 简洁性：{5: 155, 4: 60}
- 完整性：{5: 213, 4: 1, 3: 1}
- 格式：{5: 215}
- 错误类别：entity_error=98, predicate_error=13, circumstantial_error=11, out_of_context_error=3
- invalid（Gemini 评测异常未产出有效分）：5 条

#### fr (valid=217, invalid=3)
- 准确性：{5: 162, 4: 37, 3: 11, 2: 7}
- 简洁性：{5: 152, 4: 65}
- 完整性：{5: 215, 4: 2}
- 格式：{5: 217}
- 错误类别：entity_error=78, predicate_error=9, out_of_context_error=3, circumstantial_error=1
- invalid：3 条

#### ru (valid=218, invalid=2)
- 准确性：{5: 146, 4: 55, 3: 7, 2: 10}
- 简洁性：{5: 166, 4: 52}
- 完整性：{5: 216, 4: 2}
- 格式：{5: 218}
- 错误类别：entity_error=87, predicate_error=17, out_of_context_error=2, circumstantial_error=1, linking_error=1
- invalid：2 条

#### zh (valid=218, invalid=2)
- 准确性：{5: 156, 4: 42, 3: 9, 2: 11}
- 简洁性：{5: 166, 4: 51, 3: 1}
- 完整性：{5: 216, 4: 2}
- 格式：{5: 218}
- 错误类别：entity_error=63, predicate_error=15, out_of_context_error=3, circumstantial_error=1
- invalid：2 条

### 3.5 错误类别分布

按事实核查字段统计（排除 `no_error`）：

| 错误类别 | en | fr | ru | zh | total | 占比 |
|---------|----|----|----|----|-------|------|
| entity_error | 98 | 78 | 87 | 63 | 326 | 76% |
| predicate_error | 13 | 9 | 17 | 15 | 54 | 13% |
| circumstantial_error | 11 | 1 | 1 | 1 | 14 | 3% |
| out_of_context_error | 3 | 3 | 2 | 3 | 11 | 3% |
| linking_error | 0 | 0 | 1 | 0 | 1 | <1% |
| grammatical_error | 0 | 0 | 0 | 0 | 0 | 0% |
| coreference_error | 0 | 0 | 0 | 0 | 0 | 0% |
| other_error | 0 | 0 | 0 | 0 | 0 | 0% |

**主要失分集中在 entity_error（76%）**：
- 社媒互动数据图标归属搞混（点赞/评论/分享/收藏数张冠李戴）
- 外文翻译错误（如日文「フローラ」=菌群，被误翻为「氟化物」）
- 单字符/单数字偏差（人名末尾少 1 个字母、高度遮码 ID 末位）

这些多数是 9B 模型 OCR/跨语种理解的固有限制，非 RP-OPSD 训练引入的新问题。

### 3.6 各语种 Bad Case（准确性 2 分样本）

以下每语种展示 3 个 2 分 bad case 的关键错误。

#### en bad case（共 11 条，展示 3 条）

**Case 1: rid=ycP_20260713_095115_418_8j94RKwp**
- 图片：`en_image/b32967e0d50c9fdf55bc9e1ebc973539d822e2940e000689ebeec219b57798d7.jpg`
- 摘要错误：将图中明显的鸡（chicken）误识别为小狗（dog or puppy），核心主体错误
- 次要错误：界面 Follow 按钮样式判断为 Instagram Reel，实际更像 Facebook Reels
- 事实核查：
  - `[entity_error]` "small animal, appearing to be a dog or puppy" → 图中是鸡（可见鸡头、喙、羽毛）
  - `[circumstantial_error]` "screenshot of an Instagram Reel" → 界面 UI 更像 Facebook Reels

**Case 2: rid=Kp3_20260712_072523_958_FpAveXkt**
- 图片：手机锁屏通知列表
- 摘要错误：多处人名/代码 OCR 错误
- 事实核查（5 条 entity_error）：
  - USSD 代码 `*544#` 误写为 `*344#`
  - Facebook 用户名 `Sossygrace Onyango` 误写为 `Sossyana`
  - TikTok 用户名 `ADHILAMBO` 误写为 `ADHILAWO`
  - YouTube 频道名 `Rodony Muangi comedian` 误写为 `Radonyi Awangi camonian`
  - 80% off 折扣消息归到 AliExpress（红色图标），摘要误归 Shopee（橙色购物袋）
  - 商品名 `Miu Miu Bodycon dress` 误读为 `Hu Hu Bodycon dress`

**Case 3: rid=EUm_20260713_105912_484_LoBa4Dnj**
- 图片：Facebook Messenger 风格聊天截图
- 摘要错误：
  - `[circumstantial_error]` 应用识别错误：判断为 Facebook Messenger，实际从顶部用户名位置+相机图标看更像 Instagram
  - `[entity_error]` 视频标题幻觉：摘要称"PART 1: HONEYMOON TRIP"，图中实际是"WHAT IS AUGUSTINE'S ABOUT?"（完全虚构）

#### fr bad case（共 7 条，展示 3 条）

**Case 1: rid=KMf_20260727_194726_111_aa7g2MGw**
- 图片：WhatsApp 对话+智能电表照片
- 摘要错误：电表技术参数多处数值错误
- 事实核查（6 条 entity_error）：
  - 屏幕数值 `6.40` 误读为 `640`（漏小数点）
  - 电压 `230V` 误写为 `220V`
  - 电流 `5(60)A` 误写为 `6(60)A`
  - 脉冲常数 `1000 imp/kWh` 误写为 `1200 imp/kWh`
  - 型号 `DDSY101` 误写为 `DGB0101`
  - 标准编号 `CEI 62055-31/41` 误写为 `CEI 62054-11`

**Case 2: rid=26E_20260727_231511_155_Ux6X0MPP**
- 图片：YouTube 儿童动画视频推荐界面
- 摘要错误：数量级误读
- 事实核查：
  - `[entity_error]` 视图数 `3 Md de vues`（法语 Md=milliards=30 亿）误读为 `3 millions`（300 万），误差 3 个数量级
  - `[entity_error]` 字幕 `Wah, wah, wah!` 多写一个 l 变成 `wahl`

**Case 3: rid=XFz_20260727_184055_863_Nwf9XUSz**
- 图片：WhatsApp 对话+银行存款单据
- 摘要错误：银行单据细节多处 OCR 错误 + 姓名/日期幻觉
- 事实核查：
  - `[entity_error]` 银行名 `SCB Cameroun` 误写为 `SGR Cameroon`
  - `[entity_error]` 单据编号 `079666` 误写为 `070666`；年份 `2020` 误写为 `2026`
  - `[entity_error]` 客户名 `LIOKEA` + 汇款人 `MBAKU EMMANUEL MUDOH` 完全误识别为 `OUEDRAOUI` / `MBEUKI`
  - `[out_of_context_error]` 将"本次存款金额"误读为"剩余余额"

#### ru bad case（共 10 条，展示 3 条）

**Case 1: rid=zXz_20260728_115744_588_JRINCYiC**
- 图片：水上乐园价格表 VK 帖子
- 摘要错误：凭空添加服务项 + 价格归因错误
- 事实核查：
  - `[entity_error]` 包含服务凭空加了"淋浴（душ）"和"躺椅（шезлонг）"，图中只有泳池/按摩浴缸/木桶桑拿
  - `[entity_error]` 9000 ₽ 凉亭租金归因为"连续三个周末"，图中实际是"20 人以内大凉亭"
  - `[entity_error]` 生日优惠从"赠送凉亭"误改为"获取门票"

**Case 2: rid=nD3_20260728_212611_883_UVBURj7n**
- 图片：Chi Gap 通话记录界面
- 摘要错误：状态归因 + 时间计数双重错
- 事实核查：
  - `[entity_error]` "未接听"状态归给两个联系人，实际只有 110 Занак 有未接听，Модар 全是呼入
  - `[entity_error]` 时间分布计数错误：今天 3 条（实际 5）、昨天 1 条（实际 3）、周日 2 条（实际 3）

**Case 3: rid=jBw_20260728_211909_905_nhXhUo6q**
- 图片：二手车出售广告（Faw Vita 2008）
- 摘要错误：视觉细节识别 + 地区代码归属错误
- 事实核查：
  - `[entity_error]` 车型判断错：两厢车（Hatchback）误识别为旅行车（универсал）
  - `[entity_error]` 车牌地区代码 155 归为巴什科尔托斯坦共和国，实际是欧姆斯克州（Omsk Oblast）
  - `[entity_error]` 前保险杠雾灯孔洞误认为"损坏的牵引钩（фаркоп）"
  - `[entity_error]` 引擎盖漆面剥落/腻子误认为"污垢"

#### zh bad case（共 11 条，展示 3 条）

**Case 1: rid=P45_20260726_180830_490_CkySFrkm**
- 图片：抖音短视频截图（麻将+婚庆主题）
- 摘要错误：社媒互动数据图标归属搞混
- 事实核查：
  - `[entity_error]` 9.5 万是分享数（箭头图标），摘要误报为"9.5 万收藏"；实际收藏数（星星图标）是 6633
- 简洁性也被扣到 3 分：详细罗列点赞/评论/收藏数属于冗余

**Case 2: rid=NAu_20260726_203443_855_dbnoSwxC**
- 图片：聊天截图（关于鞋子，含宿务语 Cebuano）
- 摘要错误：宿务语 "Naa"（有/在）误译为"不在"，导致结论完全相反
- 事实核查：
  - `[predicate_error]` 用户回复 "Naa dw ingon roda"（Roda 说有鞋子），摘要误为"回复称鞋子不在"
  - `[predicate_error]` Leilani 最后说 "Naa man"（有的/在那儿），摘要误为"确认鞋子确实不在"

**Case 3: rid=Bus_20260727_074854_202_F55MkYIw**
- 图片：短视频（动物喂食）
- 摘要错误：动物种类识别错
- 事实核查：
  - `[entity_error]` 画面动物有长而下垂的耳朵和明显的角，是山羊幼崽，摘要误为"棕色幼犬"

### 3.7 Bad Case 共性问题归纳

跨语种反复出现的 4 类错误模式：

1. **社媒互动数据图标归属错（en/zh 高发）**
   - 星星/箭头/心形/气泡 对应 收藏/分享/点赞/评论 搞混
   - 数值正确但归属错，量级差异可达 10x（如 zh case 1：9.5 万 vs 6633）

2. **外文 OCR 错（en/fr 高发）**
   - 人名末尾字母错（ADHILAMBO→ADHILAWO, Rodony→Radonyi）
   - 小语种文字翻译错（日文フローラ=菌群→氟化物；宿务语 Naa=有→不在）
   - 数值漏小数点/数量级误读（6.40→640, 3 Md→3 millions）

3. **跨语种界面识别错（en/fr/ru）**
   - Facebook Reels vs Instagram Reels 混淆
   - Facebook Messenger vs Instagram DM 混淆
   - TikTok vs YouTube 儿童动画界面混淆

4. **视觉细节幻觉（ru/zh 高发）**
   - 凉亭包含服务凭空添加（淋浴/躺椅）
   - 动物种类误识别（鸡→狗，山羊→犬）
   - 车型/车况误判（两厢→旅行车，雾灯孔→牵引钩，漆面剥落→污垢）

这些 bad case 反映 9B 模型在 OCR 精度、跨语种理解、视觉细节分辨上的固有限制。RP-OPSD 训练在 step_150 尚未显著改善这些问题，需要观察后续 step 是否有提升。

## 4. 与其他实验对比

| 实验 | 准确性 | 简洁性 | 完整性 | 格式 | 语种% | 备注 |
|------|--------|--------|--------|------|-------|------|
| **rp_opsd_v4_fixed_step150** | **4.56** | **4.74** | **4.99** | **5.00** | **100** | 本报告，训练 10% 进度 |
| sft_gold_397b_lora_r64_m2 epoch1.0 | 参见对应报告 | | | | | LoRA SFT 同期对照 |
| sft_gold_397b (full SFT) | 参见对应报告 | | | | | m3 全参 SFT |

> global_step_150 是 RP-OPSD v4 训练的**首个 checkpoint**（save_freq=150），仅完成训练 10%。当前分数反映 base 模型经短期 RL 微调后的表现，后续 step（300/450/.../1502）的 ckpt 需继续评测以观察训练曲线。

## 5. 关键坑位记录

### 5.1 verl model_merger triple-nested key bug

- **现象**：合并后 safetensors 的 key 被嵌套成 `model.language_model.language_model.language_model.*`（多 2 层）+ `model.language_model.visual.*`（多 1 层）
- **影响**：vLLM 加载报 `ValueError: Following weights were not initialized`，visual 和 language_model 权重都加载不到
- **根因**：verl model_merger 在 FSDP shard 合并时，对多模态模型（Qwen3_5ForConditionalGeneration）的 prefix 处理有 bug，重复加了 `language_model.`
- **修复**：`scripts/fix_merged_ckpt_keys.py`，重映射 key + 从 base 模型补 `model.mtp.*`
- **后续**：每个 verl merge 后的 ckpt 都要跑一次 fix 脚本才能给 vLLM 用

### 5.2 vLLM 0.18 reasoning 字段名变更

- **现象**：`--reasoning-parser qwen3` 时，输出在 `reasoning` 字段，`content` 为 null
- **影响**：`test_image_ts_qwen35_9b.py` 原逻辑只读 `content`，会拿到空字符串
- **修复**：补 fallback 到 `reasoning` / `reasoning_content`（已改）

### 5.3 TMPDIR 必须用 tmpfs

- verl/ray/vllm 启动时 TMPDIR 如果是 NFS 路径，IPC socket bind 会失败
- m1 的 `/data1` 是 NFS 且 97% 满，不能用
- 必须用 `/dev/shm/<name>`（tmpfs，965G，支持 IPC socket bind）
- task TRAIN_CMD 和 deploy 脚本的 `export TMPDIR=` 两处都要改

### 5.4 m2 共用机部署

- rui.ni 的 vllm（TP=4，GPU 0-3，~30GB/卡）在跑，不能整卡占用
- 本评测用 `gpu_memory_utilization=0.3`（~29GB）挤进 GPU 0
- 合计 59GB / 97GB，无冲突
- 但 GPU 0 是 100% util（rui.ni 在跑推理），本评测的推理速度受共享影响

## 6. 产物路径汇总

| 类型 | 路径 |
|------|------|
| 训练 ckpt（FSDP shard） | `outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/actor/` |
| 合并后 HF ckpt（原始，有 bug） | `outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/merged_hf/` |
| 合并后 HF ckpt（修复后，可用） | `outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/merged_hf_fixed/` |
| 推理产物 | `/data4/wumeimei/flash_note/infer/infer_res_0903/flashnote_<lang>_rp_opsd_summary_9b_step150.json` |
| MOS 评测产物 | `/data4/wumeimei/flash_note/eval_results/eval_res_0903/rp_opsd_summary_9b_step150/<lang>/` |
| 合并修复脚本 | `/data4/wumeimei/flash_note/RP-OPSD/scripts/fix_merged_ckpt_keys.py` |
| 训练脚本（m1） | `/data1/meimei.wu/run_rp_opsd_v4_m1_4gpu.sh` |
| 训练 task 配置 | `/data4/wumeimei/meimei_agent/monitors/tasks/rp_opsd_v4_m1_4gpu.task` |

## 7. 下一步

- [ ] 等 RP-OPSD v4 训练继续，评测 step 300/450/.../1502 的 ckpt，画训练曲线
- [ ] 对比 RP-OPSD v4 与 sft_gold_397b_lora_r64_m2 的差异（同 base 模型，训练方法不同）
- [ ] 分析 entity_error 高频场景，看是否能通过训练数据改进
- [ ] 修 verl model_merger 上游 bug（PR 到 verl 仓库）
