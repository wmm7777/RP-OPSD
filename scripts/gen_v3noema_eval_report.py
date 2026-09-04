#!/usr/bin/env python3
"""生成 v3_no_ema step150/300 评测报告"""
import json, statistics
from collections import Counter

BASE = "/data4/wumeimei/flash_note/eval_results/eval_res_0904"
OUT = "/data4/wumeimei/flash_note/RP-OPSD/docs/rp_opsd_v3_no_ema_eval_report.md"
steps = [150, 300]
langs = ["en", "fr", "ru", "zh"]
LANG_NAME = {"en": "English", "fr": "Français", "ru": "Русский", "zh": "中文"}

results = {}
for step in steps:
    for L in langs:
        f = f"{BASE}/rp_opsd_v3noema_summary_9b_step{step}/{L}/summary_mos_results.json"
        with open(f) as fp:
            data = json.load(fp)
        valid = [d for d in data if d.get("valid_content") and d.get("准确性")]
        results[(step, L)] = valid

def mean(xs):
    return statistics.mean(xs) if xs else 0.0

lines = []
def w(s=""):
    lines.append(s)

w("# RP-OPSD v3_no_ema 训练评测报告")
w()
w("**模型**: flashnote RP-OPSD v3_no_ema (Qwen3.5-9B, Resolution-Privileged On-Policy Self-Distillation, no EMA teacher)")
w()
w("**评测 ckpt**:")
w("- `step150`（global_step_150，9/3 22:15 保存）")
w("- `step300`（global_step_300，9/4 05:38 保存）")
w()
w("**数据集**: 4 门语种 × 220 条/语种 = 880 条图片摘要（summary 任务，无 title）")
w("- en: `data_image_en.xlsx` / fr: `data_images_fr.xlsx` / ru: `data_images_ru.xlsx` / zh: `data_image_zh.xlsx`")
w()
w("**评委**: `gemini-3-flash-preview`（thinkingConfig=medium，temperature=0）")
w("- 评测维度：准确性 / 简洁性 / 完整性 / 格式（1-5 分）；语种遵循度（0/1 二值）")
w("- 评测脚本：`/data4/wumeimei/flash_note/auto_eval/evaluators/run_multilang_eval.py --modes flash_summary_mos`")
w()
w("**推理部署**: 机器2 GPU 4/5（与 rui.ni open_clip 共卡，`gpu_memory_utilization=0.40`）")
w("- vllm v0.19.1, `--max-model-len 32768 --enforce-eager --reasoning-parser qwen3`")
w("- 推理耗时：step150/300 并行 4 门语种，总耗时约 10 分钟（0 错误，880/880 成功）")
w("- 评测耗时：约 2h30min（gemini-3-flash-preview 思考模式，concurrency=4）")
w()
w("**FSDP ckpt merge**: 8-rank FSDP shard → 单文件 safetensors（18GB/ckpt），CPU-only merge，swift env")
w("- `outputs/flashnote_train_v3_no_ema/merged/step_150_m2/`")
w("- `outputs/flashnote_train_v3_no_ema/merged/step_300_m2/`")
w()
w("---")
w()
w("## 1. 总体平均分对比")
w()
w("| ckpt | N | 准确性 | 简洁性 | 完整性 | 格式 | 语种遵循 | 总均分 |")
w("|------|---|--------|--------|--------|------|----------|--------|")
for step in steps:
    all_data = []
    for L in langs:
        all_data.extend(results[(step, L)])
    n = len(all_data)
    acc = mean([d["准确性"]["分数"] for d in all_data])
    conc = mean([d["简洁性"]["分数"] for d in all_data])
    comp = mean([d["完整性"]["分数"] for d in all_data])
    fmt = mean([d["格式"]["分数"] for d in all_data])
    lf = mean([d["语种遵循度"]["分数"] for d in all_data])
    avg = (acc + conc + comp + fmt) / 4
    w(f"| step{step} | {n} | {acc:.3f} | {conc:.3f} | {comp:.3f} | {fmt:.3f} | {lf:.3f} | {avg:.3f} |")
w()
w("**结论**：step300 相比 step150 在整体上**仅微弱提升 0.016 分**（4.679→4.695），几乎无差异。两个 ckpt 均存在相似的错误模式，说明从 step150 到 step300 的训练并没有针对性改善这些 badcase 模式。")
w()
w("---")
w()
w("## 2. 各 ckpt × 各语种 平均分")
w()
w("| ckpt | 语种 | N | 准确性 | 简洁性 | 完整性 | 格式 | 语种遵循 | 均分 |")
w("|------|------|---|--------|--------|--------|------|----------|------|")
for step in steps:
    for L in langs:
        data = results[(step, L)]
        n = len(data)
        acc = mean([d["准确性"]["分数"] for d in data])
        conc = mean([d["简洁性"]["分数"] for d in data])
        comp = mean([d["完整性"]["分数"] for d in data])
        fmt = mean([d["格式"]["分数"] for d in data])
        lf = mean([d["语种遵循度"]["分数"] for d in data])
        avg = (acc + conc + comp + fmt) / 4
        w(f"| step{step} | {L} | {n} | {acc:.3f} | {conc:.3f} | {comp:.3f} | {fmt:.3f} | {lf:.3f} | {avg:.3f} |")
w()
w("**语种间对比**：")
w("- 中文 zh 在 step150 最高（4.768），在 step300 略降（4.730）")
w("- 英文 en 在 step300 最高（4.701），step150 最低（4.632）")
w("- 法语 fr、俄语 ru 两 ckpt 均接近，变化不显著")
w("- 完整性（~4.93）、格式（~4.96）、语种遵循（~1.00）三门接近满分，模型在结构化输出和语种遵循上表现稳定")
w("- 准确性（~4.27-4.37）是最薄弱维度，badcase 集中在此")
w()
w("---")
w()
w("## 3. Badcase 汇总（准确性 ≤ 2 分）")
w()
w("| ckpt | 语种 | N_total | N_bad | Bad 比例 |")
w("|------|------|---------|-------|----------|")
for step in steps:
    for L in langs:
        data = results[(step, L)]
        bad = [d for d in data if d["准确性"]["分数"] <= 2]
        ratio = len(bad) / max(len(data), 1)
        w(f"| step{step} | {L} | {len(data)} | {len(bad)} | {ratio:.3%} |")
w()
all_bad_count = sum(1 for s in steps for L in langs for d in results[(s, L)] if d["准确性"]["分数"] <= 2)
all_total = sum(len(results[(s, L)]) for s in steps for L in langs)
w(f"| **合计** | - | {all_total} | {all_bad_count} | {all_bad_count/all_total:.3%} |")
w()
w("**整体 badcase 比例 ~9.2%**。step150 共 76 个 badcase，step300 共 76 个 badcase，数量完全相同。")
w()
w("---")
w()
w("## 4. 错误类型分布（按事实核查第一条非 no_error 分类）")
w()
err_counter = Counter()
for step in steps:
    for L in langs:
        for d in results[(step, L)]:
            if d["准确性"]["分数"] <= 2:
                for fc in d.get("事实核查", []):
                    if fc.get("类别") != "no_error":
                        err_counter[fc.get("类别", "unknown")] += 1
                        break

w("| 错误类型 | 总数 | 占比 | 说明 |")
w("|----------|------|------|------|")
total_err = sum(err_counter.values())
type_desc = {
    "entity_error": "实体错误：数值/名称/对象/图标归属错误（如点赞数 10x 偏差、电表型号抄错、消息归属颠倒）",
    "predicate_error": "谓词错误：动作/关系/主客体颠倒（如把A发的说成B发的、把朝圣说成旅行事故）",
    "circumstantial_error": "情境错误：界面/场景识别错误（如 WhatsApp vs Telegram、TikTok vs Instagram Reels）",
    "out_of_context_error": "语境脱离：凭空捏造图中无依据的信息（幻觉）",
    "grammatical_error": "语法错误：句子结构混乱导致难理解",
    "coreference_error": "共指错误：代词指代错误",
    "linking_error": "链接错误：句间时序/因果关联错误",
    "other_error": "其他",
}
for k, v in err_counter.most_common():
    w(f"| {k} | {v} | {v/total_err:.1%} | {type_desc.get(k, '')} |")
w()
w("**关键发现**：`entity_error` 占 72%（111/152），是绝对主导的错误类型。其次是 `predicate_error`（15%）、`circumstantial_error`（9%）。改进应优先聚焦于数值识别、图标归属、消息发送方判断。")
w()
w("---")
w()
w("## 5. 各 ckpt × 各语种 Badcase 详情")
w()

for step in steps:
    w(f"### step{step}")
    w()
    for L in langs:
        data = results[(step, L)]
        bad = [d for d in data if d["准确性"]["分数"] <= 2]
        w(f"#### {L} ({LANG_NAME[L]}) — {len(bad)} 个 badcase")
        w()
        if not bad:
            w("（无 badcase）")
            w()
            continue
        w("| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |")
        w("|---|---------|----------|------------------|----------|-------------------|")
        for i, d in enumerate(bad, 1):
            cid = d.get("case_id", "?")
            acc = d["准确性"]["分数"]
            err_type, err_sentence, err_explain = "—", "", ""
            for fc in d.get("事实核查", []):
                if fc.get("类别") != "no_error":
                    err_type = fc.get("类别", "—")
                    err_sentence = fc.get("句", "").replace("|", "\\|")[:150]
                    err_explain = fc.get("说明", "").replace("|", "\\|").replace("\n", " ")[:200]
                    break
            reason = d["准确性"]["理由"].replace("|", "\\|").replace("\n", " ")[:250]
            w(f"| {i} | {cid} | {err_type} | {err_sentence} | {err_explain} | {reason} |")
        w()
        # 各 badcase 完整理由
        w("<details><summary>展开各 badcase 完整理由</summary>")
        w()
        for i, d in enumerate(bad, 1):
            cid = d.get("case_id", "?")
            acc = d["准确性"]["分数"]
            reason = d["准确性"]["理由"]
            err_type, err_sentence, err_explain = "—", "", ""
            for fc in d.get("事实核查", []):
                if fc.get("类别") != "no_error":
                    err_type = fc.get("类别", "—")
                    err_sentence = fc.get("句", "")
                    err_explain = fc.get("说明", "")
                    break
            w(f"**step{step}-{L}-case{cid}** (准确性={acc}, 类型={err_type})")
            w(f"- 错句: {err_sentence[:300]}")
            w(f"- 说明: {err_explain[:400]}")
            w(f"- 完整理由: {reason}")
            w()
        w("</details>")
        w()

w("---")
w()
w("## 6. 典型错误模式与改进建议")
w()
w("### 6.1 数值/数量级识别错误（entity_error，最频繁）")
w()
w("- **症状**：把 `105.3K`（10.53 万）写成 `105.3万`（放大 10×）；`79K`（7.9 万）写成 `7.9千`（缩小 10×）；`222K` 写成 `22K`；电表 `6.40 kWh` 写成 `640 kWh`；`5(80)A` 写成 `6/80A`；`109` 关卡号误为分数。")
w("- **根因**：模型对 K/M 单位换算不严格，对小数点位置敏感度低；UI 数字的图标归属推断不够严谨。")
w("- **改进**：在 summary 训练数据中增加互动数据 K/M 换算的负例；在 prompt 中要求逐个列出「图标→指标→数值」的对应关系后再整合。")
w()
w("### 6.2 消息发送方归属颠倒（predicate_error）")
w()
w("- **症状**：WhatsApp/Telegram 界面中左右气泡方向判错，把对方发的语音/文字说成用户发的，把 9 秒语音归属到错误的人；`00:28`（凌晨 12:28 发送时间）误为视频时长。")
w("- **根因**：模型未显式利用「左气泡=对方，右气泡=自己」这一稳定 UI 约定。")
w("- **改进**：训练数据补 IM 界面的发送方显式标注；prompt 中要求先标「左/右气泡→发送方」再总结。")
w()
w("### 6.3 平台/界面识别错误（circumstantial_error）")
w()
w("- **症状**：把 Instagram Reels 识别为 TikTok；把 WhatsApp 识别为 Telegram；把短信/SMS 界面识别为 WhatsApp；把 TikTok 观看历史识别为 Instagram。")
w("- **根因**：模型对不同平台 UI 细节差异（底部导航栏、绿色顶栏、相机/电话图标位置）不敏感。")
w("- **改进**：训练集中补多平台 UI 标注样本；prompt 中要求先识别「界面类型+依据」。")
w()
w("### 6.4 语种识别错误（entity_error）")
w()
w("- **症状**：把库尔德语（西里尔字母）误为波斯语；塔吉克语（西里尔字母）误为俄语；印尼语俚语 `Boles`（=Boleh，意为\"可以\"）误为\"身体不适\"；`jam set 8`（印尼语\"7:30\"）误为\"8 PM\"。")
w("- **根因**：模型对小语种（库尔德语、塔吉克语）和俚语缩写缺乏知识；倾向于套用主流语种（波斯语、俄语）的解释。")
w("- **改进**：训练数据补充小语种 + 俚语样本；prompt 中要求先做语种识别并给出依据。")
w()
w("### 6.5 幻觉/捏造（out_of_context_error）")
w()
w("- **症状**：把视频标题误为另一视频标题；把背景文字 `LA FORGE DES CHANSONS` 误为 `L'ORGANISATION DES CHANSONS`；把电表型号 `DDSD101` 抄成 `DGBD101`；凭空捏造文档标题 `FINANCEMENT PAR L'ÉTAT`。")
w("- **根因**：模型在转录可见文字时\"脑补\"近似词而非严格逐字转录；在缺乏明确文字时倾向于合理化猜测。")
w("- **改进**：训练数据补 OCR 严格转录任务；prompt 中要求\"逐字转录图内可见外文原文\"后再翻译。")
w()
w("### 6.6 重复/looping 生成")
w()
w("- **症状**：摘要末尾出现重复短语或循环内容，多见于 ru 语种 step300 case 7。")
w("- **根因**：解码 repetition penalty 不够，或模型在 summary 长度约束下的退化。")
w("- **改进**：vllm serve 增加 `--repetition-penalty` 或调整 `frequency_penalty`；训练数据剔除重复段落。")
w()
w("---")
w()
w("## 7. step150 vs step300 对比结论")
w()
w("| 维度 | step150 | step300 | 差值 |")
w("|------|---------|---------|------|")
all_150 = [d for L in langs for d in results[(150, L)]]
all_300 = [d for L in langs for d in results[(300, L)]]
for dim, key in [("准确性","准确性"),("简洁性","简洁性"),("完整性","完整性"),("格式","格式"),("语种遵循度","语种遵循度")]:
    m150 = mean([d[key]["分数"] for d in all_150])
    m300 = mean([d[key]["分数"] for d in all_300])
    diff = m300 - m150
    sign = "+" if diff > 0 else ""
    w(f"| {dim} | {m150:.3f} | {m300:.3f} | {sign}{diff:.3f} |")
w(f"| **总均分** | **4.679** | **4.695** | **+0.016** |")
w()
w("**关键观察**：")
w("1. step300 相比 step150 在所有维度上**几乎无差异**（差值均在 ±0.05 量级），准确性 +0.041、简洁 +0.016，但 zh 准确性反而下降 0.081。")
w("2. badcase 数量在两个 ckpt 完全相同（76 vs 76），错误模式也一致，说明训练并未针对性修复这些 case。")
w("3. 推测原因：RP-OPSD v3_no_ema 的 EMA 关闭后，自蒸馏信号变弱；从 step150 到 step300 的 150 步训练可能只是平滑收敛，未引入新的能力增量。")
w("4. 建议：是否继续训练到 step450/600 看是否出现拐点；或对比 v3（带 EMA）版本看 EMA 是否对 badcase 修复有实质帮助。")
w()
w("---")
w()
w("## 8. 评测产物")
w()
w("- 推理结果：`/data4/wumeimei/flash_note/infer/infer_res_0904/flashnote_{lang}_rp_opsd_v3noema_summary_9b_step{150,300}.json`")
w("- MOS 评测 JSON：`/data4/wumeimei/flash_note/eval_results/eval_res_0904/rp_opsd_v3noema_summary_9b_step{150,300}/{en,fr,ru,zh}/summary_mos_results.json`")
w("- 评测 log：`/data4/wumeimei/flash_note/infer/logs/mos_v3_m2_step{150,300}_0904_0920.log`")
w("- Merged ckpt：`/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v3_no_ema/merged/step_{150,300}_m2/`")
w()
w("报告生成时间: 2026-09-04 12:00")
w()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"written to {OUT}")
print(f"total lines: {len(lines)}")
