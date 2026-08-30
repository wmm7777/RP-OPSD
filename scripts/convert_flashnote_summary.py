#!/usr/bin/env python3
"""flash_note summary jsonl -> RP-OPSD verl parquet 转换（自文档化）

================================================================================
 任务：flash_note summary
   图片 -> 生成多语种摘要（≤500 单位），覆盖 en / fr / ru / zh 四门语种。
   "1 单位" = 1 个汉字（zh）或 1 个词（en/fr/ru 按词）。

 方法：RP-OPSD（Resolution-Privileged On-Policy Self-Distillation，arXiv:2607.24447）
   自蒸馏：Teacher = Student = Qwen3.5-9B，Teacher 是 Student 的 EMA 影子权重（ρ=0.05）。
   唯一特权 = 分辨率：Student 看 1/2 物理分辨率图，Teacher(EMA) 看原图。
   师生 prompt 完全相同（无答案特权、无 ref、reward-free）。
   Loss = bias-corrected Teacher Top-100 reverse KL（alpha=1.0，无 tail，is_clip 2.0）。
   复用仓库 verl 代码训练，本脚本只负责把 summary 数据转成 verl 期望的 parquet。

 提示词
   4 门语种各一份全文翻译（PROMPT_TEMPLATE_en/_fr/_ru/_zh），非"英文模板换语种词"。
   翻译保持 Markdown 结构一致（标题/项目符号/粗体），仅自然语言本地化；
   "500 单位"计数规则按各语种口径适配（fr/ru/en 按词、zh 按字）。
   运行时按图片路径检测语种，套用对应翻译提示词（不从 jsonl 透传 messages）。

 原始数据
   源文件: /data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl
   原图目录(teacher 用): /data4/wumeimei/flash_note/train/<lang>_image/*.jpg
   半分辨率图(student 用): /data4/wumeimei/flash_note/train/<lang>_image_lr/*.jpg
                       (由 gen_lr_images.py 把原图宽高各 /2 生成)
   图片后缀: 绝大多数 .jpg，少量 .jpeg / .png / .webp
   每条: 1 张图 + 1 条 user message（content 含 1 个 <image> 占位符 + summary 指令）

 语种数据量（采样前 / 采样后，每门封顶 2.2W）
   en: 40000 -> 22000   （超 2.2W，采样到 2.2W）
   fr: 28182 -> 22000   （超 2.2W，采样到 2.2W）
   ru: 14773 -> 14773   （不足 2.2W，全量）
   zh: 13393 -> 13393   （不足 2.2W，全量）
   合计: 96348 -> 72166

 输出 parquet schema（verl RLHFDataset 期望，见 verl/utils/dataset/rl_dataset.py）
   prompt         list<struct<role:str, content:str>>   messages，content 含 <image> 占位符
   images          list<struct<image:str>>             Student 用图（半分辨率路径）
   teacher_images  list<struct<image:str>>             Teacher 用图（原图路径）
   extra_info      struct<index:int, task_family:str>  index + "summary_<lang>"

 用法
   python convert_flashnote_summary.py \
       --src /data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl \
       --out .runtime/flashnote_summary/train.parquet
   # 先生成半分辨率图再 --check-lr 校验
   python convert_flashnote_summary.py --src ... --out ... --check-lr
   # 查看内嵌的提示词模板（默认全部，可 --lang fr 只看指定语种）
   python convert_flashnote_summary.py --show-prompt
   python convert_flashnote_summary.py --show-prompt --lang zh
================================================================================
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ──────────────────────────────────────────────────────────────────────────────
# 常量：原始数据地址
# ──────────────────────────────────────────────────────────────────────────────
SRC_DEFAULT = "/data4/wumeimei/flash_note/flashnote_useropsd_summary_train_aligned.jsonl"
TRAIN_ROOT = "/data4/wumeimei/flash_note/train"
# 原图(teacher) -> train/<lang>_image/，半分辨率图(student) -> train/<lang>_image_lr/

# 语种 + 原始数据量（采样前），供文档与校验对照
LANG_PATH_RE = re.compile(r"/(en|fr|ru|zh)_image/")
LANG_ORIGINAL_COUNTS = {"en": 40000, "fr": 28182, "ru": 14773, "zh": 13393}

# ──────────────────────────────────────────────────────────────────────────────
# 提示词模板：4 门语种各一份全文翻译。运行时按图片路径检测语种后套用。
# 翻译保持 Markdown 结构一致（标题/项目符号/粗体），仅自然语言本地化；
# "500 单位"计数规则按各语种口径适配（fr/ru/en 按词、zh 按字）。
# ──────────────────────────────────────────────────────────────────────────────



# 四语种独立变量版 图文摘要提示词（ZH/EN/FR/RU）
# 四语种完整独立变量 可直接用于代码字典配置
PROMPT_TEMPLATE_zh = """<image>
核心身份
你是专业的图文理解与摘要撰写专家，擅长从图片中提取有效关键信息，生成精准、简洁、结构清晰的中文摘要。
强制规则（最高优先级）
所有输出必须严格遵守以下硬性要求，违反任意一条即视为输出无效：
1. 语言规则：仅输出中文内容，禁止出现其他语种文字。
2. 内容真实规则：仅使用图片中可识别的内容，禁止主观推测、引用外部知识、编造虚假信息。
3. 字数规则：摘要总字数不得超过500个汉字。
4. 质量规则：仅提炼核心有效信息，禁止机械照搬原图内容，杜绝冗余、空洞描述。
任务与输出规范
根据图片可视内容生成标准摘要，依据图片信息密度自适应匹配输出结构：
模式一：高信息密度图片
适用于包含大量要点、数据、步骤、观点、规则的图片，采用Markdown结构化格式输出：
1. 开篇：用1-2句精简语句概括图片核心主题（必填）。
2. 主体内容：以有序或无序列表展示关键信息。
3. 列表规范：每条内容必须以简短加粗小标题开头，后跟严格贴合原图、客观简洁的文字说明。
模式二：简单低信息密度图片
适用于单一主题、内容简单的图片，采用纯段落自然语言输出，无需多余Markdown格式，精准输出唯一核心信息即可。
补充规则
1. 若图片有效信息极少，输出最精准、最简洁的核心摘要即可。
2. Markdown符号仅用于排版，不计入字数统计。
3. 输出内容全程逻辑清晰、客观中立、精炼无冗余。"""

PROMPT_TEMPLATE_en = """<image>
Core Identity
You are a professional image-text comprehension and summary expert. You specialize in extracting valid key information from images and generating accurate, concise, well-structured English summaries.
Mandatory Rules (Highest Priority)
All outputs must strictly comply with the following hard constraints. Any violation makes the output invalid:
1. Language Rule: Output only English, no other languages are allowed.
2. Authenticity Rule: Only use recognizable content from the image. No subjective speculation, no external knowledge, no fabricated information.
3. Length Rule: The total summary length must not exceed 500 English words.
4. Quality Rule: Extract core information only. Do not copy original content mechanically, and avoid redundant or empty descriptions.
Task & Output Standards
Generate a standard English summary based on the visible content of the image. Adapt your output structure according to the information density of the image:
Mode 1: Information-Dense Images
Applicable to images containing multiple key points, data, steps, opinions or rules. Use structured Markdown format:
1. Opening: 1–2 concise sentences to summarize the core theme of the image (required).
2. Main content: List key information in ordered or unordered lists.
3. List specification: Each list item must start with a short bold subheading, followed by a concise and objective explanation based strictly on the image content.
Mode 2: Simple & Low-Density Images
Applicable to images with a single topic or simple content. Use plain paragraph natural language without redundant Markdown formatting, and accurately output the only core information.
Supplementary Rules
1. If the image contains extremely limited information, output the most accurate and concise summary available.
2. All Markdown symbols are only for layout and do not count towards the word limit.
3. Keep all content logical, objective and refined."""

PROMPT_TEMPLATE_fr = """<image>
Identité Professionnelle
Vous êtes un expert professionnel de la compréhension image-texte et de la rédaction de résumés. Vous êtes spécialisé dans l’extraction des informations clés valides des images et dans la production de résumés français précis, concis et bien structurés.
Règles Obligatoires (Priorité Maximale)
Toute sortie doit respecter strictement les contraintes suivantes. Toute violation rend le résultat invalide :
1. Règle de langue : Produisez uniquement du texte en français, aucune autre langue n’est autorisée.
2. Règle d’authenticité : Utilisez exclusivement le contenu visible et reconnaissable de l’image. Aucune spéculation subjective, aucune connaissance externe, aucune information inventée.
3. Règle de longueur : Le résumé ne doit pas dépasser 500 mots français.
4. Règle de qualité : Extrayez uniquement les informations essentielles. Ne recopiez pas mécaniquement le contenu de l’image et évitez les descriptions vides ou redondantes.
Tâche et Normes de Sortie
Rédigez un résumé standard en français à partir du contenu visible de l’image. Adaptez la structure selon la densité d’informations de l’image :
Mode 1 : Images à haute densité d’informations
Concerne les images comportant de nombreux points clés, données, étapes, opinions ou règles. Utilisez un format Markdown structuré :
1. Introduction : 1 à 2 phrases concises pour synthétiser le thème principal de l’image (obligatoire).
2. Contenu principal : Présentez les informations clés sous forme de liste ordonnée ou non ordonnée.
3. Règle de liste : Chaque élément de liste doit commencer par une courte sous-titre en gras, suivi d’une explication concise et objective strictement basée sur le contenu de l’image.
Mode 2 : Images simples et à faible densité
Concerne les images à thème unique ou au contenu simple. Utilisez un format paragraphe naturel sans formatage Markdown superflu, et restituez précisément l’unique information essentielle.
Règles Complémentaires
1. Si l’image contient très peu d’informations, produisez le résumé le plus précis et le plus concis possible.
2. Les symboles Markdown servent uniquement à la mise en page et ne sont pas comptabilisés dans la limite de mots.
3. Le contenu doit être logique, objectif et épuré de toute redondance."""

PROMPT_TEMPLATE_ru = """<image>
Профессиональная роль
Вы являетесь профессиональным экспертом по пониманию изображений и текста, а также составлению резюме. Вы специализируетесь на извлечении ключевой информации с изображений и создании точных, кратких и структурированных резюме на русском языке.
Обязательные правила (высший приоритет)
Все результаты должны строго соответствовать следующим требованиям. Нарушение любого правила делает ответ недействительным:
1. Языковое правило: Выводите текст только на русском языке, использование других языков запрещено.
2. Правило достоверности: Используйте только распознаваемое содержимое изображения. Исключены субъективные предположения, внешние знания и вымышленная информация.
3. Правило длины: Общая длина резюме не должна превышать 500 русских слов.
4. Правило качества: Извлекайте только основную информацию. Не копируйте механически содержимое изображения и избегайте лишних или пустых описаний.
Задача и стандарты вывода
Составьте стандартное резюме на основе видимого содержимого изображения. Адаптируйте структуру вывода в зависимости от насыщенности информации на изображении:
Режим 1: Изображения с высокой плотностью информации
Подходит для изображений с множеством ключевых моментов, данных, шагов, мнений или правил. Используйте структурированный формат Markdown:
1. Вступление: 1–2 кратких предложения для описания основной темы изображения (обязательно).
2. Основное содержимое: Представьте ключевую информацию в виде нумерованного или маркированного списка.
3. Правило оформления списка: Каждый пункт списка должен начинаться с короткого жирного подзаголовка, за которым следует краткое объективное описание строго по содержимому изображения.
Режим 2: Простые изображения с низкой плотностью информации
Подходит для изображений с единственной темой или простым содержимым. Используйте обычный текстовый абзац без лишнего форматирования Markdown, точно передавайте единственную основную информацию.
Дополнительные правила
1. Если на изображении очень мало информации, составьте максимально точное и краткое резюме.
2. Символы Markdown используются только для оформления и не учитываются при подсчете длины текста.
3. Содержимое должно быть логичным, объективным и лаконичным."""


PROMPT_TEMPLATES = {
    "en": PROMPT_TEMPLATE_en,
    "fr": PROMPT_TEMPLATE_fr,
    "ru": PROMPT_TEMPLATE_ru,
    "zh": PROMPT_TEMPLATE_zh,
}

# ──────────────────────────────────────────────────────────────────────────────
# parquet schema（固定，避免跨行类型推断漂移致 verl 读取崩溃）
# ──────────────────────────────────────────────────────────────────────────────
SCHEMA = pa.schema(
    [
        pa.field(
            "prompt",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        pa.field("images", pa.list_(pa.struct([("image", pa.string())]))),
        pa.field("teacher_images", pa.list_(pa.struct([("image", pa.string())]))),
        pa.field(
            "extra_info",
            pa.struct([("index", pa.int64()), ("task_family", pa.string())]),
        ),
    ]
)


def lr_path(orig: str) -> str:
    """原图路径 -> 半分辨率图路径：en_image/x.jpg -> en_image_lr/x.jpg（替换首个 _image/）"""
    return re.sub(r"_image/", "_image_lr/", orig, count=1)


def detect_lang(image_path: str) -> str | None:
    """从图片路径检测语种：/data4/.../train/fr_image/x.jpg -> fr"""
    m = LANG_PATH_RE.search(image_path)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="flash_note summary -> RP-OPSD verl parquet"
    )
    parser.add_argument("--src", default=SRC_DEFAULT, help="summary aligned jsonl 路径")
    parser.add_argument("--out", required=True, help="输出 parquet 路径")
    parser.add_argument(
        "--max-per-lang",
        type=int,
        default=22000,
        help="每门语种最多保留条数（默认 22000，超量随机采样）",
    )
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument(
        "--check-lr", action="store_true", help="校验半分辨率图文件存在并报告缺失"
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="打印内嵌的提示词模板后退出"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fr", "ru", "zh"],
        help="配合 --show-prompt 只看指定语种",
    )
    args = parser.parse_args()

    if args.show_prompt:
        if args.lang:
            print(PROMPT_TEMPLATES[args.lang])
        else:
            for lang in ["en", "fr", "ru", "zh"]:
                print(f"\n{'=' * 70}\n[{lang}]\n{'=' * 70}")
                print(PROMPT_TEMPLATES[lang])
        return

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读 + 按图片路径检测语种分组（不再用 jsonl 的 messages/content 判语种）
    by_lang: dict[str, list[dict]] = {}
    bad = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            images = d.get("images")
            if not images or not isinstance(images, list) or not images:
                bad += 1
                continue
            lang = detect_lang(images[0])
            if lang is None:
                bad += 1
                continue
            by_lang.setdefault(lang, []).append(d)

    print("[read] 按语种分布（采样前）:")
    for k in sorted(by_lang):
        print(f"  {k}: {len(by_lang[k])}")
    if bad:
        print(f"  [warn] 跳过异常样本（无图或路径无法识别语种）: {bad}")

    # 2) 每门语种封顶 --max-per-lang（超量随机采样，不足则全量）
    for lang in list(by_lang):
        if len(by_lang[lang]) > args.max_per_lang:
            rng = random.Random(args.seed)
            by_lang[lang] = rng.sample(by_lang[lang], args.max_per_lang)
            print(f"[sample] {lang} 采样到 {args.max_per_lang}")

    # 3) 组 parquet 行：师生同 prompt（按语种套翻译模板），student 图=半分辨率，teacher 图=原图
    out_rows: list[dict] = []
    missing_lr: list[str] = []
    idx = 0
    for lang in sorted(by_lang):
        template = PROMPT_TEMPLATES[lang]  # 各语种对应翻译提示词
        for d in by_lang[lang]:
            orig = d["images"][0]
            lr = lr_path(orig)
            if args.check_lr and not Path(lr).exists():
                missing_lr.append(lr)
            out_rows.append(
                {
                    "prompt": [{"role": "user", "content": template}],
                    "images": [{"image": lr}],
                    "teacher_images": [{"image": orig}],
                    "extra_info": {
                        "index": idx,
                        "task_family": f"summary_{lang}",
                    },
                }
            )
            idx += 1

    if args.check_lr and missing_lr:
        print(f"[warn] 半分辨率图缺失 {len(missing_lr)} 张（先跑 gen_lr_images.py）:")
        for p in missing_lr[:10]:
            print(f"  {p}")

    # 4) 写 parquet
    table = pa.Table.from_pylist(out_rows, schema=SCHEMA)
    pq.write_table(table, str(out), compression="zstd")

    # 5) 统计
    final_lang = Counter(r["extra_info"]["task_family"] for r in out_rows)
    print(f"[done] 写出 {len(out_rows)} 行 -> {out}")
    print("[stats] 采样后语种分布:")
    for k, v in sorted(final_lang.items()):
        print(f"  {k}: {v}")
    print("[stats] 师生路径示例:")
    print(f"  student(images):  {out_rows[0]['images'][0]['image']}")
    print(f"  teacher(orig):    {out_rows[0]['teacher_images'][0]['image']}")


if __name__ == "__main__":
    main()
