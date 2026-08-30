#!/usr/bin/env python3
"""用 397B-A17B-FP8 批量生成 gold summary（teacher 看原图生成 gold，student SFT 也看原图训练）

================================================================================
 流程
   读 train.parquet（72k，prompt[4 语种模板] + images[半分辨率] + teacher_images[原图]）
   对每条调 397B vllm API（原图 base64 + prompt 指令，teacher 视图）→ gold summary
   输出 SFT jsonl：messages=[{user: prompt(含<image>占位符)}, {assistant: gold}],
                  images=[原图]，extra_info.index

 设计要点
   - 397B 看原图生成 gold（最优 teacher 视图）
   - SFT student 训练也看原图（与 teacher 同视图，gold 直接可用）
   - prompt 复用 convert_flashnote_summary.py 的 4 语种 PROMPT_TEMPLATE（parquet prompt 列已套好）
   - 断点续传：跳过输出 jsonl 里已有 index 的条目
   - 并发：aiohttp 异步 + semaphore

 用法（397B 服务就绪后，在机器3 跑，调 localhost:8000）
   python gen_gold_summary.py --out .runtime/flashnote_summary/sft_gold_397b.jsonl
   python gen_gold_summary.py --out ... --url http://127.0.0.1:8000 --concurrency 16
================================================================================
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re

import aiohttp
import pyarrow.parquet as pq

PARQUET_DEFAULT = "/data4/wumeimei/flash_note/RP-OPSD/.runtime/flashnote_summary/train.parquet"


def build_api_messages(prompt_content: str, image_path: str) -> list:
    """prompt_content 含 <image> 占位符 + 指令文本；API 调用拆成 image_url + text。"""
    text = re.sub(r"<image>\s*\n?", "", prompt_content, count=1).strip()
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": text},
            ],
        }
    ]


async def gen_one(session, url, model, prompt_content, image_path, idx, sem):
    async with sem:
        payload = {
            "model": model,
            "messages": build_api_messages(prompt_content, image_path),
            "max_tokens": 4096,
            "temperature": 0,
            "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        for attempt in range(3):
            try:
                async with session.post(url + "/v1/chat/completions", json=payload) as r:
                    r.raise_for_status()
                    data = await r.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    return content, reasoning
            except Exception as e:
                if attempt == 2:
                    print(f"[err] idx={idx}: {e}")
                    return None, None
                await asyncio.sleep(2)
        return None, None


async def main_async(args):
    tbl = pq.read_table(args.parquet)
    rows = tbl.to_pylist()
    out_p = __import__("pathlib").Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_p.exists():
        for line in out_p.open(encoding="utf-8"):
            d = json.loads(line)
            done.add(d["extra_info"]["index"])
    shard_ids = (
        {int(x) for x in args.shard_ids.split(",")}
        if getattr(args, "shard_ids", None)
        else {args.shard_id}
    )
    print(f"[info] shard={sorted(shard_ids)}/{args.shard_total} total {len(rows)}, already done {len(done)}, todo {len(rows) - len(done)}")

    sem = asyncio.Semaphore(args.concurrency)
    url = args.url.rstrip("/")
    fo = out_p.open("a", encoding="utf-8")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        tasks = []
        for row in rows:
            idx = row["extra_info"]["index"]
            if args.shard_total > 1 and idx % args.shard_total not in shard_ids:
                continue
            if idx in done:
                continue
            pc = row["prompt"][0]["content"]
            orig = row["teacher_images"][0]["image"]   # 原图，teacher 视图
            lr = row["images"][0]["image"]             # 半分辨率，student SFT 用

            async def run_one(idx=idx, pc=pc, orig=orig, lr=lr):
                gold, reasoning = await gen_one(session, url, args.model, pc, orig, idx, sem)
                if gold:
                    rec = {
                        "messages": [
                            {"role": "user", "content": pc},
                            {"role": "assistant", "content": gold},
                        ],
                        "reasoning": reasoning,
                        "images": [orig],
                        "extra_info": {"index": idx},
                    }
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fo.flush()
                    return 1
                return 0

            tasks.append(run_one())
        results = await asyncio.gather(*tasks)
    fo.close()
    print(f"[done] generated {sum(results)}/{len(tasks)}  -> {out_p}")


def main():
    p = argparse.ArgumentParser(description="397B 批量生成 gold summary")
    p.add_argument("--parquet", default=PARQUET_DEFAULT)
    p.add_argument("--out", required=True, help="输出 SFT jsonl")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="397B vllm 服务地址")
    p.add_argument("--model", default="qwen397b", help="served-model-name")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--shard-id", type=int, default=0, help="分片id(0-based)")
    p.add_argument("--shard-ids", default=None, help="多分片id逗号分隔,如 '0,4,8'; 覆盖 --shard-id")
    p.add_argument("--shard-total", type=int, default=1, help="总分片数")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
