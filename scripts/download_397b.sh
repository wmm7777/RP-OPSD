#!/bin/bash
# 加速下载 Qwen3.5-397B-A17B-FP8（max-workers 16 并发，断点续传）
# modelscope .incomplete 续传不丢已下进度
set -eo pipefail
export PATH=/data1/meimei.wu/.local/bin:/data1/meimei.wu/miniforge3/bin:$PATH

modelscope download \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --local-dir /data4/wumeimei/download_models/Qwen3.5-397B-A17B-FP8 \
  --max-workers 16 \
  2>&1 | tee -a /data4/wumeimei/download_models/dl397.log
