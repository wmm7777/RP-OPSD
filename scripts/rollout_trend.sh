#!/bin/bash
# 看一个 rollout 目录各 step 的退化趋势（长度/混杂/复读/超长率）
PY=/data1/meimei.wu/miniforge3/envs/swift/bin/python
S=/data4/wumeimei/flash_note/RP-OPSD/scripts/inspect_rollout.py
R=${1:-/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4/rollouts}
echo "目录: $R"
for st in 1 25 50 100 150 200 250 300 350 400 430; do
  f=$R/$st.jsonl
  [ -f "$f" ] || continue
  echo "### step $st ###"
  $PY "$S" "$f" 0 2>/dev/null | grep -E "字符长|混杂|复读|超长"
done
