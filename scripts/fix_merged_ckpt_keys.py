import json, os, shutil
from safetensors.torch import safe_open, save_file
from safetensors import safe_open as safe_open_ro
import torch

MERGED = "/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/merged_hf/model.safetensors"
BASE_DIR = "/data4/wumeimei/download_models/Qwen3.5-9B"
OUT = "/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v4_fixed_teacher/checkpoints/global_step_150/merged_hf_fixed"
os.makedirs(OUT, exist_ok=True)

# Copy non-safetensors files from merged_hf
for fn in os.listdir(os.path.dirname(MERGED)):
    if fn == "model.safetensors": continue
    src = os.path.join(os.path.dirname(MERGED), fn)
    dst = os.path.join(OUT, fn)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"copied: {fn}")

# Load base index
with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    base_idx = json.load(f)
base_shard_map = base_idx["weight_map"]

# Get base keys needed for visual & mtp (not language_model)
base_needed_keys = [k for k in base_shard_map if k.startswith("model.visual.") or k.startswith("model.mtp.")]
print(f"base needed keys (visual+mtp): {len(base_needed_keys)}")

# Group base needed keys by shard
base_shards_needed = {}
for k in base_needed_keys:
    sh = base_shard_map[k]
    base_shards_needed.setdefault(sh, []).append(k)
print(f"base shards to read: {list(base_shards_needed.keys())}")

# Read merged safetensors keys + metadata
with safe_open(MERGED, framework="pt") as f:
    merged_keys = list(f.keys())
    print(f"merged total keys: {len(merged_keys)}")

# Build rename map
rename_map = {}
for k in merged_keys:
    if k == "lm_head.weight":
        rename_map[k] = k
    elif k.startswith("model.language_model.visual."):
        new_k = "model.visual." + k[len("model.language_model.visual."):]
        rename_map[k] = new_k
    elif k.startswith("model.language_model.language_model.language_model."):
        new_k = "model.language_model." + k[len("model.language_model.language_model.language_model."):]
        rename_map[k] = new_k
    else:
        rename_map[k] = k  # keep as-is

# Verify renamed keys cover what's expected (except mtp)
renamed_set = set(rename_map.values())
mtp_in_renamed = sum(1 for k in renamed_set if k.startswith("model.mtp."))
print(f"mtp keys in renamed merged: {mtp_in_renamed} (expect 0)")
visual_in_renamed = sum(1 for k in renamed_set if k.startswith("model.visual."))
print(f"visual keys in renamed merged: {visual_in_renamed} (base has {len([k for k in base_shard_map if k.startswith('model.visual.')])})")
lm_in_renamed = sum(1 for k in renamed_set if k.startswith("model.language_model."))
print(f"language_model keys in renamed merged: {lm_in_renamed}")

# Process: load merged tensors with renamed keys, add mtp from base, save
# Since merged is 18GB single shard, load all into dict (system RAM should be enough)
print("loading merged tensors...")
state = {}
with safe_open(MERGED, framework="pt") as f:
    for k in merged_keys:
        t = f.get_tensor(k)
        state[rename_map[k]] = t
        del t
print(f"loaded {len(state)} tensors from merged")

# Add mtp tensors from base
print("loading mtp tensors from base...")
mtp_added = 0
for sh, keys in base_shards_needed.items():
    sh_path = os.path.join(BASE_DIR, sh)
    with safe_open(sh_path, framework="pt") as f:
        sh_keys = set(f.keys())
        for k in keys:
            if k not in sh_keys:
                continue
            if k in state:
                # already in merged (just renamed) - skip
                continue
            t = f.get_tensor(k)
            state[k] = t
            mtp_added += 1
            del t
print(f"added {mtp_added} mtp tensors from base (mtp keys: {sum(1 for k in state if k.startswith('model.mtp.'))})")

# Save as single shard
out_path = os.path.join(OUT, "model.safetensors")
print(f"saving {len(state)} tensors to {out_path} ...")
save_file(state, out_path, metadata={"format": "pt"})
del state

# Write index.json
total_keys = list(rename_map.values()) + [k for k in base_needed_keys if k.startswith("model.mtp.")]
total_set = set(total_keys)
index = {
    "metadata": {"total_size": os.path.getsize(out_path)},
    "weight_map": {k: "model.safetensors" for k in total_set}
}
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
print(f"saved index.json with {len(total_set)} keys")
print("DONE")
