#!/usr/bin/env bash
# =============================================================================
# RP-OPSD v5 单文件训练启动脚本（外部 27B 固定 teacher + base 9B student，机器4 m4 本机）
# 与 v4 区别：
#   - student 换回 base 9B（同 v3，未训练）
#   - teacher 从 "legacy(student 初始副本)" 改成 "fixed(外部 Qwen3.8-27B)"
#   - 27B teacher 比 9B student 强很多，teacher 提供更高质量 logit 蒸馏信号
#   - teacher 固定不更新（rate=0.0），消除了 EMA 正反馈退化（同 v3-no-ema）
# 组合动机：v3-no-ema(base 9B + 固定 9B teacher) 已证明稳定收敛；
#   v5 把 teacher 从 9B 升级到 27B = 更强的蒸馏信号 + 仍保持固定不退化
# 架构兼容：Qwen3.8-27B 和 Qwen3.5-9B 都是 Qwen3_5ForConditionalGeneration，
#   image_token_id 248056 / eos_token_id 248044 一致，vocab 一致，reverse-KL top-k 可算
# 长度口径: max_prompt(3072) + max_response(2048) = 5120 (5k)
# 启动: bash scripts/run_rp_opsd_v5_teacher27B.sh
# 日志: tee 到 $OUTPUT_DIR/logs/train.log
# =============================================================================

# ---------- 路径与运行身份 ----------
PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
MODEL_PATH="/data4/wumeimei/download_models/Qwen3.5-9B"
TEACHER_MODEL_PATH="/data4/wumeimei/download_models/Qwen3.8-27B"
CONDA_ENV="verl_opd_flashnote"
CONDA_SH="/data1/meimei.wu/miniforge3/etc/profile.d/conda.sh"
OUTPUT_DIR="$PROJECT_ROOT/outputs/flashnote_train_v5_teacher27B"
TASK_TRAIN_FILE="$PROJECT_ROOT/.runtime/flashnote_summary/train.parquet"
CUSTOM_CHAT_TEMPLATE_FILE="$PROJECT_ROOT/chat_templates/perception_chat_template_qwen35.jinja"

# ---------- 实验元数据 ----------
EXPERIMENT_NAME="RP-OPSD-Qwen3.5-9B-v5-teacher27B-no-ema"
PROJECT_NAME="RP-OPSD"
CONFIG_NAME="vopd"

# ---------- 长度字段（改一个必须全链复查） ----------
MAX_PROMPT_LENGTH=3072
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))   # 5120
PPO_MAX_TOKEN_LEN_PER_GPU=$MAX_MODEL_LEN                      # 必须等于 MAX_MODEL_LEN

# ---------- 训练批次 ----------
TRAIN_BATCH_SIZE=96
PPO_MIMI_BATCH_SIZE=96
PPO_MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_N=8
DATA_DATALOADER_NUM_WORKERS=8
TRAINER_N_GPUS_PER_NODE=8
WORLD_SIZE="${WORLD_SIZE:-1}"
TRAINER_NNODES=$WORLD_SIZE

# ---------- 学习率 / 训练步数 ----------
LR=2e-6
LR_WARMUP_STEPS=75
TRAINER_TOTAL_EPOCHS=2
TRAINER_SAVE_FREQ=150
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=10
TRAINER_LOGGER='["console","tensorboard","wandb"]'

# ---------- Actor / Rollout 内存与策略 ----------
ACTOR_USE_DYNAMIC_BSZ=True
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_AGENT_NUM_WORKERS=8

# ---------- 自蒸馏 / Teacher ----------
# ★ v5: teacher_model_source 从 legacy 改成 fixed（外部 27B teacher）
#   + teacher_model_path 指定 Qwen3.8-27B
#   + regularization=ema + rate=0.0 = teacher 永不更新（fixed source 下双保险）
TEACHER_MODEL_SOURCE="fixed"
TEACHER_MODEL_PATH="/data4/wumeimei/download_models/Qwen3.8-27B"
TEACHER_REGULARIZATION="ema"   # 框架不支持 "none"；fixed source + ema + rate=0 = teacher 永不更新
TEACHER_UPDATE_RATE=0.0
ALPHA=1.0
DONT_REPROMPT_ON_SELF_SUCCESS=True
DISTILLATION_TOPK=100
# ---------- 蒸馏目标（同 v3/v3-no-ema/v4）----------
DISTILLATION_OBJECTIVE="mopd_topk_reverse_kl"   # 论文 Eq.(5) bias-corrected reverse-KL
DISTILLATION_TOPK_SOURCE="teacher"               # teacher 选 top-k support
DISTILLATION_ADD_TAIL=False                       # no tail bucket, no renormalization

# ---------- 检查点 / 输出 ----------
TRAINER_DEFAULT_LOCAL_DIR="$OUTPUT_DIR/checkpoints"
TRAINER_ROLLOUT_DATA_DIR="$OUTPUT_DIR/rollouts"

# ---------- 环境变量 ----------
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -c 0

# ---------- wandb 配置（API key 不硬编码，从 shell env 继承）----------
# 启动前在 shell 里 export WANDB_API_KEY=xxx 再 bash 脚本
# wandb project=trainer.project_name（RP-OPSD）, run=trainer.experiment_name（v5-teacher27B-no-ema）
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"   # offline=本地存，不传外网；online=实时上传

# ---------- m4 本机配置（/dev/shm tmpfs 965G）----------
# 坑1: TMPDIR=/tmp 根盘 99G/70G 74% 易爆 → /dev/shm tmpfs 965G（记忆 verl-tmpdir-shm）
# 坑2: MASTER_PORT 29504 被 v4 用 → 换 29505
# 坑3: triton cache 冷启动竞态 → 本地化（首 step 会慢，后续正常）
export TMPDIR="/dev/shm/rp_opsd_v5"
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export HF_HOME="$TMPDIR/hf_home"
export MASTER_PORT=29505
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$HF_HOME"

# ---------- 初始化 ----------
mkdir -p "$TRAINER_DEFAULT_LOCAL_DIR" "$TRAINER_ROLLOUT_DATA_DIR" "$OUTPUT_DIR/logs"
[[ -f "$CUSTOM_CHAT_TEMPLATE_FILE" ]] || { echo "chat template not found: $CUSTOM_CHAT_TEMPLATE_FILE" >&2; exit 1; }
[[ -f "$TASK_TRAIN_FILE" ]] || { echo "train parquet not found: $TASK_TRAIN_FILE" >&2; exit 1; }
[[ -d "$TEACHER_MODEL_PATH" ]] || { echo "teacher model not found: $TEACHER_MODEL_PATH" >&2; exit 1; }

# ---------- 激活 conda env ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
echo "[run_rp_opsd_v5] experiment=$EXPERIMENT_NAME"
echo "[run_rp_opsd_v5] student=$MODEL_PATH"
echo "[run_rp_opsd_v5] teacher=$TEACHER_MODEL_PATH (fixed, rate=$TEACHER_UPDATE_RATE)"
echo "[run_rp_opsd_v5] max_prompt=$MAX_PROMPT_LENGTH max_response=$MAX_RESPONSE_LENGTH max_model_len=$MAX_MODEL_LEN ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
echo "[run_rp_opsd_v5] data=$TASK_TRAIN_FILE"
echo "[run_rp_opsd_v5] output=$OUTPUT_DIR"

# ---------- 启动训练 ----------
python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
    actor_rollout_ref.actor.self_distillation.distillation_objective=$DISTILLATION_OBJECTIVE \
    actor_rollout_ref.actor.self_distillation.distillation_topk_source=$DISTILLATION_TOPK_SOURCE \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=$DISTILLATION_ADD_TAIL \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_model_path=$TEACHER_MODEL_PATH \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    actor_rollout_ref.model.custom_chat_template_file=$CUSTOM_CHAT_TEMPLATE_FILE \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra] \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"
