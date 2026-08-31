#!/usr/bin/env bash
# =============================================================================
# RP-OPSD v2 单文件训练启动脚本（不嵌套任何其它脚本）
# 所有变量在本文件顶部统一配置，改参数只改这里。
# 长度口径: max_prompt(3072) + max_response(2048) = 5120 (5k)
#   - 数据集实测: 文本 token mean=395 (固定模板) + 图像 token ~1280 ≈ 1700
#   - 3072 对 prompt 是 1.8x 余量；2048 给 summary 足够空间
#   - 5120 比 6144 降 17%，直接压低 actor forward/backward 峰值显存
#     （step 300 seqlen max=120k 时 OOM 的根因修复）
#   - ppo_max_token_len_per_gpu 必须等于 max_model_len，否则 use_dynamic_bsz=True
#     会触发 AssertionError @ seqlen_balancing.py:382
# 启动: bash scripts/run_rp_opsd_v2.sh
# 日志: tee 到 $OUTPUT_DIR/logs/train.log
# =============================================================================

# ---------- 路径与运行身份 ----------
PROJECT_ROOT="/data4/wumeimei/flash_note/RP-OPSD"
MODEL_PATH="/data4/wumeimei/download_models/Qwen3.5-9B"
CONDA_ENV="verl_opd_flashnote"
CONDA_SH="/data1/meimei.wu/miniforge3/etc/profile.d/conda.sh"
OUTPUT_DIR="$PROJECT_ROOT/outputs/flashnote_train_v2"
TASK_TRAIN_FILE="$PROJECT_ROOT/.runtime/flashnote_summary/train.parquet"
CUSTOM_CHAT_TEMPLATE_FILE="$PROJECT_ROOT/chat_templates/perception_chat_template_qwen35.jinja"

# ---------- 实验元数据 ----------
EXPERIMENT_NAME="RP-OPSD-Qwen3.5-9B"
PROJECT_NAME="RP-OPSD"
CONFIG_NAME="vopd"

# ---------- 长度字段（改一个必须全链复查） ----------
# MAX_MODEL_LEN 6144→5120：seqlen 增长后 actor forward/backward 峰值显存超限
# （step 300 seqlen max=120k 时 OOM），压低 ppo_max_token_len_per_gpu 直接降峰值
# prompt 5120→3072：实测 prompt ~1700 token（文本 395 + 图像 ~1280），3072=1.8x 余量
# response 1024→2048：summary 需要更长输出（gen_gold max_tokens=4096，SFT 训练上限）
MAX_PROMPT_LENGTH=3072
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))   # 5120
PPO_MAX_TOKEN_LEN_PER_GPU=$MAX_MODEL_LEN                      # 必须等于 MAX_MODEL_LEN

# ---------- 训练批次 ----------
# ROLLOUT_N 保持 8（不减并发）
TRAIN_BATCH_SIZE=96
PPO_MIMI_BATCH_SIZE=96
PPO_MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_N=8
DATA_DATALOADER_NUM_WORKERS=8
TRAINER_N_GPUS_PER_NODE=8
WORLD_SIZE="${WORLD_SIZE:-1}"
TRAINER_NNODES=$WORLD_SIZE

# ---------- 学习率 / 训练步数 ----------
# LR_WARMUP_STEPS 10→75：72k 数据集 × 2epoch = 1502 步，10 步 warmup 仅 0.7%
#   teacher EMA 半衰期 ~14 步（rate=0.05），75 步 ≈ 5 个半衰期让 teacher 充分收敛
LR=2e-6
LR_WARMUP_STEPS=75
TRAINER_TOTAL_EPOCHS=2
TRAINER_SAVE_FREQ=150
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=10
TRAINER_LOGGER='["console","tensorboard"]'

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
TEACHER_MODEL_SOURCE="legacy"
TEACHER_REGULARIZATION="ema"
TEACHER_UPDATE_RATE=0.05
ALPHA=1.0
DONT_REPROMPT_ON_SELF_SUCCESS=True
DISTILLATION_TOPK=100

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

# ---------- 初始化 ----------
mkdir -p "$TRAINER_DEFAULT_LOCAL_DIR" "$TRAINER_ROLLOUT_DATA_DIR" "$OUTPUT_DIR/logs"
[[ -f "$CUSTOM_CHAT_TEMPLATE_FILE" ]] || { echo "chat template not found: $CUSTOM_CHAT_TEMPLATE_FILE" >&2; exit 1; }
[[ -f "$TASK_TRAIN_FILE" ]] || { echo "train parquet not found: $TASK_TRAIN_FILE" >&2; exit 1; }

# ---------- 激活 conda env ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"
echo "[run_rp_opsd_v2] experiment=$EXPERIMENT_NAME"
echo "[run_rp_opsd_v2] max_prompt=$MAX_PROMPT_LENGTH max_response=$MAX_RESPONSE_LENGTH max_model_len=$MAX_MODEL_LEN ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
echo "[run_rp_opsd_v2] data=$TASK_TRAIN_FILE"
echo "[run_rp_opsd_v2] output=$OUTPUT_DIR"

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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
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
