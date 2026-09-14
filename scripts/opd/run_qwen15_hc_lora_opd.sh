#!/usr/bin/env bash
# LoRA on-policy distillation: HC-SMoE Qwen1.5-MoE student, original 60-expert teacher.
# Stack: verl FSDP + PEFT LoRA + teacher inference pool. HF checkpoints, no Megatron convert.
# Does NOT pkill python/ray — this is a shared cluster.
# Do NOT pass DFLASH / verl_composed_dflash_student flags.
#
# Optional:
#   STUDENT_HF / TEACHER_HF
#   GPUS=0,1,2,3
#   SMOKE=1
#   ROLLOUT_NAME=sglang|vllm
#   ROLLOUT_MODE=sync|async
#   FREE_CACHE_ENGINE=False
#   LORA_RANK=32 TARGET_MODULES=all-linear
#   VERL_ROOT / VERL_PYTHON
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
resolve_verl

# Do not join someone else's Ray cluster. Do not pkill.
unset RAY_ADDRESS || true

STUDENT_HF="${STUDENT_HF:-$(default_student_hf)}"
TEACHER_HF="${TEACHER_HF:-$(default_teacher_hf)}"
SAVE_DIR="${SAVE_DIR:-${OPD_ARTIFACTS}/hc_smoe_lora_opd}"
PROMPT_FILE="${PROMPT_FILE:-${OPD_ARTIFACTS}/prompts.parquet}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
N_GPUS="${#GPU_ARR[@]}"
if [[ "${N_GPUS}" -lt 2 ]]; then
  echo "Need at least 2 GPUs (student + teacher). Got GPUS=${GPUS}" >&2
  exit 1
fi

if [[ ! -f "${STUDENT_HF}/config.json" ]]; then
  echo "Student HF checkpoint not found: ${STUDENT_HF}" >&2
  echo "Run HC-SMoE merge first or set STUDENT_HF." >&2
  exit 1
fi
if [[ ! -f "${TEACHER_HF}/config.json" ]]; then
  echo "Teacher HF checkpoint not found: ${TEACHER_HF}" >&2
  echo "The original hub snapshot is gone. Restore a local teacher HF, then rerun:" >&2
  echo "  bash ${SCRIPT_DIR}/convert_megatron_to_hf.sh \\" >&2
  echo "    ${OPD_ARTIFACTS}/qwen15_moe_teacher_torch_dist/release \\" >&2
  echo "    ${OPD_ARTIFACTS}/qwen15_moe_teacher_hf \\" >&2
  echo "    ${STUDENT_HF}" >&2
  echo "Or download and set TEACHER_HF / ALLOW_HF_DOWNLOAD=1 (uses Qwen/Qwen1.5-MoE-A2.7B-Chat)." >&2
  if [[ "${ALLOW_HF_DOWNLOAD:-0}" != "1" ]]; then
    exit 1
  fi
fi

ROLLOUT_NAME="${ROLLOUT_NAME:-$(detect_rollout_backend)}"
if [[ -z "${ROLLOUT_NAME}" ]]; then
  echo "Neither vLLM nor SGLang is importable in ${VERL_PYTHON}." >&2
  echo "This machine: fastrl env has sglang. Try:" >&2
  echo "  VERL_PYTHON=${CONDA_ROOT}/envs/fastrl/bin/python $0" >&2
  exit 1
fi

# Student pool + teacher pool must equal visible GPUs.
if [[ -z "${STUDENT_GPUS:-}" || -z "${TEACHER_GPUS:-}" ]]; then
  if [[ "${N_GPUS}" -ge 8 ]]; then
    STUDENT_GPUS=6
    TEACHER_GPUS=2
  elif [[ "${N_GPUS}" -ge 4 ]]; then
    STUDENT_GPUS=2
    TEACHER_GPUS=$((N_GPUS - STUDENT_GPUS))
  else
    STUDENT_GPUS=$((N_GPUS - 1))
    TEACHER_GPUS=1
  fi
fi
if [[ $((STUDENT_GPUS + TEACHER_GPUS)) -ne "${N_GPUS}" ]]; then
  echo "STUDENT_GPUS (${STUDENT_GPUS}) + TEACHER_GPUS (${TEACHER_GPUS}) != N_GPUS (${N_GPUS})" >&2
  exit 1
fi

TEACHER_TP="${TEACHER_TP:-}"
if [[ -z "${TEACHER_TP}" ]]; then
  if [[ "${TEACHER_GPUS}" -ge 2 ]]; then
    TEACHER_TP=2
  else
    TEACHER_TP=1
  fi
fi
if (( TEACHER_GPUS % TEACHER_TP != 0 )); then
  echo "TEACHER_GPUS=${TEACHER_GPUS} must be divisible by TEACHER_TP=${TEACHER_TP}" >&2
  exit 1
fi
if [[ -z "${ROLLOUT_TP:-}" ]]; then
  if (( STUDENT_GPUS >= 2 )); then
    ROLLOUT_TP=2
  else
    ROLLOUT_TP=1
  fi
fi
if (( STUDENT_GPUS % ROLLOUT_TP != 0 )); then
  echo "STUDENT_GPUS=${STUDENT_GPUS} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}" >&2
  exit 1
fi
# AgentLoopManager initializes only the rollout-server ranks.  With a colocated
# FSDP actor, vLLM sleep/wake and dynamic LoRA loading can then make the other
# ranks enter a model all-gather before every rollout rank is ready.  The
# resulting mismatched collective eventually trips the NCCL watchdog.
#
# This job is single-turn OPD and does not need the async agent loop, so keep
# startup and weight synchronization collective across the WorkerGroup.  Also
# keep the vLLM cache engine resident to avoid the problematic sleep/wake path.
ROLLOUT_MODE="${ROLLOUT_MODE:-sync}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-False}"

SMOKE="${SMOKE:-0}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# all-linear hits attention + expert gate/up/down_proj (and the router). For experts only:
#   TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
LR="${LR:-3e-5}"
LOSS_MODE="${DISTILLATION_LOSS_MODE:-k1}"
USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-True}"

if [[ "${SMOKE}" == "1" ]]; then
  MAX_PROMPT=256
  MAX_RESP=256
  TRAIN_BS=4
  MICRO_BS=1
  TOTAL_STEPS=2
  SAVE_FREQ=1
  EPOCHS=1
  # Prefer a boring initialization path for the connectivity smoke.  These can
  # be explicitly re-enabled to isolate an offload/layered-summon regression.
  PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
  OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
  LAYERED_SUMMON="${LAYERED_SUMMON:-False}"
else
  MAX_PROMPT="${MAX_PROMPT:-512}"
  MAX_RESP="${MAX_RESP:-1024}"
  TRAIN_BS="${TRAIN_BS:-8}"
  MICRO_BS="${MICRO_BS:-1}"
  TOTAL_STEPS="${TOTAL_STEPS:-}"
  SAVE_FREQ="${SAVE_FREQ:-10}"
  EPOCHS="${EPOCHS:-2}"
  PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
  OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
  LAYERED_SUMMON="${LAYERED_SUMMON:-True}"
fi
MAX_NUM_TOKENS=$((MAX_PROMPT + MAX_RESP + 1))
STUDENT_MAX_TOKEN_LEN_PER_GPU="${STUDENT_MAX_TOKEN_LEN_PER_GPU:-$((MICRO_BS * (MAX_PROMPT + MAX_RESP)))}"

mkdir -p "${OPD_ARTIFACTS}" "${SAVE_DIR}"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  "${VERL_PYTHON}" "${SCRIPT_DIR}/prepare_prompts.py" --output "${PROMPT_FILE}" --repeat 8 --format parquet
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/wandb_offline}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "${TARGET_MODULES}" == *","* ]]; then
  TARGET_MODULES_ARG="[${TARGET_MODULES}]"
else
  TARGET_MODULES_ARG="${TARGET_MODULES}"
fi

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="${PROMPT_FILE}"
  data.val_files="${PROMPT_FILE}"
  data.train_batch_size="${TRAIN_BS}"
  data.max_prompt_length="${MAX_PROMPT}"
  data.max_response_length="${MAX_RESP}"
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=True
)

MODEL=(
  actor_rollout_ref.model.path="${STUDENT_HF}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
  "actor_rollout_ref.model.target_modules=${TARGET_MODULES_ARG}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff=0
)

ACTOR=(
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BS}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}"
)

ROLLOUT=(
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM:-0.35}"
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-${TRAIN_BS}}"
  actor_rollout_ref.rollout.temperature=0.8
  actor_rollout_ref.rollout.max_model_len="${MAX_NUM_TOKENS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_TOKENS}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.layered_summon="${LAYERED_SUMMON}"
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER:-True}"
)

DISTILLATION=(
  distillation.enabled=True
  distillation.n_gpus_per_node="${TEACHER_GPUS}"
  distillation.nnodes=1
  distillation.teacher_models.teacher_model.model_path="${TEACHER_HF}"
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}"
  distillation.teacher_models.teacher_model.inference.name="${ROLLOUT_NAME}"
  distillation.teacher_models.teacher_model.inference.temperature=1.0
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM:-0.45}"
  distillation.teacher_models.teacher_model.inference.max_model_len="${MAX_NUM_TOKENS}"
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_TOKENS}"
  distillation.teacher_models.teacher_model.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS:-${TRAIN_BS}}"
  distillation.teacher_models.teacher_model.inference.enforce_eager="${ENFORCE_EAGER:-True}"
  distillation.distillation_loss.loss_mode="${LOSS_MODE}"
  distillation.distillation_loss.topk=64
  distillation.distillation_loss.use_task_rewards=False
  distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}"
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
)

TRAINER=(
  trainer.logger='["console"]'
  trainer.project_name=moe_prune_lora_opd
  trainer.experiment_name=qwen15_hc_lora_opd
  trainer.default_local_dir="${SAVE_DIR}"
  trainer.n_gpus_per_node="${STUDENT_GPUS}"
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.test_freq=-1
  trainer.save_freq="${SAVE_FREQ}"
  trainer.total_epochs="${EPOCHS}"
  trainer.resume_mode=disable
  trainer.critic_warmup=0
)
if [[ -n "${TOTAL_STEPS}" ]]; then
  TRAINER+=(trainer.total_training_steps="${TOTAL_STEPS}")
fi

echo "============================================================"
echo "HC-SMoE LoRA-OPD via verl (FSDP + PEFT)"
echo "  verl      ${VERL_ROOT}"
echo "  python    ${VERL_PYTHON}"
echo "  rollout   ${ROLLOUT_NAME} mode=${ROLLOUT_MODE} tp=${ROLLOUT_TP} free_cache_engine=${FREE_CACHE_ENGINE}"
echo "  fsdp      param_offload=${PARAM_OFFLOAD} optimizer_offload=${OPTIMIZER_OFFLOAD} layered_summon=${LAYERED_SUMMON}"
echo "  student   ${STUDENT_HF}"
echo "  teacher   ${TEACHER_HF}"
echo "  gpus      ${GPUS}  student=${STUDENT_GPUS} teacher=${TEACHER_GPUS} teacher_tp=${TEACHER_TP}"
echo "  lora      rank=${LORA_RANK} alpha=${LORA_ALPHA} target=${TARGET_MODULES}"
echo "  save      ${SAVE_DIR}"
echo "============================================================"
echo "After a ckpt appears:"
echo "  bash ${SCRIPT_DIR}/merge_lora_ckpt.sh ${SAVE_DIR}/global_step_N ${OPD_ARTIFACTS}/hc_smoe_lora_opd_hf"
echo "  bash ${ROOT}/scripts/run_qwen15_c4_mc.sh eval 0 ${OPD_ARTIFACTS}/hc_smoe_lora_opd_hf"
echo "============================================================"

cd "${VERL_ROOT}"
"${VERL_PYTHON}" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${DISTILLATION[@]}" \
  "${TRAINER[@]}" \
  "$@"
