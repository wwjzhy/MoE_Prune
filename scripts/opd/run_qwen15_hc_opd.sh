#!/usr/bin/env bash
# Leftover full-param OPD via slime/Megatron. Prefer LoRA on verl:
#   bash scripts/opd/run_qwen15_hc_lora_opd.sh
#
# OPD calibration: HC-SMoE Qwen1.5-MoE student, original 60-expert teacher.
# Uses slime (SGLang rollout + Megatron train) from MoE_Prune/slime.
# Does NOT pkill python/ray — this is a shared cluster.
#
# Required env:
#   SLIME_ROOT     slime checkout
#   MEGATRON_LM    Megatron-LM checkout
#
# Optional:
#   STUDENT_HF / TEACHER_HF
#   STUDENT_DIST / TEACHER_DIST
#   GPUS=0,1,2,3
#   SMOKE=1
#   SKIP_CONVERT=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
# shellcheck source=models/qwen1.5-moe-a2.7b.sh
source "${SCRIPT_DIR}/models/qwen1.5-moe-a2.7b.sh"
resolve_slime

if ! "${OPD_PYTHON}" -c "import transformer_engine" >/dev/null 2>&1; then
  filtered=()
  for arg in "${MODEL_ARGS[@]}"; do
    [[ "${arg}" == "--moe-grouped-gemm" ]] && continue
    filtered+=("${arg}")
  done
  MODEL_ARGS=("${filtered[@]}")
  MODEL_ARGS+=(
    --transformer-impl local
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --no-bias-swiglu-fusion
    --no-bias-dropout-fusion
    --no-bias-gelu-fusion
  )
  echo "[opd] TransformerEngine/Apex missing; local impl to match converted torch_dist"
fi

STUDENT_HF="${STUDENT_HF:-$(default_student_hf)}"
TEACHER_HF="${TEACHER_HF:-$(default_teacher_hf)}"
STUDENT_DIST="${STUDENT_DIST:-${OPD_ARTIFACTS}/hc_smoe_torch_dist}"
TEACHER_DIST="${TEACHER_DIST:-${OPD_ARTIFACTS}/qwen15_moe_teacher_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${OPD_ARTIFACTS}/hc_smoe_opd}"
PROMPT_FILE="${PROMPT_FILE:-${OPD_ARTIFACTS}/prompts.jsonl}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
N_GPUS="${#GPU_ARR[@]}"
if [[ "${N_GPUS}" -lt 2 ]]; then
  echo "Need at least 2 GPUs (actor + rollout). Got GPUS=${GPUS}" >&2
  exit 1
fi

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  NUM_ROLLOUT=4
  MAX_RESP=256
  ROLLOUT_BS=4
  N_SAMPLES=2
  SAVE_INTERVAL=2
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
  MAX_RESP="${MAX_RESP:-1024}"
  ROLLOUT_BS="${ROLLOUT_BS:-8}"
  N_SAMPLES="${N_SAMPLES:-2}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
fi
GLOBAL_BS=$((ROLLOUT_BS * N_SAMPLES))

if [[ ! -f "${STUDENT_HF}/config.json" ]]; then
  echo "Student HF checkpoint not found: ${STUDENT_HF}" >&2
  echo "Run HC-SMoE merge first or set STUDENT_HF." >&2
  exit 1
fi

mkdir -p "${OPD_ARTIFACTS}" "${SAVE_DIR}"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  python "${SCRIPT_DIR}/prepare_prompts.py" --output "${PROMPT_FILE}" --repeat 8
fi

if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  if ! is_torch_dist "${TEACHER_DIST}"; then
    echo "[opd] converting teacher ${TEACHER_HF}"
    bash "${SCRIPT_DIR}/convert_hf_to_megatron.sh" "${TEACHER_HF}" "${TEACHER_DIST}"
  fi
  if ! is_torch_dist "${STUDENT_DIST}"; then
    echo "[opd] converting student ${STUDENT_HF}"
    bash "${SCRIPT_DIR}/convert_hf_to_megatron.sh" "${STUDENT_HF}" "${STUDENT_DIST}"
  fi
fi
if ! is_torch_dist "${TEACHER_DIST}" || ! is_torch_dist "${STUDENT_DIST}"; then
  echo "Need torch_dist checkpoints:" >&2
  echo "  teacher ${TEACHER_DIST}" >&2
  echo "  student ${STUDENT_DIST}" >&2
  exit 1
fi

# Megatron OPD loads the teacher on actor GPUs (no extra SGLang teacher).
# Default 4-GPU split: 2 actor (TP=2) + 2 SGLang rollout.
if [[ -z "${ACTOR_GPUS:-}" || -z "${ROLLOUT_GPUS:-}" ]]; then
  if [[ "${N_GPUS}" -ge 4 ]]; then
    ACTOR_GPUS=2
    ROLLOUT_GPUS=$((N_GPUS - 2))
  elif [[ "${N_GPUS}" -eq 3 ]]; then
    ACTOR_GPUS=2
    ROLLOUT_GPUS=1
  else
    ACTOR_GPUS=1
    ROLLOUT_GPUS=1
  fi
fi
TP="${TP:-${ACTOR_GPUS}}"

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_HF}"
   --load "${STUDENT_DIST}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_FILE}"
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BS}"
   --n-samples-per-prompt "${N_SAMPLES}"
   --rollout-max-response-len "${MAX_RESP}"
   --rollout-temperature 0.8
   --global-batch-size "${GLOBAL_BS}"
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type megatron
   --opd-kl-coef 1.0
   --opd-teacher-load "${TEACHER_DIST}"
   --kl-loss-coef 0.00
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

echo "============================================================"
echo "HC-SMoE OPD via slime (megatron teacher)"
echo "  slime     ${SLIME_ROOT}"
echo "  megatron  ${MEGATRON_LM}"
echo "  student   ${STUDENT_HF}"
echo "  teacher   ${TEACHER_HF}"
echo "  gpus      ${GPUS}  actor=${ACTOR_GPUS} tp=${TP} rollout=${ROLLOUT_GPUS}"
echo "  save      ${SAVE_DIR}"
echo "============================================================"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
cd "${SLIME_ROOT}"

if ray status >/dev/null 2>&1; then
  echo "[opd] reusing existing Ray cluster"
else
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${N_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_LM}:${SLIME_ROOT}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

echo "[opd] job submitted. After a ckpt appears:"
echo "  bash ${SCRIPT_DIR}/convert_megatron_to_hf.sh ${SAVE_DIR}/iter_xxx ${OPD_ARTIFACTS}/hc_smoe_opd_hf ${STUDENT_HF}"
echo "  bash ${ROOT}/scripts/run_qwen15_c4_mc.sh eval 0 ${OPD_ARTIFACTS}/hc_smoe_opd_hf"
