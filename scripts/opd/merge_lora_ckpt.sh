#!/usr/bin/env bash
# Convert a verl FSDP LoRA checkpoint into a merged HF dir for lm-eval.
#
# Usage:
#   bash scripts/opd/merge_lora_ckpt.sh artifacts/opd/hc_smoe_lora_opd/global_step_2 \
#       artifacts/opd/hc_smoe_lora_opd_hf
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
resolve_verl

CKPT="${1:?usage: $0 <global_step_N or .../actor> <output_hf_dir>}"
OUT="${2:?usage: $0 <global_step_N or .../actor> <output_hf_dir>}"
STUDENT_HF="${STUDENT_HF:-$(default_student_hf)}"

if [[ -d "${CKPT}/actor" ]]; then
  ACTOR="${CKPT}/actor"
elif [[ -d "${CKPT}" ]]; then
  ACTOR="${CKPT}"
else
  echo "checkpoint dir not found: ${CKPT}" >&2
  exit 1
fi

EXTRACT="${OUT}_extract"
mkdir -p "${EXTRACT}" "${OUT}"
echo "[merge] FSDP actor ${ACTOR} -> ${EXTRACT}"
cd "${VERL_ROOT}"
"${VERL_PYTHON}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR}" \
  --target_dir "${EXTRACT}"

ADAPTER="${EXTRACT}/lora_adapter"
if [[ -f "${ADAPTER}/adapter_config.json" ]]; then
  echo "[merge] PEFT ${ADAPTER} into ${STUDENT_HF} -> ${OUT}"
  "${VERL_PYTHON}" "${SCRIPT_DIR}/merge_lora_for_eval.py" \
    --base "${STUDENT_HF}" \
    --adapter "${ADAPTER}" \
    --output "${OUT}"
else
  echo "[merge] no lora_adapter in extract; copying HF weights to ${OUT}"
  cp -a "${EXTRACT}/." "${OUT}/"
fi

echo "[merge] eval with:"
echo "  bash ${ROOT}/scripts/run_qwen15_c4_mc.sh eval 0 ${OUT}"
