#!/usr/bin/env bash
# Convert a Qwen1.5-MoE HF checkpoint (original or HC-SMoE merged) to slime torch_dist.
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
  echo "[convert] TransformerEngine/Apex missing; local impl and fusion kernels disabled"
fi

HF_CKPT="${1:?usage: $0 <hf_checkpoint_dir> [save_dir]}"
SAVE_DIR="${2:-${OPD_ARTIFACTS}/$(basename "${HF_CKPT}")_torch_dist}"
NPROC="${NPROC:-1}"
GPU="${GPU:-0}"

if [[ ! -f "${HF_CKPT}/config.json" ]]; then
  echo "HF checkpoint missing config.json: ${HF_CKPT}" >&2
  exit 1
fi
mkdir -p "${SAVE_DIR}"
echo "[convert] ${HF_CKPT} -> ${SAVE_DIR}  nproc=${NPROC} gpu=${GPU} python=${OPD_PYTHON}"
cd "${SLIME_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29551}"
if [[ "${NPROC}" -gt 1 ]]; then
  "${OPD_PYTHON}" -m torch.distributed.run --nproc_per_node="${NPROC}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}"
else
  "${OPD_PYTHON}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}"
fi
echo "[convert] done -> ${SAVE_DIR}"
