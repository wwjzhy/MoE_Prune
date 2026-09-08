#!/usr/bin/env bash
# Convert a slime torch_dist iter_* checkpoint back to HuggingFace.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
resolve_slime

INPUT_DIR="${1:?usage: $0 <torch_dist_iter_dir> <output_hf_dir> [origin_hf_dir]}"
OUTPUT_DIR="${2:?usage: $0 <torch_dist_iter_dir> <output_hf_dir> [origin_hf_dir]}"
ORIGIN_HF="${3:-$(default_student_hf)}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "missing input: ${INPUT_DIR}" >&2
  exit 1
fi
cd "${SLIME_ROOT}"
python tools/convert_torch_dist_to_hf.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --origin-hf-dir "${ORIGIN_HF}" \
  --force
echo "[convert] HF written -> ${OUTPUT_DIR}"
echo "Eval: bash ${ROOT}/scripts/run_qwen15_c4_mc.sh eval 0 ${OUTPUT_DIR}"
