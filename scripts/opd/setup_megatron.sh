#!/usr/bin/env bash
# Clone slime-pinned Megatron-LM, apply slime patches, make a convert venv.
# Full SGLang/Ray training env is NOT installed here (needs slime Docker).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEGATRON_LM="${MEGATRON_LM:-${ROOT}/Megatron-LM}"
SLIME_ROOT="${SLIME_ROOT:-${ROOT}/slime}"
COMMIT="${MEGATRON_COMMIT:-1dcf0dafa884ad52ffb243625717a3471643e087}"
PATCH_DIR="${SLIME_ROOT}/docker/patch/v0.5.15.post1"
VENV="${ROOT}/.venv_opd"
BASE_PY="${BASE_PY:-/zju_0038/wenjun/envs/miniconda3/envs/reasoningqat/bin/python}"

if [[ ! -d "${MEGATRON_LM}/.git" ]]; then
  git clone --filter=blob:none --single-branch https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_LM}"
fi
git -C "${MEGATRON_LM}" fetch --depth 1 origin "${COMMIT}"
git -C "${MEGATRON_LM}" checkout "${COMMIT}"

cd "${MEGATRON_LM}"
for patch_name in megatron.patch megatron-sglang-aligned.patch; do
  patch_path="${PATCH_DIR}/${patch_name}"
  [[ -f "${patch_path}" ]] || continue
  if git apply --reverse --check "${patch_path}" >/dev/null 2>&1; then
    echo "[setup] ${patch_name} already applied"
  else
    git apply "${patch_path}" --3way
    echo "[setup] applied ${patch_name}"
  fi
done

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE_PY}" -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/pip" install -e "${SLIME_ROOT}" --no-deps
"${VENV}/bin/python" -c "import torch,megatron,slime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('megatron', megatron.__file__); print('slime', slime.__file__)"
echo "[setup] MEGATRON_LM=${MEGATRON_LM}"
echo "[setup] OPD_PYTHON=${VENV}/bin/python"
