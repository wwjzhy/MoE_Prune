# Shared paths for OPD glue. Source from other scripts in this directory.
# Primary path is verl FSDP + PEFT LoRA. slime/Megatron leftovers stay for full-param convert.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/zju_0038/wenjun/envs/miniconda3}"

if [[ -z "${SLIME_ROOT:-}" ]]; then
  if [[ -d "${ROOT}/slime/tools" ]]; then
    SLIME_ROOT="${ROOT}/slime"
  elif [[ -d "${ROOT}/third_party/slime/tools" ]]; then
    SLIME_ROOT="${ROOT}/third_party/slime"
  fi
fi

if [[ -z "${MEGATRON_LM:-${MEGATRON_ROOT:-}}" ]]; then
  if [[ -d "${ROOT}/Megatron-LM/megatron/training" ]]; then
    MEGATRON_LM="${ROOT}/Megatron-LM"
  elif [[ -d "${ROOT}/third_party/Megatron-LM/megatron/training" ]]; then
    MEGATRON_LM="${ROOT}/third_party/Megatron-LM"
  fi
else
  MEGATRON_LM="${MEGATRON_LM:-${MEGATRON_ROOT}}"
fi
OPD_ARTIFACTS="${OPD_ARTIFACTS:-${ROOT}/artifacts/opd}"
if [[ -z "${OPD_PYTHON:-}" && -x "${ROOT}/.venv_opd/bin/python" ]]; then
  OPD_PYTHON="${ROOT}/.venv_opd/bin/python"
fi
OPD_PYTHON="${OPD_PYTHON:-python}"

if [[ -z "${VERL_ROOT:-}" ]]; then
  if [[ -d "${ROOT}/verl/verl/trainer" ]]; then
    VERL_ROOT="${ROOT}/verl"
  elif [[ -d "/zju_0038/wenjun/speculative/Draft-OPD/verl/verl/trainer" ]]; then
    # Has distillation.enabled + FSDP LoRA. Do not use its DFLASH example scripts.
    VERL_ROOT="/zju_0038/wenjun/speculative/Draft-OPD/verl"
  fi
fi

_first_existing_python() {
  local py
  for py in "$@"; do
    if [[ -x "${py}" ]]; then
      echo "${py}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${VERL_PYTHON:-}" ]]; then
  # fastrl has ray + peft + sglang (needed to actually launch). draft-opd matches
  # Draft-OPD/verl but has neither vLLM nor SGLang.
  VERL_PYTHON="$(_first_existing_python \
    "${CONDA_ROOT}/envs/fastrl/bin/python" \
    "${CONDA_ROOT}/envs/draft-opd/bin/python" \
    "${ROOT}/.venv/bin/python" \
    )"
  VERL_PYTHON="${VERL_PYTHON:-python}"
fi

resolve_slime() {
  if [[ -z "${SLIME_ROOT:-}" || ! -f "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" ]]; then
    echo "slime not found under ${ROOT}/slime. Copy or clone it there." >&2
    echo "  rsync -a /zju_0038/wenjun/OnlineQAT/slime/ ${ROOT}/slime/" >&2
    echo "  # or: git clone https://github.com/THUDM/slime.git ${ROOT}/slime" >&2
    return 1
  fi
  if [[ -z "${MEGATRON_LM:-}" || ! -d "${MEGATRON_LM}" ]]; then
    echo "Megatron-LM not found under ${ROOT}/Megatron-LM. Set MEGATRON_LM or copy the checkout." >&2
    return 1
  fi
  export PYTHONPATH="${MEGATRON_LM}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
}

python_has_mod() {
  local py="$1" mod="$2"
  "${py}" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('${mod}') else 1)"
}

detect_rollout_backend() {
  if python_has_mod "${VERL_PYTHON}" vllm; then
    echo vllm
  elif python_has_mod "${VERL_PYTHON}" sglang; then
    echo sglang
  else
    echo ""
  fi
}

resolve_verl() {
  if [[ -z "${VERL_ROOT:-}" || ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]]; then
    echo "verl not found. Set VERL_ROOT to a checkout that has distillation.enabled" >&2
    echo "  (this machine: /zju_0038/wenjun/speculative/Draft-OPD/verl)" >&2
    return 1
  fi
  if ! grep -q "distillation.enabled" "${VERL_ROOT}/verl/trainer/config/ppo_trainer.yaml" 2>/dev/null \
    && ! grep -q "distillation@" "${VERL_ROOT}/verl/trainer/config/ppo_trainer.yaml" 2>/dev/null; then
    echo "VERL_ROOT=${VERL_ROOT} has no distillation config. Use Draft-OPD/verl, not fastrl's tree." >&2
    return 1
  fi
  export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

default_teacher_hf() {
  local hub="/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
  local snap cfg
  for snap in \
    "${hub}/snapshots/ec052fda178e241c7c443468d2fa1db6618996be" \
    "${OPD_ARTIFACTS}/qwen15_moe_teacher_hf"
  do
    if [[ -f "${snap}/config.json" ]]; then
      echo "${snap}"
      return 0
    fi
  done
  if [[ -d "${hub}/snapshots" ]]; then
    cfg="$(find "${hub}/snapshots" -mindepth 2 -maxdepth 2 -name config.json 2>/dev/null | head -n 1)"
    if [[ -n "${cfg}" ]]; then
      dirname "${cfg}"
      return 0
    fi
  fi
  echo "Qwen/Qwen1.5-MoE-A2.7B-Chat"
}

default_student_hf() {
  echo "${ROOT}/artifacts/Qwen1.5-MoE-A2.7B-Chat/c4/merged_models/hc_smoe-seed_42_0.50/hc_smoe"
}

is_torch_dist() {
  [[ -f "${1}/latest_checkpointed_iteration.txt" ]]
}
