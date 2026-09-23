#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT}/scripts/opd/run_qwen15_hc_lora_opd.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

mkdir -p "${TMP}/verl/verl/trainer/config" "${TMP}/student" "${TMP}/teacher" \
  "${TMP}/save" "${TMP}/global_step_10/actor"
printf '%s\n' '# distillation.enabled' > "${TMP}/verl/verl/trainer/config/ppo_trainer.yaml"
printf '%s\n' '# fake entrypoint' > "${TMP}/verl/verl/trainer/main_ppo.py"
printf '%s\n' '{}' > "${TMP}/student/config.json"
printf '%s\n' '{}' > "${TMP}/teacher/config.json"
printf '%s\n' 'prompt' > "${TMP}/prompts.parquet"
printf '%s\n' 'fake dataloader state' > "${TMP}/global_step_10/data.pt"
printf '%s\n' '#!/usr/bin/env bash' 'printf "%s\n" "$@" > "${CAPTURE}"' > "${TMP}/python"
chmod +x "${TMP}/python"

COMMON=(
  VERL_ROOT="${TMP}/verl"
  VERL_PYTHON="${TMP}/python"
  STUDENT_HF="${TMP}/student"
  TEACHER_HF="${TMP}/teacher"
  PROMPT_FILE="${TMP}/prompts.parquet"
  SAVE_DIR="${TMP}/save"
  OPD_ARTIFACTS="${TMP}/artifacts"
  ROLLOUT_NAME=vllm
  GPUS=0,1
  STUDENT_GPUS=1
  TEACHER_GPUS=1
  ROLLOUT_TP=1
  TEACHER_TP=1
  SMOKE=1
)

assert_arg() {
  grep -Fxq -- "$2" "$1" || {
    echo "missing argument: $2" >&2
    exit 1
  }
}

env "${COMMON[@]}" CAPTURE="${TMP}/auto.args" RESUME_MODE=auto KEEP_LAST=2 \
  TOTAL_STEPS=20 SAVE_FREQ=10 bash "${LAUNCHER}" >/dev/null
assert_arg "${TMP}/auto.args" 'trainer.resume_mode=auto'
assert_arg "${TMP}/auto.args" 'trainer.resume_from_path=null'
assert_arg "${TMP}/auto.args" 'trainer.max_actor_ckpt_to_keep=2'
assert_arg "${TMP}/auto.args" 'trainer.total_training_steps=20'
assert_arg "${TMP}/auto.args" 'trainer.save_freq=10'

env "${COMMON[@]}" CAPTURE="${TMP}/path.args" \
  RESUME_FROM="${TMP}/global_step_10" bash "${LAUNCHER}" >/dev/null
assert_arg "${TMP}/path.args" 'trainer.resume_mode=resume_path'
assert_arg "${TMP}/path.args" "trainer.resume_from_path=${TMP}/global_step_10"

if env "${COMMON[@]}" CAPTURE="${TMP}/invalid.args" RESUME_MODE=invalid \
  bash "${LAUNCHER}" >/dev/null 2>&1; then
  echo 'invalid RESUME_MODE unexpectedly succeeded' >&2
  exit 1
fi

echo 'resume argument checks passed'
