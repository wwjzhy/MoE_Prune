#!/bin/bash
# Reproduce HC-SMoE / M-SMoE on Qwen1.5-MoE-A2.7B-Chat
# Setting: C4 calibration (32 x 2048 tokens, matching HC-SMoE paper), 50% merge, 8 MC tasks.
set -euo pipefail
ROOT=/zju_0038/wenjun/Prune/MoE_Prune
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null || true)}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/.venv/bin/python"

MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat
DATASET=allenai/c4
SEED=42
RATIO=0.50
METHOD="${1:?usage: $0 hc_smoe|m_smoe|reap|eval|all [gpu] [model_dir]}"
GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

COMMON_OBS=(
  --dataset-name "$DATASET"
  --model-name "$MODEL"
  --seed "$SEED"
  --batches_per_category 32
  --batch_size 1
  --model_max_length 2048
  --profile false
  --plot_clusters false
  --smoke_test false
  --do-eval false
  --overwrite_observations false
  --output-file-name "observations_32_c4-seed_${SEED}.pt"
)

run_hc() {
  echo "[HC-SMoE] GPU=$GPU"
  "$PY" src/reap/main.py \
    "${COMMON_OBS[@]}" \
    --compression_ratio "$RATIO" \
    --merge-method frequency_weighted_average \
    --expert-sim characteristic_activation \
    --distance_measure euclidean \
    --linkage-method average \
    --frequency-penalty false \
    --cluster_method agglomerative \
    --cluster-description hc_smoe \
    --merged-model-dir-name "hc_smoe-seed_${SEED}_${RATIO}"
}

run_msmoe() {
  echo "[M-SMoE] GPU=$GPU"
  "$PY" src/reap/main.py \
    "${COMMON_OBS[@]}" \
    --compression_ratio "$RATIO" \
    --merge-method frequency_weighted_average \
    --permute wm \
    --cluster-method mc_smoe \
    --expert-sim router_logits \
    --distance_measure cosine \
    --frequency-penalty false \
    --cluster-description m_smoe \
    --merged-model-dir-name "m_smoe-seed_${SEED}_${RATIO}"
}

run_reap() {
  echo "[REAP] GPU=$GPU"
  "$PY" src/reap/prune.py \
    "${COMMON_OBS[@]}" \
    --compression_ratio "$RATIO" \
    --prune-method reap
}

run_eval() {
  local model_dir="$1"
  echo "[lm-eval] $model_dir GPU=$GPU"
  mkdir -p "$model_dir/eval"
  "$PY" src/reap/eval.py \
    --model-name "$model_dir" \
    --use_server false \
    --run-lm-eval true \
    --run-evalplus false \
    --run-livecodebench false \
    --run-math false \
    --run-wildbench false \
    --results_dir "$model_dir/eval" \
    --seed "$SEED"
}

find_merged_dir() {
  local tag="$1"
  local found
  found=$(find "$ROOT/artifacts/Qwen1.5-MoE-A2.7B-Chat/c4" -mindepth 3 -maxdepth 4 -type d -name "$tag" 2>/dev/null | while read -r d; do
    if [[ -f "$d/config.json" ]]; then
      echo "$d"
      break
    fi
  done)
  if [[ -z "${found:-}" ]]; then
    echo "ERROR: could not find merged model dir for $tag" >&2
    find "$ROOT/artifacts/Qwen1.5-MoE-A2.7B-Chat/c4" -name config.json 2>/dev/null || true
    exit 1
  fi
  echo "$found"
}

run_all() {
  echo "[pipeline] HC-SMoE then eval, then M-SMoE then eval. GPU=$GPU"
  run_hc
  local hc_dir
  hc_dir=$(find_merged_dir hc_smoe)
  echo "[pipeline] HC merged model: $hc_dir"
  run_eval "$hc_dir"
  run_msmoe
  local m_dir
  m_dir=$(find_merged_dir m_smoe)
  echo "[pipeline] M-SMoE merged model: $m_dir"
  run_eval "$m_dir"
  echo "[pipeline] done"
}

case "$METHOD" in
  hc_smoe) run_hc ;;
  m_smoe) run_msmoe ;;
  reap) run_reap ;;
  eval) run_eval "${3:?eval requires model_dir}" ;;
  all) run_all ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac
