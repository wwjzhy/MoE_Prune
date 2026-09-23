# OPD on compressed MoE (verl LoRA)

实验怎么排、在 8×H20 上跑哪几个：见仓库根目录 **[Exp.md](../../Exp.md)**。现在只批准 `#1` baseline 评测 → `#2` LoRA 通路 smoke → `#3` expert LoRA-OPD。不要一次把 ablation 全排队。

Prune/merge stays REAP. On-policy distillation now uses **verl FSDP + PEFT LoRA** so we can try LoRA-OPD without a Megatron convert. Student is the HC-SMoE Hugging Face checkpoint; teacher is the original 60-expert Qwen1.5-MoE. No DFLASH / draft-model flags.

```
HC-SMoE merge (HF)
  ->  run_qwen15_hc_lora_opd.sh
  ->  merge_lora_ckpt.sh
  ->  8-task MC eval
```

Compare the 8-task average to HC-SMoE **0.5187** (paper dense baseline **0.6008**).

The slime/Megatron scripts in this directory are the old full-param path. Leave them; do not use them for LoRA.

## Layout

| Path | Role |
|------|------|
| `VERL_ROOT` (default: `/zju_0038/wenjun/speculative/Draft-OPD/verl`) | verl with `distillation.enabled` + FSDP LoRA. Do **not** copy its DFLASH example scripts. |
| `run_qwen15_hc_lora_opd.sh` | LoRA-OPD launch: student=HC, teacher=original |
| `prepare_prompts.py` | parquet (verl) or JSONL (slime) |
| `merge_lora_ckpt.sh` | FSDP ckpt → merged HF |
| `merge_lora_for_eval.py` | PEFT `merge_and_unload` |
| `run_qwen15_hc_opd.sh` | leftover slime full-param launch |

`scripts/opd/env.sh` picks `VERL_PYTHON` from the **fastrl** conda env when present (ray + peft + sglang). The **draft-opd** env matches this verl tree but has neither vLLM nor SGLang.

Do not `pkill` python/ray on this cluster.

## Teacher HF

The original hub snapshot (`/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/...`) is gone. Restore a local teacher HF from the already-converted Megatron checkpoint (tokenizer/config from the HC student; weights from the original teacher). This convert needs a free GPU:

```bash
bash scripts/opd/convert_megatron_to_hf.sh \
  artifacts/opd/qwen15_moe_teacher_torch_dist/release \
  artifacts/opd/qwen15_moe_teacher_hf \
  artifacts/Qwen1.5-MoE-A2.7B-Chat/c4/merged_models/hc_smoe-seed_42_0.50/hc_smoe
```

Or set `ALLOW_HF_DOWNLOAD=1` / `TEACHER_HF=Qwen/Qwen1.5-MoE-A2.7B-Chat` to pull from the HF mirror.

## Train (LoRA smoke)

Wait until GPUs are free. Default 4-GPU split: 2 student (FSDP train + colocated rollout) + 2 teacher inference (`TP=2`).

```bash
SMOKE=1 GPUS=0,1,2,3 bash scripts/opd/run_qwen15_hc_lora_opd.sh
```

Useful overrides:

```bash
# 专家侧 LoRA（#3 主实验；不要 all-linear）
TARGET_MODULES=gate_proj,up_proj,down_proj \
LORA_RANK=32 LR=3e-5 \
bash scripts/opd/run_qwen15_hc_lora_opd.sh

# 3 GPUs: 2 student + 1 teacher
GPUS=0,1,2 STUDENT_GPUS=2 TEACHER_GPUS=1 TEACHER_TP=1 \
bash scripts/opd/run_qwen15_hc_lora_opd.sh
```

Recipe: `distillation.enabled=True`, `use_task_rewards=False`, sampled reverse-KL (`k1` + `use_policy_gradient=True`), `load_format=safetensors` for LoRA rollout. Student and teacher stay HF; no `torch_dist` convert.

## Resume

Resume defaults to `auto`: verl reads the latest complete checkpoint tracked under the same `SAVE_DIR`. Use a new `SAVE_DIR` or `RESUME_MODE=disable` for a fresh run.

```bash
# Auto-resume the latest checkpoint in SAVE_DIR.
RESUME_MODE=auto SAVE_DIR=$PWD/artifacts/opd/exp6 \
bash scripts/opd/run_qwen15_hc_lora_opd.sh

# Resume one exact checkpoint. RESUME_FROM implies resume_path mode.
RESUME_FROM=$PWD/artifacts/opd/exp6/global_step_200 \
SAVE_DIR=$PWD/artifacts/opd/exp6 \
bash scripts/opd/run_qwen15_hc_lora_opd.sh
```

Keep model/data/LoRA/batch settings and `TOTAL_STEPS` unchanged across the restart. `KEEP_LAST=2` controls actor checkpoint retention; use `KEEP_LAST=null` to retain all. The native verl checkpoint restores actor/LoRA, optimizer, scheduler, RNG/extra state, global step and dataloader state.

Argument wiring can be checked without GPUs:

```bash
bash scripts/opd/test_lora_opd_resume.sh
```

## Eval

```bash
bash scripts/opd/merge_lora_ckpt.sh \
  artifacts/opd/hc_smoe_lora_opd/global_step_2 \
  artifacts/opd/hc_smoe_lora_opd_hf

bash scripts/run_qwen15_c4_mc.sh eval 0 artifacts/opd/hc_smoe_lora_opd_hf
```

## Env notes

This node currently has no free GPU memory, and neither `draft-opd` nor REAP `.venv` ships vLLM. Launch uses fastrl's SGLang as the rollout/teacher backend against Draft-OPD's verl tree (`PYTHONPATH`). If that combo fails, install vLLM into `draft-opd` and rerun with:

```bash
VERL_PYTHON=/zju_0038/wenjun/envs/miniconda3/envs/draft-opd/bin/python \
ROLLOUT_NAME=vllm \
bash scripts/opd/run_qwen15_hc_lora_opd.sh
```
