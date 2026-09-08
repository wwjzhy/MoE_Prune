# HC-SMoE + LoRA-OPD 实验记录

给**另一个集群上的 GPT / 人**看。本机 4×A800 正在被占，实验在 **8×H20** 上跑。脚本里的 `/zju_0038/...` 是旧机器路径，**新集群必须先 export 下面的路径**。

**写法：** 按时间往下追加。每次新增实验要求，在文末开一节 `Exp #N`，写清「要什么 / 怎么跑 / 产出 / 状态」。**不要改已经写下的旧节。** 跑的时候按 `#1 → #2 → …` 顺序；后面的实验若依赖前面的产物，在该节写明。

**现在只批准跑 #1 → #2 → #3。** `#4` 及以后等 #3 出了 8 任务平均分再追加，不要一次把 ablation 全排队。

公共约定（各 Exp 除非自己改，否则都遵守）：

- Student = HC-SMoE 50% merge 的 HF（仍 60 expert，cluster 内权重拷贝）。Teacher = 原始 `Qwen1.5-MoE-A2.7B-Chat`
- 训练走 **verl FSDP + PEFT LoRA + `distillation.enabled`**。不要 DFLASH / `verl_composed_dflash_student`。不要 slime 全参（除非某节显式要求）
- **不要 `pkill -9 python/ray`**
- 评测：`bash scripts/run_qwen15_c4_mc.sh eval <gpu> <hf_dir>`，8 任务 MC（arc_c / arc_e / boolq / hellaswag / mmlu / openbookqa / piqa / winogrande），报告 **per-task + mc_average**
- 对照：HC 已测 **0.5187**；论文 dense baseline **0.6008**（#1 要在新机器复测 teacher，确认 eval 管线没漂）
- 硬件：8 张 H20。OPD 默认 **6 student + 2 teacher（teacher TP=2）**
- 结果不只交分数：状态里写 wall time、global step、LoRA rank / target_modules、prompt 文件、ckpt 路径。KL 下降不算成功，**只看 mc_average**

---

## 机器准备（不是实验；每台新机器做一次）

| 变量 | 是什么 | 怎么查 |
|------|--------|--------|
| `CONDA_ROOT` | conda 根目录 | `dirname $(dirname $(which conda))` |
| `STUDENT_HF` | HC merge 目录，内有 `config.json` | 从旧机器 rsync，或在新机器重跑 HC |
| `TEACHER_HF` | 原始 60-expert HF | 下 `Qwen/Qwen1.5-MoE-A2.7B-Chat`，或从旧机器 rsync |
| `HF_HOME` | Hugging Face 缓存 | 写成仓库内 `$PWD/hf_cache` |
| `VERL_ROOT` | 带 `distillation.enabled` 的 verl | 推荐 `speculative/Draft-OPD/verl` 或 clone 官方 verl（必须有 OPD） |
| `VERL_PYTHON` | 能 `import ray, peft` 且有 **vLLM 或 SGLang** 的 python | 不要用没有推理引擎的 env |

```bash
cd Prune/MoE_Prune

export CONDA_ROOT=/改成你的conda根目录
export HF_HOME=$PWD/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export STUDENT_HF=/改成你的/hc_smoe
export TEACHER_HF=/改成你的/Qwen1.5-MoE-A2.7B-Chat
export VERL_ROOT=/改成你的/verl
export VERL_PYTHON=/改成你的/python
export GPUS=0,1,2,3,4,5,6,7
export STUDENT_GPUS=6
export TEACHER_GPUS=2
export TEACHER_TP=2

test -f "$STUDENT_HF/config.json" && echo student ok
test -f "$TEACHER_HF/config.json" && echo teacher ok
test -f "$VERL_ROOT/verl/trainer/main_ppo.py" && echo verl ok
nvidia-smi -L
```

旧机器需要拷到新机器的（代码走 git，权重不要进 git）：

| 路径 | 作用 | 没有怎么办 |
|------|------|------------|
| HC `.../merged_models/hc_smoe-seed_42_0.50/hc_smoe` | student | 新机器 `bash scripts/run_qwen15_c4_mc.sh hc_smoe`（C4 32×2048，seed 42，ratio 0.50） |
| 原始 Qwen1.5-MoE HF | teacher | `HF_ENDPOINT=https://hf-mirror.com` 拉 `Qwen/Qwen1.5-MoE-A2.7B-Chat` |
| （可选）`artifacts/opd/qwen15_moe_teacher_torch_dist` | 仅当要从 Megatron 反转 teacher HF | 直接下 HF 更简单 |

H20 上 verl 环境：优先 **vLLM + peft + ray** 装到和 Draft-OPD/官方 verl 匹配的 python。`ROLLOUT_NAME` 有 vLLM 就用 `vllm`，否则 `sglang`。不要混用不带 `distillation` 的 fastrl verl 树当 `VERL_ROOT`。

每开一个新 shell 都要重新 export。缺模型 / GPU 不是 8 张就停下来问人。

---

## Exp #1（2026-09-08）— 新机器复测两条 baseline

**要求：** 不训练。只确认评测管线和权重没漂。必须先做，后面所有「恢复了多少」都相对这组数字。

| | 旧机器（已有） | 本实验要交 |
|--|----------------|------------|
| Teacher 原始 60-expert | 论文 0.6008，本仓库没复测 | **必须测** mc_average |
| HC-SMoE 50% | **0.5187** | 再测一遍，允许 ±0.01；差更多先查 lm-eval / chat template |

**怎么跑：**

```bash
# teacher
bash scripts/run_qwen15_c4_mc.sh eval 0 "$TEACHER_HF"

# HC student
bash scripts/run_qwen15_c4_mc.sh eval 0 "$STUDENT_HF"
```

两路可以各占 1 卡并行（H20 80G+ 够 14B bf16 eval）。不要改 seed、不要 `--use_server`。

**产出：** 各模型目录下 `eval/`；把 8 任务分和 mc_average 写进本节状态。

**依赖：** `STUDENT_HF`、`TEACHER_HF` 都有 `config.json`。

**状态：** 未跑（等 8×H20）。

---

## Exp #2（2026-09-08）— LoRA-OPD 通路 smoke（不看 8 任务）

**要求：** 证明 verl LoRA-OPD 在 H20 上能跑完、能存 ckpt、distill loss / reverse-KL 在降。**不要用这次分数做方法结论。** Prompt 用 `scripts/opd/prepare_prompts.py` 那点 toy 数据就行。步数短。

| | 设定 |
|--|------|
| student / teacher | HC / 原始 |
| LoRA | rank 32, alpha 32, `all-linear` |
| loss | `k1` + `use_policy_gradient=True`，`use_task_rewards=False` |
| 步数 | `SMOKE=1`（脚本内 2 step）若 trop 太短看不出 loss，改 `TOTAL_STEPS=20 SAVE_FREQ=10` |
| 数据 | toy parquet |
| GPU | 6+2 |

**怎么跑：**

```bash
SMOKE=1 \
SAVE_DIR=$PWD/artifacts/opd/exp2_lora_smoke \
PROMPT_FILE=$PWD/artifacts/opd/prompts.parquet \
bash scripts/opd/run_qwen15_hc_lora_opd.sh
```

成功标准：Ray 起来、student rollout、teacher logprob、至少 1 次 optimizer update、`SAVE_DIR/global_step_*` 出现。OOM 则 `STUDENT_GPUS=4 TEACHER_GPUS=4 TEACHER_TP=2` 或把 `MAX_RESP` 降到 128。

**产出：** `artifacts/opd/exp2_lora_smoke/`。本节状态写：是否跑通、OOM 与否、用的 `ROLLOUT_NAME`、wall time。可选 merge 后 eval，但 **mc_average 不作为 pass/fail**。

**依赖：** Exp #1 的权重路径（不必等 #1 评测写完，但必须是同一份 HC ckpt）。

**状态：** 未跑。

---

## Exp #3（2026-09-08）— 主实验：只训 expert LoRA-OPD 恢复 HC

**要求：** 这是唯一要回答「idea 行不行」的实验。**只训 routed expert 的 FFN LoRA**（`gate_proj,up_proj,down_proj`）。**冻住注意力和 router**（不要 `q/k/v/o_proj`，不要 router 的 `gate`）。不要 full-param，不要 `all-linear`。数据不要 toy prompt：用 8 任务的 **train split 题干**（chat user，无 label），禁止用 test/validation 题干。MMLU 没有 train 就跳过 MMLU，或只用其它 7 个任务的 train。

目标不是训回 0.6008（那等于撤销 merge），而是看 mc_average 相对 #1 的 HC **有没有稳定回升**。事先预期：回升缺口的一部分算正信号；持平或掉点则先查数据/步数，不要立刻上 full-param 或加 router。

| | 设定 |
|--|------|
| LoRA | rank 32, **仅** `gate_proj,up_proj,down_proj`（expert FFN；会捎上 shared_expert 同名层，可接受） |
| 冻住 | 注意力、router `gate` |
| LR | `3e-5` |
| 数据 | 8 任务 train 题干 parquet，至少几千条；`filter_overlong_prompts=True` |
| 序列 | **总长 512**：prompt 256 + response 256（`max_model_len≈513`） |
| 步数 | 先 200–500 global step，`SAVE_FREQ=50` |
| GPU | 6+2 |
| 评测 | merge 后对 **最后 ckpt 和中间一份** 做 8 任务；对比 #1 HC |

**怎么跑：**

1. 生成 train-stem parquet（禁用 test/val；跳过 MMLU）：

```bash
python scripts/opd/prepare_mc_train_stems.py \
  --output artifacts/opd/mc_train_stems.parquet
```
2. 训练：

```bash
TARGET_MODULES=gate_proj,up_proj,down_proj \
LORA_RANK=32 LORA_ALPHA=32 LR=3e-5 \
PROMPT_FILE=$PWD/artifacts/opd/mc_train_stems.parquet \
SAVE_DIR=$PWD/artifacts/opd/exp3_hc_expert_lora \
MAX_PROMPT=256 MAX_RESP=256 TRAIN_BS=16 \
TOTAL_STEPS=400 SAVE_FREQ=50 EPOCHS=1 \
bash scripts/opd/run_qwen15_hc_lora_opd.sh
```

3. merge + eval（对 `global_step_400` 和一份中间 step）：

```bash
bash scripts/opd/merge_lora_ckpt.sh \
  artifacts/opd/exp3_hc_expert_lora/global_step_400 \
  artifacts/opd/exp3_hc_expert_lora_hf_400
bash scripts/run_qwen15_c4_mc.sh eval 0 artifacts/opd/exp3_hc_expert_lora_hf_400
```

**产出：** LoRA ckpt、merged HF、8 任务表、相对 #1 HC 的 Δmc_average。

**依赖：** #2 通路成功；#1 的 HC 分数（旧机器 0.5187 可暂用，新机器测完后改用 #1）。#2 仍用 `all-linear` smoke，**不要**把 #2 改成只训 expert。

**状态：** 未跑。未批准在 #2 失败时开跑。

---

## 先不要跑（等 #3 数字）

下面只是排队草稿，**没有 Exp 编号、不要在 H20 上先占队列**。#3 若 mc_average 相对 HC 回升 ≥0.01，再在文末按 OnlineQAT 格式追加正式节。

| 草稿 | 问什么 | 相对 #3 只改 |
|------|--------|--------------|
| A | 损伤是否在 expert | `TARGET_MODULES=q_proj,k_proj,v_proj,o_proj`（注意力 only，预期弱） |
| B | router 要不要加 | #3 的 modules 再加 `gate`（joint） |
| B' | 只改门控的上限 | **仅** `TARGET_MODULES=gate` |
| C | 是否只是 in-domain 刷分 | 数据换成 C4/通用指令，**不用** 8 任务 train |
| D | OPD 是否必要 | 同数据 teacher greedy 轨迹做 offline SFT/KD，对比 #3 |
| E | 结构变化是否还能修 | student 换成 REAP 30 expert（更难，最后做） |

Full-param OPD（slime/Megatron）只有在 #3 LoRA **完全不动分** 且 A 也排除「打错模块」之后才值得上；H20×8 可以做，但解释会变成「把 merge 训回去」。

---

## 结果报告清单（#3 结案必填）

- `#1` teacher mc_average / HC mc_average（新机器）
- `#3` 最终和中间 ckpt 的 mc_average 与 8 任务分
- Δ vs HC、vs teacher
- LoRA target / rank / LR / steps / prompt 文件行数
- 训练 wall time、是否 OOM、student/teacher GPU 切分
- merged HF 路径，方便复测
