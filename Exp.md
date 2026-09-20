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

---

# 下一阶段：Hybrid compression + localized LoRA-OPD

> 本节在 Exp #3 已经跑通并完成 8-task MC 评测后生效。旧实验不改；后续只批准按 **Exp #4 → Exp #5** 顺序执行。当前目标从“只恢复 MC”切换为“极低 expert 保留率下的数学/代码生成恢复”。

统一术语：

- `compression ratio`：被移除的 expert 比例；`0.75` 表示从 60 个 routed experts 压缩到 15 个。
- `retention ratio`：最终保留比例；主实验为 `15/60 = 25%`。
- `D_cal`：只用于 expert 打分、相似度和 prune/merge 决策。
- `D_train`：只用于压缩后的 LoRA-OPD 恢复。
- benchmark test set：只评测，禁止进入 `D_cal` 或 `D_train`。

公共设置：

| 项目 | 设置 |
|------|------|
| Teacher | 原始 `Qwen1.5-MoE-A2.7B-Chat`，60 routed experts |
| 主压缩点 | 每层保留 15/60 experts；`compression_ratio=0.75` |
| sanity point | 每层保留 30/60；仅在 15/60 管线异常时排查，不作为主结论 |
| calibration seed | 42 |
| calibration size | 3,072 sequences × 512 tokens |
| calibration mix | NuminaMath 1,536 + The-Stack-Smol 1,536；Math:Code = 1:1 |
| 主评测 | GSM8K、MATH-500、HumanEval+、MBPP+、LiveCodeBench |
| 回归检查 | MC-8：ARC-c/e、BoolQ、HellaSwag、MMLU、OBQA、RTE、WinoGrande |
| 解码 | generative tasks 使用 greedy / temperature 0；代码报告 pass@1 |
| 结果汇总 | `Math Avg`、`Code Avg`、`GEN Avg`、`MC Avg` |

不要使用 OpenThought 做 compression calibration；OpenThought 只属于 Exp #5 的恢复训练集。

---

## Exp #4 — Hybrid：逐 expert 选择 prune 或 merge（无训练）

**研究问题：** 在相同的 15/60 expert 保留率下，逐 expert 选择直接剪枝或合并，能否优于 REAP 的全部剪枝、REAM 的全部合并，以及固定比例 hybrid？

**本实验不进行 LoRA、SFT 或 OPD。** 所有结果都是 one-shot compression 结果。

### 方法

1. 在 `D_cal` 上计算 REAP saliency，选择每层 saliency 最高的 15 个 experts 作为 protected centroids。
2. 对每个 non-centroid expert `j`，按 REAM gated-output + router-logit similarity 找到最相似的 centroid `c*(j)`。
3. 对 `j` 比较两个 counterfactual layer-output distortion：

```text
D_prune(j)       = E_x ||Y_teacher(x) - Y_prune(j)(x)||²
D_merge(j -> c)  = E_x ||Y_teacher(x) - Y_merge(j -> c)(x)||²
```

`D_merge` 必须在神经元对齐、权重融合和 router row 删除之后测量，因此同时包含：

- 被吸收 expert 的信息保留收益；
- 融合对 centroid 原有功能造成的 interference。

决策规则固定为：

```text
if min_c D_merge(j -> c) < D_prune(j):
    merge j into argmin_c D_merge(j -> c)
else:
    prune j
```

不预设 50/50 比例，也不在 benchmark test set 上选择阈值。最终 merge/prune 数量由 `D_cal` 自动产生，但所有方法最终都只保留 15 个物理 experts。

### 合并设置

- centroid 作为 alignment reference；
- activation cost + weight cost 构造 Hungarian matching；
- 对齐后按 REAP saliency 做归一化加权融合；
- 逐层 sequential compression，每压缩一层后重新计算下一层输入激活；
- 删除 non-centroid router rows，不保留重复 expert 副本；
- 一个 centroid 可吸收多个 experts；多 expert 融合一次完成，禁止反复 pairwise average 引入顺序偏差。

### 必须比较的模型

| ID | 方法 | non-centroid 处理 |
|----|------|-------------------|
| E4-T | Teacher | 不压缩 |
| E4-P | REAP | 全部 prune |
| E4-M | REAM | 全部 merge 到 centroid |
| E4-F | Fixed Hybrid | 按 similarity 排序，最高 50% merge，其余 prune |
| E4-H | Ours Hybrid | 按 `D_merge < D_prune` 逐 expert 决策 |

`E4-F` 是必要对照：如果 E4-H 只超过 REAP/REAM、但不超过固定 50/50 hybrid，就不能证明 action selection 有效。

### 评测

仓库已有 generative evaluator，模型导出为标准 HF 后统一运行：

```bash
bash experiments/eval.sh \
  <MODEL_DIR> 42 <PORT> <SERVER_LOG> \
  true true true true false
```

对应开关依次为：MC、EvalPlus、LiveCodeBench、Math、WildBench。最终报告：

```text
Math Avg = mean(GSM8K, MATH-500)
Code Avg = mean(HumanEval+, MBPP+, LiveCodeBench)
GEN Avg  = mean(GSM8K, MATH-500, HumanEval+, MBPP+, LiveCodeBench)
MC Avg   = mean(MC-8)
```

### 结果表（完成后回填）

| Method | Retained | #Merged | #Pruned | Math Avg | Code Avg | GEN Avg | MC Avg |
|--------|---------:|--------:|--------:|---------:|---------:|--------:|-------:|
| Teacher | 60 | 0 | 0 | | | | |
| REAP | 15 | 0 | 45 | | | | |
| REAM | 15 | 45 | 0 | | | | |
| Fixed Hybrid | 15 | 约 23 | 约 22 | | | | |
| Ours Hybrid | 15 | 回填 | 回填 | | | | |

同时保存每层 action manifest：`expert_id → action → target_centroid → D_prune → D_merge`。Exp #5 只依据该 manifest 选择 LoRA 模块。

**产出：** 5 个模型的完整评测表、Ours HF checkpoint、action manifest、每层 merge/prune 数量、wall time 和显存峰值。

**状态：** 未跑。需要先实现/确认 hybrid action selector 和 action manifest；实现完成前不要启动 Exp #5。

---

## Exp #5 — 只对 merged clusters 做 LoRA-OPD 恢复

**研究问题：** 在 Exp #4 的同一个 Ours Hybrid checkpoint 上，只训练发生过融合的 centroids，能否以更少的 trainable parameters 达到或接近 all-retained-expert LoRA，并优于相同参数量的随机 expert LoRA？

### 固定输入

- Student：Exp #4 的 `E4-H` checkpoint，15 retained experts；
- Teacher：原始 60-expert 模型，完全冻结；
- action manifest：必须与 student checkpoint 同一次压缩产生；
- `merged centroid`：manifest 中至少吸收过一个 non-centroid expert 的 retained expert。

### 恢复训练集

生成一个固定文件：

```text
artifacts/opd/openthought_math_code_32k.parquet
```

数据要求：

| 项目 | 设置 |
|------|------|
| 总量 | 32,000 prompts |
| 领域比例 | Math 16,000 + Code 16,000 |
| train | 30,720 |
| validation | 1,280；Math/Code 各 640 |
| split seed | 42 |
| 去重 | 对 GSM8K、MATH-500、HumanEval(+)、MBPP(+)、LiveCodeBench 做 exact + near-duplicate filtering |

在运行前必须把 OpenThought 的准确 HF dataset ID、revision 和字段映射写入本节；不能只写“OpenThought”后使用浮动版本。

### LoRA 对照组

所有训练组使用同一份 student、数据顺序、OPD objective 和训练步数。

| ID | 可训练位置 | 参数量控制 |
|----|------------|------------|
| E5-0 | No LoRA | Exp #4 起点 |
| E5-R | 每层随机选择与 merged centroids 相同数量的 retained experts | 与 E5-H 严格同参数量 |
| E5-H | 只训练 merged centroids | Ours |
| E5-A | 所有 15 个 retained experts | 参数更多的 upper bound |

Random expert 列表在训练前用 seed 42 固定并保存。E5-R 与 E5-H 必须逐层具有相同 LoRA expert 数量。

### LoRA/OPD 设置

沿用已跑通的 Exp #3 optimizer 侧配置，只改变数据、序列长度和精确模块选择：

| 项目 | 设置 |
|------|------|
| LoRA rank / alpha | 32 / 32 |
| target projection | routed expert 的 `gate_proj,up_proj,down_proj` |
| 禁止训练 | attention、router gate、shared expert、未选中的 routed experts |
| LR | `3e-5` |
| OPD loss | `k1`，`use_policy_gradient=True`，`use_task_rewards=False` |
| prompt / response | 2,048 / 1,024 tokens |
| vLLM max model length | 至少 4,096，禁止设成 2,048 |
| max steps | 1,200 |
| save/eval steps | 0、200、400、800、1,200 |
| GPU | 延续验证过的 6 student + 2 teacher；不要重新尝试 async agent-loop |

PEFT 的普通 `target_modules=gate_proj,up_proj,down_proj` 会命中所有同名层，不能直接用于 E5-H。启动前必须打印并保存：

```text
trainable module names
trainable expert ids per layer
trainable parameter count
```

并用 action manifest 断言 E5-H 只命中 merged centroids，E5-R 与 E5-H 参数量完全一致。

### 执行顺序

1. E5-H 先跑 20-step smoke：确认真实 rollout、teacher logprob、optimizer update 和 checkpoint。
2. E5-H 跑到 step 200，先做小规模 validation；没有生成或 loss 异常就停，不排其它组。
3. E5-H 正常后，E5-R、E5-A 使用相同配置跑到 1,200 steps。
4. 对 0/200/400/800/1200 做 validation，只选择一次 checkpoint。
5. 选定 checkpoint 后才跑完整 benchmark test set。

### 评测和指标

使用与 Exp #4 完全相同的 evaluator 和 prompt template。训练 rollout 的 `MAX_RESP=1024` 不限制最终 benchmark 的生成长度；final evaluation 使用各 benchmark 官方设置。

除原始分数外报告 gap recovery：

```text
Recovery = (Score_after_OPD - Score_E4-H)
           / (Score_teacher - Score_E4-H) * 100%
```

### 结果表（完成后回填）

| Method | LoRA experts/layer | Trainable params | Math Avg | Code Avg | GEN Avg | MC Avg | GEN Recovery |
|--------|-------------------:|-----------------:|---------:|---------:|--------:|-------:|-------------:|
| E4-H / no LoRA | 0 | 0 | | | | | 0% |
| Random LoRA-OPD | 与 merged 数相同 | 回填 | | | | | |
| Merged-cluster LoRA-OPD | 回填 | 回填 | | | | | |
| All-expert LoRA-OPD | 15 | 回填 | | | | | |

核心判定：

1. `E5-H > E5-R`：证明定位 merge-induced damage 有效，而不只是 LoRA 参数量作用；
2. `E5-H` 接近或超过 `E5-A`：证明 localized recovery 的参数效率；
3. Math 和 Code 都恢复，MC-8 不发生明显灾难性下降。

**产出：** LoRA checkpoints、merged HF checkpoints、完整生成/MC 表、gap recovery、trainable parameter 数量、训练 wall time、action manifest 和随机 expert manifest。

**状态：** 未跑；依赖 Exp #4 的 E4-H checkpoint 和 action manifest。

---

## 当前停止线

现在只实现并运行 Exp #4，然后运行 Exp #5。不要同时追加 router LoRA、attention LoRA、不同 LoRA rank、不同 calibration mixture 或更多 compression ratio；只有 E4-H 与 E5-H 出现正信号后再单独设计 ablation。
