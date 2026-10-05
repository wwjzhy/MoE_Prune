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

---

# Exp #6（2026-09-23）— 12U+3M + Joint KD/OPD

> **最新调度：** 本节是当前批准执行的实验，优先于文首及旧节中的历史停止线。旧Exp #4/#5保留原编号和内容，不改名、不覆盖。

论文方法保持两个 stage：

```text
Stage 1：每层保护12个重要experts，将其余48个融合为3个super-experts
Stage 2：只在3个super-experts上联合优化teacher-forced KD和on-policy distillation
```

统一命名：`12U+3M` 表示每层最终保留12个 untouched experts 和3个 merge-modified super-experts。最终物理expert数为15/60，retention为25%，compression为75%。

## Exp #6 Stage 1 — 12U+3M 极端压缩（无训练）

**研究问题：** 在仅保留25%物理experts时，保护高saliency experts并将剩余experts集中融合为少量super-experts，是否优于相同保留率的纯剪枝？

### 固定设置

| 项目 | 设置 |
|------|------|
| 模型 | `Qwen1.5-MoE-A2.7B-Chat` |
| MoE层数 | 24 |
| 原始routed experts | 每层60；shared expert不参与压缩 |
| Protected experts | 每层12，保持原权重 |
| Merge pool | 每层剩余48 |
| Super-experts | 每层3，每组固定16个原始experts |
| 最终物理experts | 每层15；12 untouched + 3 super |
| Calibration | NuminaMath 1,536 + The-Stack-Smol 1,536 |
| Calibration长度 | 3,072 sequences × 512 tokens |
| Seed | 42 |

### Stage 1算法

1. 在calibration set上计算REAP saliency，选择每层Top-12作为protected experts。
2. 对剩余48个experts计算组合相似度：

```text
similarity(i, j) = 0.5 * router_profile_cosine(i, j)
                 + 0.5 * gated_output_cosine(i, j)
```

3. 使用balanced/constrained k-medoids划分成3组，严格满足每组16个experts。
4. 每组选择saliency最高的expert作为神经元对齐reference。
5. 使用activation cost + weight cost构造Hungarian matching；同一个排列必须同时应用于`gate_proj`/`up_proj`的行和`down_proj`的列。
6. 对齐后按组内归一化REAP saliency一次性融合16个原始权重：

```text
alpha_j = saliency_j / sum_group(saliency)
W_super = sum_j alpha_j * aligned(W_j)
```

禁止递归pairwise average；每个super-expert必须始终从16个原始expert权重一次性生成。

### Group router

不得删除48个router rows后直接给3个super-experts新建随机/平均gate，也不得保存60份重复expert权重。

保留原始60维router logits，并聚合成15个group logits：

```text
protected singleton: z_group = z_expert
merge group G:       z_group = logsumexp({z_j | j in G})
```

随后在15个group logits上执行Top-4 softmax。router保留60行的开销相对expert FFN可忽略，但实际只允许存在15个物理expert modules。

### 必须断言并保存

每层压缩结束后必须检查：

```text
len(protected) == 12
len(groups) == 3
all(len(group) == 16)
protected与3个groups两两不相交
protected ∪ group_1 ∪ group_2 ∪ group_3 == {0, ..., 59}
physical_expert_count == 15
```

保存`group_manifest.json`，至少包含：

```text
layer_id
protected_expert_ids
三组source_expert_ids
alignment_reference_id
组内saliency和fusion weights
原始expert_id到物理expert_id的router mapping
```

### 最小对照

| ID | 方法 | 每层最终结构 |
|----|------|--------------|
| E6-T | Teacher | 60 experts |
| E6-P | Pure REAP prune | 15 experts |
| E6-S | 12U+3M | 12 untouched + 3×(16→1) super-experts |

旧版dynamic Hybrid和Fixed Hybrid先不跑；当前只验证最简核心假设。

### 评测与停止线

先完成小规模generation smoke，再运行完整：

- GSM8K、MATH-500；
- HumanEval+、MBPP+、LiveCodeBench；
- MC-8回归检查。

报告`Math Avg`、`Code Avg`、`GEN Avg`、`MC Avg`以及真实checkpoint大小。

**停止线：** 若E6-S出现乱码/重复退化，或GEN Avg明显低于E6-P，则先检查group router和物理权重共享；确认实现正确后仍明显更差，则不启动Exp #6 Stage 2。

**复用：** clustering/alignment/fusion优先复用`src/reap/cluster.py`、`src/reap/permute.py`、`src/reap/merge.py`，只新增balanced assignment、group router和manifest输出。

**状态：** 未实现、未跑。

---

## Exp #6 Stage 2 — Super-3 Joint KD + OPD恢复

**研究问题：** 对极端融合产生的3个super-experts同时施加teacher-forced KD和on-policy distillation，能否在只更新5%原始routed expert位置的情况下恢复数学与代码能力？

这是一个统一的recovery stage，不保存KD阶段checkpoint后重新启动OPD；同一个LoRA、optimizer和global-step内联合计算两个loss。

### 固定输入

| 项目 | 设置 |
|------|------|
| Student | Exp #6 Stage 1的E6-S checkpoint |
| Teacher | 原始60-expert模型，完全冻结 |
| Trainable experts | 每层3个super-experts |
| Frozen | 12 protected experts、attention、router、shared expert、base weights |
| Manifest | 必须与Student来自同一次压缩 |

### 数据

准备固定文件：

```text
artifacts/opd/openthought_math_code_32k.parquet
```

| 项目 | 设置 |
|------|------|
| 总量 | 32,000；Math 16,000 + Code 16,000 |
| Train | 30,720 |
| Validation | 1,280；Math/Code各640 |
| Split seed | 42 |
| 去重 | GSM8K、MATH-500、HumanEval(+)、MBPP(+)、LiveCodeBench exact + near dedup |

运行前必须在本节回填准确的Hugging Face dataset ID、revision、prompt字段和reference-response字段，禁止使用浮动版本。

### LoRA

```yaml
lora_rank: 32
lora_alpha: 32
lora_dropout: 0.0
learning_rate: 3e-5
target_projections: [gate_proj, up_proj, down_proj]
target_experts_per_layer: 3 super-experts only
```

启动前打印并断言：

```text
每层恰好3个expert命中LoRA
24层共72个expert实例命中LoRA
protected/shared/router/attention均无trainable parameters
完整trainable module names和parameter count已保存
```

### 联合loss

Teacher-forced KD在OpenThought reference response上计算forward KL：

```text
L_KD = mean_response_tokens KL(p_teacher || p_student)
```

OPD在Student自己生成的response上计算现有K1 distillation loss：

```text
y_student ~ pi_student(. | prompt)
L_OPD = mean_response_tokens K1(p_student, p_teacher; y_student)
```

总loss：

```text
progress   = global_step / total_steps
lambda_kd  = 0.8 - 0.6 * progress
lambda_opd = 1.0 - lambda_kd
loss       = lambda_kd * L_KD + lambda_opd * L_OPD
```

开始时KD:OPD=0.8:0.2，结束时为0.2:0.8。两个loss必须分别按有效response token取均值，并分别记录到日志。

### 单个optimizer step

为降低显存，不同时保留两条计算图：

```text
KD microbatch：reference response → backward(lambda_kd * L_KD)
OPD microbatch：student rollout → backward(lambda_opd * L_OPD)
gradient accumulation完成 → optimizer.step()
```

两条microbatch使用相同domain mix，但不要求同一个prompt。不得把KD和OPD都计算在Student rollout上，否则两项监督退化为重复信号。

### 8×H20配置

```yaml
GPUS: 0,1,2,3,4,5,6,7
STUDENT_GPUS: 6       # FSDP actor + colocated rollout
TEACHER_GPUS: 2
ROLLOUT_TP: 2
TEACHER_TP: 2
ROLLOUT_MODE: sync
FREE_CACHE_ENGINE: false
PARAM_OFFLOAD: false
OPTIMIZER_OFFLOAD: false
LAYERED_SUMMON: false
micro_batch_size: 1
```

20-step smoke：

```yaml
effective_batch_size: 4  # 2 KD + 2 OPD
max_prompt_length: 256
max_response_length: 256
max_model_len: 513
total_steps: 20
```

正式训练在smoke通过且显存允许后使用：

```yaml
effective_batch_size: 4  # 2 KD + 2 OPD；有余量再增至8
max_prompt_length: 2048
max_response_length: 1024
max_model_len: 4096
total_steps: 1200
save_steps: [200, 400, 800, 1200]
```

### 训练记录与断点续训

每次运行创建独立`run_id`目录，禁止复用目录覆盖旧实验。启动时保存一份不可变的`run_config.yaml`，至少记录：

```text
git commit和dirty diff
完整启动命令、环境变量及软件版本
student/teacher checkpoint路径与revision
group_manifest.json的SHA256
训练文件的SHA256、split seed和样本数
LoRA配置、loss配置、batch/token长度、总步数和所有seed
GPU型号/数量、hostname和开始时间
trainable module names及参数量
```

训练过程按global step写入可追加的`metrics.jsonl`，至少记录：

```text
global_step、已处理的KD/OPD samples和有效tokens
L_KD、L_OPD、lambda_kd、lambda_opd、total_loss
learning_rate、grad_norm、rollout response length
step time、累计wall time、tokens/s和各GPU峰值显存
checkpoint保存/恢复事件及异常重试
```

必须支持resume。checkpoint至少包含LoRA/FSDP权重、optimizer、scheduler、`global_step`、随机数状态和data sampler进度；恢复后KD/OPD权重调度必须继续使用恢复出的`global_step / total_steps`，不得重新从0开始。使用verl原生`latest_checkpointed_iteration.txt`选择已完成的checkpoint，不另外维护第二套checkpoint格式。

训练入口提供：

```yaml
RESUME_MODE: auto       # 默认寻找SAVE_DIR中最新的完整checkpoint
RESUME_FROM: null       # 指定checkpoint时优先于auto
SAVE_FREQ: 200
KEEP_LAST: 2
```

`scripts/opd/run_qwen15_hc_lora_opd.sh`已接入verl原生`auto / disable / resume_path`，并支持`RESUME_FROM`和`KEEP_LAST`。正式训练前仍需完成GPU恢复验收：以`total_steps=20`启动，在step 10 checkpoint完整写入后终止作业，再用完全相同配置继续；确认从step 11继续、optimizer/LR/loss权重未重置、旧checkpoint未被覆盖，并成功生成step 20 checkpoint。受vLLM/Ray非确定性影响，不要求恢复后的rollout逐token完全一致。

### 最小对照

| ID | 方法 | 总训练步数 |
|----|------|-----------:|
| E6-0 | E6-S，无恢复 | 0 |
| E6-O | Super-3 OPD-only | 1,200 |
| E6-J | Super-3 Joint KD+OPD | 1,200 |

第一轮只跑这三个。只有E6-J优于E6-O，才追加Random-3和All-15位置消融。除steps外还必须报告teacher/student tokens和GPU hours，避免联合loss因额外计算获得不公平优势。

### 评测与判定

与Exp #6 Stage 1使用完全相同的prompt template和benchmark。训练rollout的`MAX_RESP=1024`不限制final benchmark生成长度。

报告gap recovery：

```text
Recovery = (Score_after - Score_E6-S)
           / (Score_teacher - Score_E6-S) * 100%
```

核心判定：

1. E6-J训练稳定，无乱码、重复循环或reward/loss爆炸；
2. E6-J的GEN Avg高于E6-O；
3. Math和Code均有恢复，MC-8不发生灾难性下降；
4. 训练日志能够证明只有Super-3 LoRA收到梯度。

**复用：** 从`scripts/opd/run_qwen15_hc_lora_opd.sh`派生最小入口；保留现有sync rollout和teacher inference，只新增精确expert LoRA选择、teacher-forced KD microbatch及loss schedule。

**状态：** 未实现、未跑；依赖Exp #6 Stage 1通过停止线。

---

## Exp #6当前执行顺序

1. 实现并自检12U+3M物理压缩、group router和manifest。
2. 跑Exp #6 Stage 1无训练评测；未通过停止线则停止。
3. 实现Super-3精确LoRA选择和Joint KD/OPD loss。
4. 在8×H20上完成10→20 step resume smoke。
5. smoke通过后跑E6-O与E6-J各1,200步并完成统一评测。

当前不要追加更多compression ratio、router LoRA、attention LoRA、LoRA rank或数据配比消融。

---

# Exp #7（2026-09-27）— 数学/代码上的剪枝融合与恢复训练

> **最新调度：** Exp #7补齐数学与代码上的零训练压缩结果，并使用Exp #6 Stage 2的Joint KD+OPD基础设施比较恢复能力；MC-8保留为通用能力回归，不再作为主指标。

**研究问题：** 在相同的75% expert压缩率下，`12U+3M`是否在数学推理和代码生成上稳定优于`REAP-15`、`REAM-15`和`HC-MoE-15`，并且能否只训练3个Super-Experts恢复主要性能？

本实验包含两个部分：Stage A不训练，用于隔离压缩方法本身的贡献；Stage B使用统一Joint KD+OPD恢复，用于比较不同压缩初始化的可恢复性和Super-3局部训练的参数效率。

## 对照方法

| ID | 方法 | 每层结构 | 权重来源 |
|----|------|----------|----------|
| E7-T | Teacher | 60 routed experts | 原始Qwen1.5-MoE-A2.7B-Chat |
| E7-RP | REAP-15 | REAP Top-15，其他45个直接删除 | 复用E6-P |
| E7-RM | REAM-15 | REAP Top-15 centroids吸收其余45个experts | 新生成 |
| E7-HC | HC-MoE-15（论文名HC-SMoE） | 输出相似度层次聚类成15组 | 新生成 |
| E7-S | 12U+3M | 12 untouched + 3×(16→1) super-experts | 复用E6-S |

五种模型必须来自同一份teacher并使用同一份calibration数据。若E6产物的`group_manifest.json`、模型revision或calibration hash不一致，则停止评测并按Exp #6 Stage 1重新生成；一致时禁止重复生成E7-RP和E7-S。

### 基线实现口径

- `REAP-15`：按REAP saliency保留每层Top-15 experts，同时删除其余45个expert权重和对应router rows。
- `REAM-15`：使用完整REAM配置，`merge_size=15`、`saliency=reap`、`grouping=ream`、`merging=logits+weights`、`group_size=16`，启用gated-output similarity、gate-logit similarity和sequential layer-wise recalibration。Top-15为protected centroids，其余45个按pseudo-pruning贪心分配、神经元对齐后按saliency加权融合；按REAM定义删除非centroid router rows。
- `HC-MoE-15`：遵循HC-SMoE基线，使用expert-output similarity、average-linkage hierarchical clustering、activation-based permutation alignment和frequency-weighted merging，将每层60个experts聚为15组；不使用REAP saliency选择centroid。

REAM官方实现未直接覆盖Qwen1.5-MoE，因此移植后必须用小模型/单层数值检查验证分组、排列和融合公式，禁止用近似实现冒充REAM。HC-MoE若保留60个逻辑router slots，可将同组slot映射到同一个物理expert，但checkpoint中只能保存15份唯一expert权重；同时报告逻辑slots、唯一expert数和真实checkpoint大小。

## 固定压缩设置

| 项目 | 设置 |
|------|------|
| 模型 | `Qwen1.5-MoE-A2.7B-Chat` |
| Calibration | NuminaMath 1,536 + The-Stack-Smol 1,536 |
| Calibration长度 | 3,072 sequences × 512 tokens |
| 压缩seed | 42 |
| E7-RP | 每层按REAP saliency保留Top-15 |
| E7-RM | 完整REAM pseudo-pruning，`merge_size=15, group_size=16` |
| E7-HC | HC-SMoE average-linkage聚类成15组 |
| E7-S | 每层保护Top-12，其余48个balanced clustering成3组并对齐融合 |
| 最终物理experts | 四种压缩方法均为15/60；shared expert不参与压缩 |
| Stage A训练 | 无 |

第一轮只跑seed 42。若E7-S在`Math Avg`和`Code Avg`上都高于三个压缩基线中的最佳结果，再补压缩seed 43和44，最终报告3个seed的mean ± std；不要先扩展更多压缩比例。

## 评测任务

### 数学

| Benchmark | Split | 指标 |
|-----------|-------|------|
| GSM8K | test | exact match；使用统一的flexible answer extraction |
| MATH-500 | test | answer accuracy / pass@1 |

### 代码

| Benchmark | Split | 指标 |
|-----------|-------|------|
| HumanEval+ | test | EvalPlus pass@1 |
| MBPP+ | test | EvalPlus pass@1 |
| LiveCodeBench | 固定并记录release/date范围 | pass@1 |

不得只报告E7-S。五种模型必须使用完全相同的数据版本、prompt template、答案抽取器、代码执行器和超时设置。

## 统一生成设置

当前grouped router未被vLLM原生支持，因此五种模型统一使用HF Transformers后端；禁止基线走vLLM而E7-S走HF。

```yaml
backend: hf
dtype: bfloat16
do_sample: false
batch_size: 1
max_input_length: 2048
max_new_tokens: 1024
seed: 42
trust_remote_code: true
```

代码执行必须在隔离sandbox中设置单样本时间和内存限制。LiveCodeBench的数据release、评测日期及grader版本必须写入结果目录，禁止使用浮动latest版本。

## Stage B：Joint KD+OPD恢复训练

### 训练对照

| ID | 初始化 | LoRA experts/layer | 目的 |
|----|--------|-------------------:|------|
| E7-RP-J15 | REAP-15 | 15 | 等参数恢复基线 |
| E7-RM-J15 | REAM-15 | 15 | 等参数恢复基线 |
| E7-HC-J15 | HC-MoE-15 | 15 | 等参数恢复基线 |
| E7-S-J15 | 12U+3M | 15 | 12U+3M恢复上限及等参数比较 |
| E7-S-J3 | 12U+3M | 3个Super-Experts | 提议的局部恢复方法 |

`J15`四个实验必须具有相同LoRA rank和每层命中数量，用于公平比较压缩初始化。`E7-S-J3`只命中每层3个Super-Experts；不得命中12个untouched experts、router、attention或shared expert。OPD-only与Joint KD+OPD的loss消融由Exp #6完成，本实验不为每种压缩方法重复OPD-only。

### 固定训练配置

| 项目 | 设置 |
|------|------|
| Teacher | 原始60-expert模型，完全冻结 |
| 数据 | `openthought_math_code_32k.parquet` |
| Train / Validation | 30,720 / 1,280；Math与Code各半 |
| LoRA | rank 32，alpha 32，dropout 0 |
| Target projections | `gate_proj, up_proj, down_proj` |
| Learning rate | `3e-5` |
| Effective batch | 4；2个KD samples + 2个OPD samples |
| 长度 | prompt 2,048；response 1,024；model length 4,096 |
| 总步数 | 1,200 |
| Checkpoints | 200、400、800、1,200；主结果固定使用step 1,200 |
| 硬件 | 8×H20；6 student/rollout + 2 teacher，teacher TP=2 |
| Resume | `RESUME_MODE=auto`，`KEEP_LAST=2` |

使用与Exp #6完全相同的联合loss：

```text
progress   = global_step / 1200
lambda_kd  = 0.8 - 0.6 * progress
lambda_opd = 1.0 - lambda_kd
loss       = lambda_kd * L_KD + lambda_opd * L_OPD
```

五个训练实验必须使用相同的数据顺序、seed、有效KD/OPD tokens和optimizer steps。除最终分数外，报告trainable parameters、teacher/student tokens、GPU hours和峰值显存；不得用更多rollout或更长训练为某个方法单独调参。

### 训练前置验收

grouped router当前未被vLLM原生支持。正式训练前必须让E7-S在训练rollout后端正确执行grouped-router逻辑，并在20个固定prompts上与HF前向核对生成及next-token logits；未通过一致性检查时不得启动E7-S-J3/J15，也不得用标准15-row router替换后宣称为同一方法。

所有训练方法统一使用同一个sync rollout后端。先完成20-step smoke和`step 10 → resume → step 20`恢复测试，确认optimizer、global step、loss权重调度和LoRA模块命中均正确。

## 运行顺序

1. 校验五个checkpoint、tokenizer、chat template、manifest和数据hash，并断言四个压缩模型每层只有15份唯一expert权重。
2. 每个模型先在每个benchmark上运行20条smoke，检查乱码、重复生成、答案抽取和代码执行。
3. 完整运行五种模型的GSM8K和MATH-500。
4. 完整运行五种模型的HumanEval+、MBPP+和LiveCodeBench。
5. 完成rollout grouped-router一致性、20-step训练和resume验收。
6. 运行E7-RP-J15、E7-RM-J15、E7-HC-J15、E7-S-J15和E7-S-J3各1,200步。
7. 使用完全相同的数学/代码评测配置测试step 1,200，并复测MC-8检查通用能力回退。
8. 汇总seed 42；零训练和训练结果均为正信号后，再对关键对照补seed 43、44。

## 报告指标

分别报告，不用一个总平均掩盖领域差异：

```text
Math Avg = mean(GSM8K, MATH-500)
Code Avg = mean(HumanEval+, MBPP+, LiveCodeBench)
```

每个任务同时报告相对纯剪枝恢复的teacher gap：

```text
Recovery_task = (Score_E7-S - Score_E7-RP)
                / (Score_E7-T - Score_E7-RP) * 100%
```

Recovery固定以REAP-15为参照。另报告E7-S相对三个压缩基线最佳值的绝对差`Delta_best`；若Recovery分母小于等于0，则该任务只报告绝对差值。

训练后的恢复率以每种方法自己的零训练checkpoint为起点：

```text
TrainRecovery(m, task) = (Score_trained(m) - Score_zero(m))
                         / (Score_teacher - Score_zero(m)) * 100%
```

若teacher不高于对应零训练模型，则只报告训练前后绝对差值。

### 结果表（完成后回填）

| Method | GSM8K | MATH-500 | Math Avg | HumanEval+ | MBPP+ | LiveCodeBench | Code Avg |
|--------|------:|---------:|---------:|-----------:|------:|--------------:|---------:|
| E7-T Teacher | | | | | | | |
| E7-RP REAP-15 | | | | | | | |
| E7-RM REAM-15 | | | | | | | |
| E7-HC HC-MoE-15 | | | | | | | |
| E7-S 12U+3M | | | | | | | |
| E7-S − best baseline | | | | | | | |

### 训练结果表（完成后回填）

| Method | LoRA experts/layer | Trainable params | Math Avg | Code Avg | MC Avg | TrainRecovery Math | TrainRecovery Code | GPU hours |
|--------|-------------------:|-----------------:|---------:|---------:|-------:|-------------------:|-------------------:|----------:|
| E7-RP-J15 | 15 | | | | | | | |
| E7-RM-J15 | 15 | | | | | | | |
| E7-HC-J15 | 15 | | | | | | | |
| E7-S-J15 | 15 | | | | | | | |
| E7-S-J3 | 3 | | | | | | | |

## 判定

1. `E7-S > E7-RP`：12U+3M比纯REAP剪枝保留更多目标领域能力；
2. `E7-S > E7-RM`且`E7-S > E7-HC`：优势不是“任何融合都有效”，而来自12U+3M的保护与均衡融合设计；
3. E7-S在Math Avg和Code Avg上都超过三个压缩基线：MC-8优势迁移到论文目标领域，可以启动Super-3恢复训练；
4. 只在一个领域成立：报告领域偏置，并检查mixed calibration比例，不得宣称数学和代码均有效；
5. 两项均未超过最佳基线：Exp #6的MC-8正信号不能支撑论文主结论，先检查grouped router和融合实现，再决定是否训练。
6. `E7-S-J15`超过其他`J15`：12U+3M不仅零训练更强，而且在相同训练参数预算下更容易恢复；
7. `E7-S-J3`接近或超过最佳`J15`基线：仅训练3个受融合影响的experts具有更高参数效率；
8. 所有方法训练后相近：主要收益来自Joint KD+OPD，而不是压缩初始化，论文不得把恢复增益全部归因于12U+3M。

**产出：** 五种零训练模型和五个训练run的逐任务原始输出、grader结果、Math/Code汇总表、gap recovery、训练日志、LoRA checkpoints、运行时间、GPU hours、峰值显存、唯一expert数、真实checkpoint大小及数据hash；代码任务额外保存每个样本的编译/运行状态。

**已保存样例：** `results/exp7_rollout_samples_2026-09-29.txt`（E7-T、E7-RP、E7-RM、E7-HC、E7-S的原始rollout，保留截断、空输出与乱码）。

**状态：** 已完成一轮定性rollout；正式benchmark、grader与训练尚未完成。E6-P/E6-S可复用为E7-RP/E7-S，REAM-15和HC-MoE-15尚需生成；HF版代码评测适配、训练rollout grouped-router支持及Joint KD+OPD实现尚需完成。

---

# Exp #8（2026-10-01）— 低 Expert 保留率曲线（优先 50% Retention）

> **当前最高优先级：** 先完成 `R50`（每层保留30/60 experts）的零训练压缩与评测。Exp #8不启动LoRA、KD或OPD；只测Stage 1。已有checkpoint和统计量能够复用时禁止重复生成。

## 研究问题

随着最终物理expert数量减少，`Protected + Residual Merging`相对纯剪枝和已有融合方法的优势是否持续，并定位模型从稳定到坍缩的retention拐点？

本文统一使用：

```text
retention ratio   = 最终物理expert数 / 60
compression ratio = 1 - retention ratio
```

因此本节的`R50`表示保留30/60、压缩50%，不是“保留50%后再压缩50%”。

## Retention点与本文结构

| ID | Retention | Compression | 最终experts/layer | 本文结构 | 状态 |
|----|----------:|------------:|--------------------:|----------|------|
| R50 | 50% | 50% | 30 | `24U+6M`；剩余36个分成6组，每组6个 | **第一优先级** |
| R33 | 33.33% | 66.67% | 20 | `16U+4M`；剩余44个分成4组，每组11个 | R50完成后运行 |
| R25 | 25% | 75% | 15 | `12U+3M`；剩余48个分成3组，每组16个 | Exp #6/#7已有mixed-calibration参考结果；域专用曲线需重跑 |

暂不把13.33%（8/60）放入主曲线：当前25%时部分基线已经接近任务下限，继续压缩会产生无法区分方法的floor effect。只有当Ours在R25仍保持有效生成和明显非零能力，或Stage 2能显著恢复R25时，才把`6U+2M`作为附录stress test。3/60=5%也不运行，因为3个物理experts小于原模型Top-4，会改变路由定义，不能与其它点直接比较。

## 为什么主曲线暂时止于25%

现有post-hoc expert压缩通常止于25% retention或更高：REAM主要报告75%/50% retention；EEP报告过25% retention；Generic TB-Coverage在Qwen1.5-MoE上报告25%/50%/75% retention。低于25%的代表性结果主要来自为模块化专门预训练的EMO（12.5%和6.25% expert subset），与本文对现成MoE进行post-hoc压缩的设置不同。因此主曲线先用R50、R33、R25定位坍缩拐点，不为了追求更低数字而运行缺乏区分度的R13。

参考：

- REAM: https://arxiv.org/abs/2604.04356
- EEP: https://arxiv.org/abs/2407.00945
- Generic TB-Coverage: https://arxiv.org/abs/2607.01710
- EMO: https://arxiv.org/abs/2605.06663

## Benchmark与域专用Calibration

不同benchmark域使用不同的calibration checkpoint，但同一个benchmark域内所有方法必须使用完全相同的calibration文件、样本顺序、tokenizer、长度、seed和hash。禁止使用benchmark test/validation样本做calibration。

| Track | Calibration `D_cal` | 数量与长度 | Evaluation |
|-------|---------------------|------------|------------|
| General | C4 | 3,072 sequences × 512 tokens | MC-8 |
| Math | NuminaMath | 3,072 sequences × 512 tokens | GSM8K、MATH-500 |
| Code | The-Stack-Smol | 3,072 sequences × 512 tokens | HumanEval+、MBPP+、LiveCodeBench |

三个track分别画曲线，不允许把使用不同calibration checkpoint得到的General、Math、Code分数混成一个总体平均。每个calibration文件固定revision、预处理脚本、seed 42并保存SHA256；Math/Code calibration继续执行与Exp #7相同的benchmark去重规则。

## 对照方法

第一轮使用当前仓库已有且可复现的四种方法：

| ID | 方法 | R50目标 | R33目标 | R25目标 |
|----|------|--------:|--------:|--------:|
| RP | REAP | 30 | 20 | 15 |
| RM | REAM | 30 | 20 | 15 |
| HC | HC-SMoE | 30 | 20 | 15 |
| S | Ours | `24U+6M` | `16U+4M` | `12U+3M` |

所有方法均使用相同teacher、同一track的`D_cal`和相同最终物理expert数。shared expert不压缩。EEP与DM-MoE暂不阻塞R50；主曲线跑通后再作为强基线补入，禁止用近似实现冒充官方方法。

## R50立即执行顺序

1. 生成并冻结General、Math、Code三个calibration文件，记录dataset ID、revision、行数和SHA256。
2. 对每个track先生成Ours `24U+6M`，逐层断言：24个protected、6个merge groups、每组6个source experts、全集覆盖且互不相交、物理expert数为30。
3. 每个checkpoint先在对应track运行20条smoke；检查乱码、重复输出、答案抽取和代码sandbox。
4. Ours smoke通过后，使用完全相同的calibration文件生成REAP-30、REAM-30和HC-SMoE-30。
5. 完整评测对应track；保存逐样本输出、逐任务分数、checkpoint大小、峰值显存、压缩wall time和manifest。
6. 三个track的R50均完成后依次运行R33和R25域专用checkpoint。Exp #6/#7的mixed-calibration R25只作sanity reference，不与域专用R50/R33连成同一条曲线。

## 曲线与结果表

横轴统一使用真实retention：`50%`、`33.33%`、`25%`；纵轴分别绘制`MC Avg`、`Math Avg`和`Code Avg`。每张图只包含对应域专用calibration的结果。

| Track | Method | R50 | R33 | R25 |
|-------|--------|----:|----:|----:|
| General / MC Avg | REAP | | | |
|  | REAM | | | |
|  | HC-SMoE | | | |
|  | Ours | | | |
| Math / Math Avg | REAP | | | |
|  | REAM | | | |
|  | HC-SMoE | | | |
|  | Ours | | | |
| Code / Code Avg | REAP | | | |
|  | REAM | | | |
|  | HC-SMoE | | | |
|  | Ours | | | |

## 停止线

1. R50的Ours若低于同track的REAP，先检查saliency、balanced grouping、neuron alignment和group router，不启动R33。
2. R33正常但R25出现乱码或输出崩溃，先验证Top-4 group routing及物理expert映射；实现正确后仍崩溃则如实报告方法边界。
3. Ours只有在General、Math、Code三条曲线中的至少两条持续优于最佳压缩基线，才能声称低retention优势。
4. Exp #8全程不训练；Stage 2是否保留由Exp #8零训练曲线和单次Joint KD+OPD去留实验另行决定。

**产出：** 三份冻结calibration数据及hash、R50/R33/R25的域专用压缩checkpoints、group manifests、逐任务原始输出与grader结果、三张retention曲线、真实checkpoint大小、压缩时间和峰值显存。

**状态：** 已设计，未运行。当前只批准先跑R50：`General→Math→Code`，且每个track先Ours后基线。

---

# Exp #9（2026-10-01）— W4A16量化兼容性：REAP、REAM与Ours

> **目标：** 验证Expert压缩后的模型能否继续进行post-training weight quantization，以及Ours在量化后是否仍保持相对REAP/REAM的优势。本实验只做量化与评测，不重新压缩、不训练、不运行KD/OPD。

## 与Exp #8的关系

Exp #8已经负责生成和评测BF16下的R50、R33、R25压缩模型。Exp #9只复用其中R50与R25的checkpoint及BF16结果：

```text
BF16压缩与评测：Exp #8
      ↓ 复用相同checkpoint
W4A16 GPTQ量化：Exp #9
      ↓
相同benchmark复测并计算quantization drop
```

若某个Exp #8条件尚未完成，则先补齐对应BF16结果；checkpoint、manifest、tokenizer、chat template和评测版本一致时禁止重复运行BF16评测。R33不进入Exp #9，避免把兼容性实验扩成第二条完整retention曲线。

## 实验矩阵

| ID | 方法 | Retention | 来源checkpoint | 精度 |
|----|------|----------:|----------------|------|
| E9-T-Q | Teacher | 100% | 原始60-expert模型 | W4A16 |
| E9-RP50-Q | REAP | 50%（30/60） | Exp #8 REAP-R50 | W4A16 |
| E9-RM50-Q | REAM | 50%（30/60） | Exp #8 REAM-R50 | W4A16 |
| E9-S50-Q | Ours | 50%（`24U+6M`） | Exp #8 Ours-R50 | W4A16 |
| E9-RP25-Q | REAP | 25%（15/60） | Exp #8 REAP-R25 | W4A16 |
| E9-RM25-Q | REAM | 25%（15/60） | Exp #8 REAM-R25 | W4A16 |
| E9-S25-Q | Ours | 25%（`12U+3M`） | Exp #8 Ours-R25 | W4A16 |

`E9-T-Q`只量化一次，作为未压缩量化参考。HC-SMoE与R33不进入本实验；Exp #9的核心比较是在相同retention和量化设置下比较REAP、REAM与Ours。

## 固定量化设置

第一轮只使用GPTQ，禁止为不同方法分别选择量化器或超参数。

```yaml
quantizer: GPTQModel
weight_bits: 4
activation_dtype: bfloat16
group_size: 128
sym: true
desc_act: false
quant_calibration: C4
num_calibration_sequences: 128
sequence_length: 2048
seed: 42
```

- 量化所有受支持的`Linear`权重，包括attention、routed experts与shared expert；router gate、normalization、embedding和`lm_head`保持BF16。
- Ours必须保留Exp #8的group manifest与grouped-router逻辑；不得把group router替换为普通15/30-row router。
- 所有条件使用同一份冻结C4量化校准文件、相同样本顺序、tokenizer和SHA256。该量化校准集独立于Exp #8用于expert选择/融合的域专用calibration set。
- 量化前统计每个物理expert获得的校准token数。若任一条件存在零命中expert，则统一把校准集扩展到512条并从头量化全部条件，禁止只给某个方法增加校准数据。
- 保存量化日志，并核对每个checkpoint的实际4-bit模块数量；若某类expert权重被静默跳过，该run无效。

## Benchmark与非量化对照

量化模型必须复用对应Exp #8 track的压缩checkpoint并在同一track上评测：

| Track | BF16结果来源 | 量化后评测 |
|-------|--------------|------------|
| General | Exp #8 C4-calibrated checkpoint | MC-8 |
| Math | Exp #8 NuminaMath-calibrated checkpoint | GSM8K、MATH-500 |
| Code | Exp #8 The-Stack-Smol-calibrated checkpoint | HumanEval+、MBPP+、LiveCodeBench |

同一个BF16 checkpoint与其W4A16版本必须使用完全相同的prompt、生成参数、答案抽取器、grader和数据版本。所有模型统一使用HF Transformers后端；在grouped router未获得相同量化kernel支持前，不报告跨后端吞吐对比。

第一优先级先完成General track的7个量化模型。General结果有效后，再量化Math与Code对应的6个压缩checkpoint；Teacher量化checkpoint可在三个track间复用。

## 指标

除量化后原始分数外，对每个方法、retention与任务计算：

```text
QuantDrop(m, task) = Score_W4A16(m, task) - Score_BF16(m, task)
```

同时报告：

1. W4A16与对应BF16的逐任务绝对差；
2. Ours相对同retention下最佳量化基线的差值：

```text
Delta_best_Q = Score_W4A16(Ours) - max(Score_W4A16(REAP), Score_W4A16(REAM))
```

3. checkpoint磁盘大小、加载后峰值GPU显存、量化wall time；
4. 每层最终物理expert数、唯一expert权重数和实际被量化模块数。

## 运行顺序

1. 从Exp #8读取General track的Teacher、REAP-R50/R25、REAM-R50/R25和Ours-R50/R25，并校验checkpoint与manifest hash。
2. 冻结128条C4量化校准数据，先运行`E9-T-Q`和`E9-S50-Q`的量化及20条MC smoke。
3. smoke通过后完成General track其余五个量化模型及MC-8完整评测。
4. 比较对应Exp #8 BF16结果，生成R50/R25的`Score_W4A16`、`QuantDrop`与`Delta_best_Q`。
5. General未发生实现性崩溃后，以同样流程完成Math与Code track；Teacher量化模型不重复生成。
6. 仅当主结果需要补充时，对`Teacher、REAM-R50、Ours-R50`增加W8A16或AWQ消融；该消融不阻塞Exp #9主结果。

## 结果表（完成后回填）

### General

| Method | Retention | BF16 MC Avg（Exp #8） | W4A16 MC Avg | QuantDrop | Checkpoint GB | Peak VRAM GB |
|--------|----------:|----------------------:|-------------:|----------:|--------------:|-------------:|
| Teacher | 100% | | | | | |
| REAP | 50% | | | | | |
| REAM | 50% | | | | | |
| Ours | 50% | | | | | |
| REAP | 25% | | | | | |
| REAM | 25% | | | | | |
| Ours | 25% | | | | | |

### Math与Code

| Track | Method | Retention | BF16 Avg（Exp #8） | W4A16 Avg | QuantDrop | Delta best Q |
|-------|--------|----------:|-------------------:|-----------:|----------:|-------------:|
| Math | REAP | 50% | | | | |
| Math | REAM | 50% | | | | |
| Math | Ours | 50% | | | | |
| Math | REAP | 25% | | | | |
| Math | REAM | 25% | | | | |
| Math | Ours | 25% | | | | |
| Code | REAP | 50% | | | | |
| Code | REAM | 50% | | | | |
| Code | Ours | 50% | | | | |
| Code | REAP | 25% | | | | |
| Code | REAM | 25% | | | | |
| Code | Ours | 25% | | | | |

## 判定与停止线

1. Ours在R50和R25量化后仍高于同retention的REAP与REAM，且`QuantDrop`没有显著大于两者：说明本文压缩结果与4-bit weight-only量化兼容。
2. Ours的BF16优势在W4A16后消失，但三种方法的`QuantDrop`接近：结论应写为量化噪声缩小方法差异，不能声称量化兼容性优势。
3. Ours的`QuantDrop`明显更大：优先检查Super-Expert权重范围、异常值和grouped-router量化模块覆盖；修复实现前不得归因于方法本身。
4. R50正常而R25量化后坍缩：将R25报告为极低retention下量化的失败边界，不为单个方法调节bit-width或校准集。
5. `E9-T-Q`也严重下降：先检查GPTQ实现、chat template和评测环境，暂停全部压缩模型量化。

**产出：** 量化校准文件及SHA256、7个General量化checkpoint、12个Math/Code域专用量化checkpoint、量化配置与日志、逐任务原始输出、BF16/W4A16对照表、checkpoint大小、峰值显存、量化时间、expert token coverage与实际量化模块清单。

**状态：** 已设计，未运行。先执行General：`Teacher → Ours-R50 smoke → REAP/REAM-R50 → Ours/REAP/REAM-R25`；General通过后再运行Math与Code。

---

# Exp #10（2026-10-06）— Figure 1 失败机制：Functional Drop 与 Salient-Expert Drift

> **目标：** 用GSM8K上的三张子图验证极低expert retention下的两个互补问题：纯剪枝造成全局功能丢失，直接把source experts融合进高重要性experts会改变其原有函数。本文方法应同时降低整体输出偏差，并避免显著expert发生漂移。本实验只分析Stage 1，不训练、不运行KD/OPD/LoRA。

## 研究问题与图结构

Figure 1固定为三个panel，不再增加第三种drift：

| Panel | 横轴 | 纵轴 | 回答的问题 |
|-------|------|------|------------|
| (a) Pruning-induced functional drop | Retention：100%、50%、33.3%、25%、13.3%（图中从高到低排列） | $D_{\mathrm{drop}}\downarrow$ | retention降低时，纯剪枝是否持续丢失原模型功能？ |
| (b) Merge-induced salient-expert drift | 每个显著expert吸收的source expert数 $m\in\{0,1,2,3,4,6\}$ | $D_{\mathrm{drift}}\downarrow$ | 即使融合功能最相近的source experts，原显著expert是否随merge load增加而漂移？ |
| (c) Fixed-budget mechanism map | $D_{\mathrm{comp}}\downarrow$ | $D_{\mathrm{drift}}\downarrow$ | 在25% retention下，REAP、REAM与Ours分别落在“信息丢失—显著expert干扰”的什么位置？ |

Panel (c)的每个方法点旁标注GSM8K accuracy；左下角更好。`D_drop`只用于纯剪枝。跨REAP、REAM和Ours比较时必须使用更一般的`D_comp`，不得把merge方法的整体偏差写成`D_drop`。

## 固定模型与数据

| 项目 | 设置 |
|------|------|
| Teacher / Original MoE | `Qwen1.5-MoE-A2.7B-Chat`，BF16、eval mode |
| 分析对象 | 全部24个MoE层的60个routed experts；shared expert保持原样并从三个偏差指标中排除 |
| Compression calibration $\mathcal D_{cal}$ | GSM8K train固定3,072题，seed 42 |
| Mechanism diagnostic $\mathcal D_{diag}$ | 与$\mathcal D_{cal}$不重叠的GSM8K train固定512题 |
| End-to-end evaluation $\mathcal D_{test}$ | GSM8K official test全部1,319题 |
| 校准/诊断序列 | `question + gold reasoning + gold answer`，teacher forcing，截断到512 tokens |
| 生成评测 | 沿用Exp #7：HF Transformers、BF16、greedy、`max_input_length=2048`、`max_new_tokens=1024` |

`calibration.jsonl`、`diagnostic.jsonl`和测试集版本必须保存dataset revision、原始样本ID、预处理脚本版本与SHA256。三个集合不得重叠；official test不得参与saliency、grouping、merging或阈值选择。所有方法必须共享同一份tokenizer、chat template、$\mathcal D_{cal}$和$\mathcal D_{diag}$。

第一轮只使用compression seed 42。图中误差条来自对诊断样本或layer-expert pair的1,000次bootstrap，而不是把token当作独立样本。若Figure 1进入主文并据此声称结果对expert选择稳定，再补seed 43、44并报告跨seed mean $\pm$ std。

## 统一前向口径

对每个$x\in\mathcal D_{diag}$，先用Original MoE做一次teacher-forced前向并缓存：

```text
h[l, x, t]       = Original MoE进入第l个MoE block前的hidden state
router[l, x, t]  = Original router logits、Top-K ids与weights
Y0[l, x, t]      = Original routed-expert branch的聚合输出
```

计算所有方法的偏差时都喂入相同的`h[l,x,t]`，禁止让不同压缩模型各自滚动产生hidden state。这样测量的是当前MoE层压缩造成的局部函数变化，而不是把前层误差重复累计。`Y`只包含routed-expert branch；shared expert、residual connection和attention不进入分子或分母。

统计统一使用FP32累加，$\epsilon=10^{-8}$。先计算每个独立单位的归一化偏差，再做macro average；禁止先平均向量，也禁止把高频expert的大量tokens直接pool成一个全局ratio。

## 指标一：Pruning-induced Functional Drop

对retention $r$的REAP纯剪枝模型，在Original hidden states上定义：

$$
d_{\mathrm{drop}}^{(x,l)}(r)=
\frac{\sum_t\left\|Y_l^0(h_{l,x,t})-Y_l^{\mathrm{prune}(r)}(h_{l,x,t})\right\|_2^2}
{\sum_t\left\|Y_l^0(h_{l,x,t})\right\|_2^2+\epsilon},
$$

$$
D_{\mathrm{drop}}(r)=
\frac{1}{|\mathcal D_{diag}|\,|\mathcal L|}
\sum_{x\in\mathcal D_{diag}}\sum_{l\in\mathcal L}
d_{\mathrm{drop}}^{(x,l)}(r).
$$

运行`60/60、30/60、20/60、15/60、8/60`五个点。除`60/60`外，每层均按同一份REAP saliency保留Top-$K$，并严格使用REAP实际的router-row删除与重新归一化逻辑。Panel (a)画mean与95% bootstrap CI；同时保存被删除experts承接的Original router mass，作为diagnostic，不放入主图：

$$
R_{\mathrm{drop}}(r)=
\mathbb E_{l,x,t}\left[\sum_{i\in\mathcal R_l(r)}p^0_{l,i}(h_{l,x,t})\right].
$$

## 指标二：Merge-induced Salient-Expert Drift

每层用Original MoE的REAP saliency固定Top-12显著expert集合$\mathcal P_l$。对每个$i\in\mathcal P_l$，从$\mathcal D_{diag}$收集被Original router Top-K路由到$i$的hidden states，固定最多256个tokens，记为$\mathcal H_{l,i}$。若任一layer-expert pair不足128个tokens，则把所有panel共用的$\mathcal D_{diag}$统一扩展到1,024道不重叠GSM8K-train样本并从头重算；不能只为某个expert扩数据，更不能使用test补齐。

对每个显著expert，仅把REAP saliency rank 16--60、即REAM-15会删除的45个experts作为source候选，并按REAM使用的functional similarity从高到低排序。对$m\in\{0,1,2,3,4,6\}$，取最相近的$m$个source experts，使用REAM完全相同的neuron alignment与saliency-weighted fusion生成$\widetilde E_{l,i}^{(m)}$。这是单expert受控stress test；不同$i$之间允许复用source候选，不保存为完整压缩checkpoint。$m=3$对应15/60 retention时每个REAM centroid平均吸收3个source experts的负载。

先对每个layer-expert pair计算：

$$
d_{\mathrm{drift}}^{(l,i)}(m)=
\frac{\sum_{h\in\mathcal H_{l,i}}
\left\|E^0_{l,i}(h)-\widetilde E_{l,i}^{(m)}(h)\right\|_2^2}
{\sum_{h\in\mathcal H_{l,i}}\left\|E^0_{l,i}(h)\right\|_2^2+\epsilon},
$$

再对所有显著expert等权平均：

$$
D_{\mathrm{drift}}(m)=
\frac{1}{\sum_l|\mathcal P_l|}
\sum_l\sum_{i\in\mathcal P_l}d_{\mathrm{drift}}^{(l,i)}(m).
$$

Panel (b)画REAM式in-place merge的mean与95% bootstrap CI；另外用蓝色虚线标出Ours protected experts的$D_{\mathrm{drift}}=0$参考线。该虚线只表达“Ours从不把source expert融合进protected expert”，不是同一$m$下的第二条merge曲线。

## 指标三：25% Retention下的二维机制图

只比较三个零训练checkpoint，全部来自本实验同一份GSM8K calibration：

| ID | 方法 | 每层最终结构 | 作用 |
|----|------|--------------|------|
| E10-RP15 | REAP-15 | Top-15原expert，删除其余45个 | 纯剪枝参考 |
| E10-RM15 | REAM-15 | Top-15 centroids吸收其余45个 | in-place merge参考 |
| E10-S15 | Ours | `12 protected + 3 super`，其余48个只融合进3个super experts | 本文方法 |

三个模型均为15/60=25% retention，shared expert不变，不训练。全局压缩输出偏差定义为：

$$
d_{\mathrm{comp}}^{(x,l)}(M)=
\frac{\sum_t\left\|Y_l^0(h_{l,x,t})-Y_l^M(h_{l,x,t})\right\|_2^2}
{\sum_t\left\|Y_l^0(h_{l,x,t})\right\|_2^2+\epsilon},
$$

$$
D_{\mathrm{comp}}(M)=
\frac{1}{|\mathcal D_{diag}|\,|\mathcal L|}
\sum_{x,l}d_{\mathrm{comp}}^{(x,l)}(M).
$$

三个方法的$D_{\mathrm{drift}}(M)$统一在Original Top-12集合$\mathcal P_l$上计算：

$$
D_{\mathrm{drift}}(M)=
\frac{1}{\sum_l|\mathcal P_l|}
\sum_l\sum_{i\in\mathcal P_l}
\frac{\sum_{h\in\mathcal H_{l,i}}
\left\|E^0_{l,i}(h)-\widetilde E^M_{l,\pi_M(i)}(h)\right\|_2^2}
{\sum_{h\in\mathcal H_{l,i}}\left\|E^0_{l,i}(h)\right\|_2^2+\epsilon}.
$$

$\pi_M(i)$由method manifest给出Original显著expert到压缩后expert的映射。对REAP-15和Ours，Top-12权重保持不变，因此$\pi_M(i)=i$且理论上$D_{\mathrm{drift}}\approx0$；REAM-15使用被融合后的对应centroid。该共同Top-12集合不得按方法分别选择，否则纵轴不可比较。

Panel (c)以`D_comp`为横轴、`D_drift`为纵轴，二者都越低越好；每个点标注GSM8K flexible exact-match，误差条分别对prompt和layer-expert pair做bootstrap。Original MoE只作为$(0,0)$灰色参考，不计入三方法比较。

## GSM8K评测与Figure 1制图

Original MoE、E10-RP15、E10-RM15、E10-S15在official test上使用完全相同的greedy generation、prompt template与answer extractor。主标注使用Exp #7的flexible exact-match，同时保存strict exact-match、逐题response、抽取答案和正确性。

Figure 1统一使用色盲安全配色：

```text
REAP  #D55E00  orange-red
REAM  #CC79A7  pink
Ours  #0072B2  blue
Original / reference  #7A7A7A  gray
```

除颜色外，REAP/REAM/Ours分别使用`circle/triangle/star` marker，保证灰度打印可区分。输出vector PDF和300-dpi PNG；三个panel共享字体、线宽和字号。Panel (c)左下角标注`Lower is better`，不要用面积、颜色深浅或第三坐标重复编码accuracy。

## 运行顺序

1. 冻结$\mathcal D_{cal}$、$\mathcal D_{diag}$与GSM8K test manifest，检查样本ID零重叠并保存SHA256。
2. 复测Original MoE的GSM8K；若与Exp #7同配置结果相差超过1.0个绝对百分点，先修复prompt/evaluator，禁止继续。
3. 缓存Original MoE的hidden states、router信息和routed-branch outputs；完成`Original→Original`数值自检。
4. 运行Panel (a)五个REAP retention点并计算$D_{\mathrm{drop}}$与removed router mass。
5. 运行Panel (b)六个merge-load点；只做局部expert forward，不生成六份完整模型。
6. 生成或复用同hash的E10-RP15、E10-RM15、E10-S15，计算Panel (c)的$D_{\mathrm{comp}}$和共同Top-12上的$D_{\mathrm{drift}}$。
7. 对三个25%模型运行完整GSM8K test，并把accuracy标到Panel (c)。
8. 做1,000次bootstrap、导出Figure 1 PDF/PNG和全部raw metrics。

## 数值验收与结论边界

1. `Original→Original`的$D_{\mathrm{drop}}$、$D_{\mathrm{comp}}$和$D_{\mathrm{drift}}$必须小于$10^{-8}$；Panel (a)的100% retention及Panel (b)的$m=0$也必须满足该条件。
2. REAP-15和Ours的共同Top-12没有被改写，故$D_{\mathrm{drift}}$应小于$10^{-6}$；不满足时优先检查checkpoint是否物理共享/覆盖了权重、expert ID映射和缓存hidden states。
3. Panel (a)只有在retention降低时$D_{\mathrm{drop}}$总体上升，才能支持“极端剪枝加剧功能丢失”；不要求每个相邻点严格单调。
4. Panel (b)只有在$m$增大时$D_{\mathrm{drift}}$总体上升，才能支持“in-place merging扰动显著expert”。该图证明functional drift，不使用`expert collapse`一词；若要声称collapse，必须另补effective-rank或representation-diversity证据。
5. Panel (c)若Ours比REAP具有更低$D_{\mathrm{comp}}$、比REAM具有更低$D_{\mathrm{drift}}$，且GSM8K accuracy最高，则支持“两类失败机制需要同时处理”。该结果是机制证据与性能相关性，不表述为严格因果证明。
6. 若Ours的$D_{\mathrm{comp}}$更低但accuracy未提高，应报告局部输出偏差不足以完全预测端到端推理性能，不得只展示有利的两个偏差指标。

## 结果表（完成后回填）

### Panel (a)

| Retention | Experts/layer | $D_{\mathrm{drop}}$ | 95% CI | Removed router mass |
|-----------|--------------:|--------------------:|-------:|--------------------:|
| 100% | 60 | | | |
| 50% | 30 | | | |
| 33.3% | 20 | | | |
| 25% | 15 | | | |
| 13.3% | 8 | | | |

### Panel (b)

| Sources merged per salient expert $m$ | $D_{\mathrm{drift}}$ | 95% CI | Layer-expert pairs |
|---------------------------------------:|---------------------:|-------:|-------------------:|
| 0 | | | |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 6 | | | |

### Panel (c)

| Method | Retention | $D_{\mathrm{comp}}$ | $D_{\mathrm{drift}}$ | GSM8K flexible EM | GSM8K strict EM |
|--------|----------:|--------------------:|---------------------:|-------------------:|-----------------:|
| REAP-15 | 25% | | | | |
| REAM-15 | 25% | | | | |
| Ours `12U+3M` | 25% | | | | |

## 必须保存的产物

```text
artifacts/fig1_failure/
  data_manifest.json
  teacher_cache_manifest.json
  pruning_metrics.csv
    # seed, retention, prompt_id, layer, numerator, denominator,
    # routed_token_count, removed_router_mass
  drift_metrics.csv
    # seed, method, merge_size, layer, expert_id, mapped_expert_id,
    # n_tokens, numerator, denominator
  method_summary.csv
    # seed, method, retention, d_comp, d_drift,
    # gsm8k_flexible, gsm8k_strict
  gsm8k_outputs/
  figure1_failure.pdf
  figure1_failure.png
```

**产出：** 三个无训练25% retention checkpoints及manifest、五个纯剪枝retention点、六个受控merge-load点、完整GSM8K输出、两个偏差指标的raw numerator/denominator、bootstrap置信区间和最终三panel Figure 1。

**状态：** 已设计，未运行。优先执行`数据冻结 → Original自检 → Panel (a) → Panel (b) → 三个25%模型与Panel (c)`；本实验不被Exp #7 Stage B训练阻塞。
