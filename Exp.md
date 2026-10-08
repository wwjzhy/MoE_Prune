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

---

# Exp #11（2026-10-08）— Qwen3-30B上的大规模Stage 1验证

> **目标：** 在更大规模、专家数更多的`Qwen3-30B-A3B-Instruct-2507`上验证SPRM的零训练压缩效果。本实验只运行Stage 1；不运行LoRA、KD、OPD或任何压缩后恢复训练。主比较统一使用67%和75%的**routed-expert压缩率**，不得写成整个模型参数量或checkpoint大小缩小67%/75%。

## 研究问题

1. SPRM在128-expert、30B总参数量的MoE上是否仍优于纯剪枝和纯融合方法？
2. 该优势能否同时出现在通用理解、数学推理和代码生成任务上？
3. 从67%提高到75% routed-expert压缩后，SPRM是否比基线具有更平缓的性能下降？

## 固定模型与架构

| 项目 | 设置 |
|------|------|
| Teacher | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| 精度 | BF16 |
| Transformer层数 | 48 |
| Routed experts / layer | 128 |
| Active experts / token | 8 |
| Shared expert | 该checkpoint无独立shared-expert配置 |
| 训练 | 无；所有checkpoint均为one-shot、training-free压缩 |

启动前必须将模型revision、`config.json`、tokenizer revision和chat template保存到实验manifest。禁止不同方法使用不同teacher revision。

## routed-expert压缩预算

定义：

```text
ExpertCompression = 1 - K / 128
```

| ID | 目标压缩率 | 最终物理experts/layer $K$ | 删除/合并的expert槽位 | 实际压缩率 |
|----|-----------:|---------------------------:|-------------------------:|-----------:|
| C67 | 67% | 42 | 86 | 67.1875% |
| C75 | 75% | 32 | 96 | 75.0000% |

论文和结果文件同时记录`K`与实际压缩率。所有方法在同一个压缩率下必须具有完全相同的物理expert预算；不得将42与43个experts都标成同一个67%条件。

SPRM第一轮沿用当前约80%物理槽位保护、20%物理槽位用于Super-Experts的分配：

| ID | SPRM结构 | Residual grouping |
|----|----------|-------------------|
| C67 | `34U+8M` | 剩余94个experts均衡分为8组，每组11或12个 |
| C75 | `26U+6M` | 剩余102个experts均衡分为6组，每组17个 |

这里的`U`表示权重完全不变的protected experts，`M`表示独立Super-Experts。若后续Protect-ratio消融冻结了不同的全局比例，必须在查看Exp #11 test结果前统一修改两个预算；禁止根据Qwen3的GSM8K、MATH-500或代码测试结果单独调节`P:C`。

## 比较方法与checkpoint矩阵

| ID | 方法 | C67（K=42） | C75（K=32） | 说明 |
|----|------|-------------|-------------|------|
| E11-T | Teacher / 未剪枝 | 128 | 128 | 只生成和评测一次 |
| E11-RP | REAP | Top-42 | Top-32 | 按相同calibration上的REAP saliency进行纯剪枝 |
| E11-HC | HC-SMoE（记录中亦称HC-MoE） | 聚类为42 | 聚类为32 | 使用同一实现与固定聚类设置 |
| E11-RM | REAM | 融合为42 | 融合为32 | in-place saliency-centroid merging |
| E11-S | SPRM | `34U+8M` | `26U+6M` | saliency protection + residual merging + grouped router |

总计生成8个压缩checkpoint，加1个Teacher参考。各方法必须从同一Teacher开始，不允许从另一个方法的压缩checkpoint继续处理。

## Calibration与公平性约束

主实验只生成一套任务无关的压缩checkpoint，并用它同时评测General、Math和Code，禁止针对三个benchmark track分别选择experts或调整分组。

```yaml
compression_calibration: C4
num_calibration_sequences: 2048
sequence_length: 2048
seed: 42
dtype: bfloat16
training: false
```

若仓库中已经存在论文主实验冻结的任务无关C4 manifest，则优先复用；若不存在，按上述配置创建一次。必须保存dataset revision、原始样本ID、预处理版本、样本顺序和SHA256。所有方法与两个压缩率共享同一份calibration manifest、teacher activations、tokenizer和chat template。

REAP saliency只计算一次并同时供REAP、REAM和SPRM使用。HC-SMoE可使用自身论文定义的聚类信号，但不得获得更多calibration sequences或tokens。禁止使用GSM8K/MATH-500 test、HumanEval+或MBPP+题目进行expert selection、分组、超参数选择或停止判断。

SPRM必须保留完整`group_manifest.json`与grouped log-sum-exp router；不得为了适配推理后端把它替换成普通42-row或32-row centroid router。当前grouped router未被vLLM原生支持时，五种方法统一使用HF Transformers后端评测。

## Benchmark与评测口径

| Track | Benchmark | 汇总指标 |
|-------|-----------|----------|
| General | ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、MMLU、OpenBookQA、RTE、WinoGrande | MC8 Avg |
| Code | MBPP+、HumanEval+ | Code Avg |
| Math | GSM8K、MATH-500 | Math Avg |

`HumanEval+`使用与Exp #7--#9一致的EvalPlus test与pass@1口径；数据revision、EvalPlus执行器版本、超时和沙箱限制必须在首次运行前冻结。禁止混入标准HumanEval结果。

所有条件共享相同prompt template、few-shot设置、generation参数、答案抽取器、代码执行器和grader。生成任务第一轮统一使用greedy decoding：

```yaml
do_sample: false
temperature: 0
max_input_length: 2048
max_new_tokens: 1024
```

若某个benchmark的官方协议要求不同长度或特殊stop tokens，以官方协议为准，但必须对所有方法一致并在manifest中记录。主文分别报告MC8 Avg、Code Avg和Math Avg，不把八个MC任务与四个生成任务直接按题目数pool成一个总分；如需单一摘要指标，只能额外报告三个domain average的macro average。

## 指标

每个方法、压缩率和任务保存原始分数，并计算：

```text
Delta_Teacher(method, domain)
  = Score(method, domain) - Score(Teacher, domain)

Delta_Best(method=SPRM, domain)
  = Score(SPRM, domain)
    - max(Score(REAP), Score(HC-SMoE), Score(REAM))

CompressionDrop_67_to_75(method, domain)
  = Score_C75(method, domain) - Score_C67(method, domain)
```

同时记录：

1. 每层最终物理expert数、唯一expert权重数与active experts/token；
2. checkpoint磁盘大小、加载后峰值GPU显存；
3. saliency/scoring、grouping、alignment、fusion各阶段wall time及总GPU hours；
4. 每个benchmark的空输出、乱码、重复循环、超时和代码执行失败数；
5. SPRM每层protected expert IDs、残差分组、对齐参考expert、融合权重和source-to-physical router映射。

## 机器占用与一卡一配置并行调度

Exp #11独占一台`8×H20`机器。共享Teacher统计完成后，8个压缩配置必须一一绑定到8张GPU并行运行；禁止让单个配置默认占用整机，也禁止在同一张GPU上同时驻留两个模型进程。

| GPU | 实验配置 | 物理experts/layer | SPRM结构（如适用） |
|----:|----------|-------------------:|----------------------|
| 0 | E11-RP-C67 | 42 | — |
| 1 | E11-HC-C67 | 42 | — |
| 2 | E11-RM-C67 | 42 | — |
| 3 | E11-S-C67 | 42 | `34U+8M` |
| 4 | E11-RP-C75 | 32 | — |
| 5 | E11-HC-C75 | 32 | — |
| 6 | E11-RM-C75 | 32 | — |
| 7 | E11-S-C75 | 32 | `26U+6M` |

每个worker必须显式设置单卡可见性并保持单进程：

```text
CUDA_VISIBLE_DEVICES=<gpu_id>
WORLD_SIZE=1
LOCAL_RANK=0
```

不得继承会自动启动8卡`torchrun`、Ray或vLLM tensor-parallel worker的环境变量。每个配置使用独立的`output_dir`、`tmp_dir`、日志文件和随机数状态；Teacher checkpoint、tokenizer、calibration manifest与只读teacher statistics可以共享。启动前先把Teacher完整下载到机器本地缓存，禁止8个worker同时从远端下载或写同一cache文件。

### 共享预计算

以下内容只计算一次，然后由8个worker只读复用：

1. 冻结后的C4 calibration样本与tokenized inputs；
2. REAP saliency与每层expert排序；
3. 可共享的router profiles、expert activation statistics和teacher output statistics；
4. tokenizer、chat template、benchmark manifests及其SHA256。

共享统计必须带Teacher revision、数据hash、层号、dtype和shape manifest。任一worker发现hash不一致时立即停止，禁止自行重新计算一份不同的calibration统计。HC-SMoE需要的独有聚类统计、REAM需要的matching/fusion统计和SPRM需要的grouping/alignment统计由各自worker在单卡上继续计算。

### 单卡显存预检

正式并发前，先在一张空闲H20上对预计峰值最高的`E11-S-C75`执行：

```text
完整Teacher加载 → 一个MoE层的grouping/alignment/fusion → 20条prompt前向
```

记录峰值HBM和host RAM。只有完整模型与压缩工作区能够稳定驻留单张卡、且峰值HBM不超过该卡容量的90%，才启动8路并行。压缩实现必须逐层处理并及时释放临时对齐矩阵与source-expert副本，不能在GPU上同时保留全模型的第二份权重。

若单卡预检OOM，不允许直接让8个worker各自反复重试；此时“一卡一配置”在当前实现下不可行，应改为`2 GPUs/config`的tensor-parallel方案并分两轮运行。该降级必须记录在资源表中，不能把多卡配置的GPU hours与单卡配置直接比较。

## 运行顺序

1. Exp #11申请并独占一台8×H20机器，冻结Teacher revision、tokenizer、chat template、C4 calibration manifest与全部benchmark revision。
2. 预下载Teacher到本地只读缓存；运行E11-T的20条固定prompt smoke并完成Teacher的MC8、Math、Code参考评测。若已有完全相同revision和评测hash的结果可以直接复用，否则不得跳过。
3. 完成一次共享teacher-statistics预计算，保存FP32统计量、shape manifest与hash；不得让8个压缩worker重复执行该步骤。
4. 对`E11-S-C75`完成单卡显存预检。通过后，按上表同时启动8个独立worker；模型加载阶段可短暂错峰以避免本地磁盘和host RAM瞬时拥塞，但加载完成后8张卡应同时计算。
5. 每个worker依次完成`压缩 → 结构检查 → 20条固定prompt smoke → MC8 → MBPP+ → HumanEval+ → GSM8K → MATH-500`，并持续写入自身状态文件。一个worker失败不得终止其他七个有效worker。
6. 每个worker必须检查最终物理expert数、router shape、Top-8物理expert无重复、前向finite values与checkpoint hash。SPRM额外检查grouped-router映射覆盖全部128个source experts且每个source只映射一次。
7. 8个worker全部完成后汇总逐任务结果、domain averages、`Delta_Best`、C67到C75的性能下降和资源统计；不得在未齐全时用先完成的方法组成临时主表。
8. 第一轮只运行compression seed 42。若SPRM在至少两个domain上优于最强基线，再单独申请下一台机器或下一轮8卡时段，对SPRM及对应最强基线补calibration seed 43、44并报告mean ± std；不要占用第一轮的8个主配置槽位。

## 结果表（完成后回填）

### C67：67.1875% routed-expert压缩

| Method | Experts/layer | MC8 Avg | MBPP+ | HumanEval+ | Code Avg | GSM8K | MATH-500 | Math Avg |
|--------|--------------:|--------:|------:|----------:|---------:|------:|---------:|---------:|
| Teacher | 128 | | | | | | | |
| REAP | 42 | | | | | | | |
| HC-SMoE | 42 | | | | | | | |
| REAM | 42 | | | | | | | |
| SPRM `34U+8M` | 42 | | | | | | | |
| SPRM $-$ best baseline | 0 | | | | | | | |

### C75：75% routed-expert压缩

| Method | Experts/layer | MC8 Avg | MBPP+ | HumanEval+ | Code Avg | GSM8K | MATH-500 | Math Avg |
|--------|--------------:|--------:|------:|----------:|---------:|------:|---------:|---------:|
| Teacher | 128 | | | | | | | |
| REAP | 32 | | | | | | | |
| HC-SMoE | 32 | | | | | | | |
| REAM | 32 | | | | | | | |
| SPRM `26U+6M` | 32 | | | | | | | |
| SPRM $-$ best baseline | 0 | | | | | | | |

### 资源统计

| Method | Compression | Checkpoint GB | Load peak VRAM GB | Compression GPU hours | Eval GPU hours |
|--------|------------:|--------------:|------------------:|----------------------:|---------------:|
| Teacher | 0% | | | 0 | |
| REAP | 67% | | | | |
| HC-SMoE | 67% | | | | |
| REAM | 67% | | | | |
| SPRM | 67% | | | | |
| REAP | 75% | | | | |
| HC-SMoE | 75% | | | | |
| REAM | 75% | | | | |
| SPRM | 75% | | | | |

## 判定与结论边界

1. SPRM在C67与C75的MC8、Code、Math三个domain averages中均高于同预算最佳基线：可以声称方法在更大规模Qwen3-MoE上跨领域、跨两个极端压缩率泛化。
2. SPRM只在C75明显领先、C67与最强基线接近：可以声称保护与残差融合的优势主要出现在更激进的压缩区间。
3. SPRM只在Math或Code领先：结论必须限定到对应domain，并检查任务无关C4 calibration下的expert覆盖，不得写成通用提升。
4. SPRM低于REAM但高于REAP：只能证明融合残差信息优于纯剪枝，不能证明saliency protection优于纯融合；需要结合组件消融解释。
5. E11-S出现NaN、重复物理expert选择、乱码或明显高于其他方法的生成失败率：先检查grouped-router聚合、Top-8去重、权重对齐和tensor保存；修复前不得计入方法结果。
6. 不将单次seed 42结果描述为统计显著或稳定；只有补齐预先规定的seed后才报告mean ± std。

## 必须保存的产物

```text
artifacts/exp11_qwen3_30b/
  data_manifest.json
  teacher_manifest.json
  calibration_manifest.json
  reap_saliency/
  c67/
    reap/
    hc_smoe/
    ream/
    sprm/
  c75/
    reap/
    hc_smoe/
    ream/
    sprm/
  eval_outputs/
  benchmark_scores.json
  domain_summary.csv
  resource_summary.csv
```

每个压缩目录必须包含模型config、source revision、compression config、最终expert计数、checkpoint hash与方法特有manifest。SPRM额外保存完整group/router manifest；REAM保存centroid/source映射；HC-SMoE保存聚类树或最终cluster assignment；REAP保存每层保留expert列表。

**产出：** 8个Qwen3-30B BF16零训练压缩checkpoint、1份Teacher结果、C67/C75的MC8/Math/Code完整评测、逐任务原始输出、domain summary、expert与router manifests、checkpoint大小、峰值显存和GPU-hour统计。

**状态：** 已设计，未运行。执行优先级为`数据与Teacher冻结 → Teacher评测 → C67四方法 → C75四方法 → 结果汇总 → 必要时补seed`；本实验不依赖任何Stage 2训练产物。

---

# Exp #12（2026-10-08）— Protect比例与SPRM组件消融

> **目标：** 在`Qwen1.5-MoE-A2.7B-Chat`的75% routed-expert压缩下，分别回答两个问题：（A）固定15个物理expert槽位时，多少槽位用于保护原始expert最合适；（B）saliency protection、functional grouping、neuron alignment、saliency-weighted fusion和group-preserving router是否分别必要。本实验只有Stage 1，不训练、不运行LoRA/KD/OPD；全部配置在同一台8卡机器上完成。

## 固定模型、数据与评测

| 项目 | 设置 |
|------|------|
| Teacher | `Qwen1.5-MoE-A2.7B-Chat`，冻结revision |
| MoE结构 | 24层；每层60个routed experts；Top-4 routing |
| Expert压缩率 | 75% |
| 最终物理expert预算 | 每层`K=15` |
| Calibration | 与Exp #6一致：NuminaMath 1,536 + The-Stack-Smol 1,536 |
| Calibration长度 | 3,072 sequences × 512 tokens |
| Calibration seed | 42 |
| 评测 | MC8：ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、MMLU、OpenBookQA、RTE、WinoGrande |
| 训练 | 无；全部为one-shot、training-free压缩 |

所有配置必须使用同一份Teacher checkpoint、tokenizer、chat template、calibration manifest、REAP saliency、teacher activation statistics、MC8 evaluator和prompt配置。若Exp #6的manifest与上述配置完全一致，必须复用其Teacher统计、E6-S checkpoint和Teacher MC8结果；hash不一致时不得混用。

## Part A：Protect槽位比例消融

### 定义

本实验中的“保护率”不是原始expert保留率，而是压缩后15个物理槽位中用于protected experts的比例：

```text
ProtectSlotRatio = P / K,  K = 15
P = protected experts
C = residual super-experts
P + C = 15
```

所有配置均执行同一套SPRM流程：选择Top-$P$ saliency experts保持原权重，将剩余`60-P`个source experts均衡分成$C$组，经过functional grouping、neuron alignment和saliency-weighted fusion生成$C$个Super-Experts，并使用grouped log-sum-exp router。`P0`没有protected experts，其余步骤不变。

| ID | ProtectSlotRatio | $P$ | $C$ | 最终结构 | Residual group size |
|----|-----------------:|----:|----:|----------|--------------------:|
| E12-P0 | 0% | 0 | 15 | `0U+15M` | 60个sources分15组，每组4个 |
| E12-P20 | 20% | 3 | 12 | `3U+12M` | 57个sources分12组，每组4或5个 |
| E12-P40 | 40% | 6 | 9 | `6U+9M` | 54个sources分9组，每组6个 |
| E12-P60 | 60% | 9 | 6 | `9U+6M` | 51个sources分6组，每组8或9个 |
| E12-P80 | 80% | 12 | 3 | `12U+3M` | 48个sources分3组，每组16个 |

`P0`是无protected expert的all-merge SPRM端点。本实验不运行100% Protect端点；纯剪枝证据来自主实验中相同Teacher、预算和评测hash下的REAP-15结果，hash不一致时只注明不可直接比较，不在Exp #12补跑。

`E12-P80`与Exp #6 Stage 1的Full SPRM `12U+3M`完全相同；checkpoint、calibration和实现hash一致时只评测/引用一次，禁止重复压缩。该曲线用于分析方法对$P:C$分配的敏感性，不允许根据MC8 test结果回头调整已经报告的主实验配置。

## Part B：组件消融

所有组件消融固定为75% expert压缩、`P=12`、`C=3`、每组16个source experts。除表中指定组件外，其余设置必须与Full SPRM逐项一致。

| ID | Variant | 唯一变化 | 其余组件 |
|----|---------|----------|----------|
| E12-Full | Full SPRM | 无；直接复用`E12-P80` | 完整方法 |
| E12-RP | Random Protect | 每层从60个experts中均匀无放回随机选择12个protected experts | functional balanced grouping + alignment + saliency fusion + grouped router |
| E12-RG | Random Group | 固定Top-12 protected，将剩余48个experts随机均衡分成3组×16 | alignment + saliency fusion + grouped router |
| E12-NA | No Alignment | 组内不执行Hungarian permutation，使用identity neuron order | saliency protection + functional grouping + saliency fusion + grouped router |
| E12-UF | Uniform Fusion | 对齐后使用`alpha_j=1/16`，替代组内saliency归一化权重 | saliency protection + functional grouping + alignment + grouped router |
| E12-CR | Centroid Router | protected rows不变；每个Super-Expert只保留其alignment reference/centroid对应router row，不做group log-sum-exp | saliency protection + functional grouping + alignment + saliency fusion |

全部组件消融只运行compression seed 42。`Random Protect`和`Random Group`必须保存实际随机选择结果，但本实验不据此声称跨seed稳定性或统计显著性。任何消融不得改变calibration样本、最终物理expert数、Top-4 active physical experts、融合dtype或评测后端。

## 指标与结果表

主指标为MC8 Avg，同时报告八个任务的完整分数。每个配置计算：

```text
Delta_Full(task) = Score_variant(task) - Score_Full(task)
Delta_Full_MC8   = MC8Avg_variant - MC8Avg_Full
```

额外保存但不作为主表排序依据：layer-output distortion、每层physical routing load、空输出/乱码数量、checkpoint大小、压缩wall time和峰值显存。

### Protect槽位比例

| Protect slots | Structure | ARC-C | ARC-E | BoolQ | HellaSwag | MMLU | OBQA | RTE | WinoGrande | MC8 Avg |
|--------------:|-----------|------:|------:|------:|----------:|-----:|-----:|----:|----------:|--------:|
| 0% | `0U+15M` | | | | | | | | | |
| 20% | `3U+12M` | | | | | | | | | |
| 40% | `6U+9M` | | | | | | | | | |
| 60% | `9U+6M` | | | | | | | | | |
| 80% | `12U+3M` | | | | | | | | | |

### 组件消融

| Variant | Seed | MC8 Avg | $\Delta$ Full | Output distortion | Failure count |
|---------|-----:|---------:|--------------:|------------------:|--------------:|
| Full SPRM | 42 | | 0 | | |
| Random Protect | 42 | | | | |
| Random Group | 42 | | | | |
| No Alignment | 42 | | | | |
| Uniform Fusion | 42 | | | | |
| Centroid Router | 42 | | | | |

## 单机8卡并行调度

Exp #12独占一台8-GPU机器。Qwen1.5与每个压缩配置均使用单卡、单进程；共享Teacher与calibration统计只读复用。共有11个逻辑行，但`E12-Full`与`E12-P80`是同一配置，因此只有10个唯一checkpoint，需要两轮完成。

### Wave 1：五个比例点 + 三个组件消融

| GPU | 配置 |
|----:|------|
| 0 | E12-P0 |
| 1 | E12-P20 |
| 2 | E12-P40 |
| 3 | E12-P60 |
| 4 | E12-P80 / E12-Full（只生成一次） |
| 5 | E12-RP，seed 42 |
| 6 | E12-RG，seed 42 |
| 7 | E12-NA，seed 42 |

### Wave 2：剩余两个组件

| GPU | 配置 |
|----:|------|
| 0 | E12-UF，seed 42 |
| 1 | E12-CR，seed 42 |
| 2 | Teacher MC8复测（仅当不能安全复用时） |
| 3--7 | 失败配置重试或空闲；不得启动未批准的新消融 |

每个worker设置独立的`CUDA_VISIBLE_DEVICES`、`output_dir`、`tmp_dir`和日志文件，保持`WORLD_SIZE=1`。不得由`torchrun`、Ray或vLLM自动占用整机。两个wave之间只等待Wave 1的压缩checkpoint安全落盘；Wave 1单个非关键worker失败不阻塞其他完成项，失败配置转入Wave 2的GPU 7或单独补跑。

Teacher结果只有在模型revision、MC8 evaluator、prompt、few-shot、dtype与Exp #6结果hash全部一致时才能复用；否则在Wave 2的GPU 2上重测。所有worker必须检查：

```text
physical_expert_count == 15
active_physical_experts_per_token == 4
protected与merge groups互不重叠
protected与merge groups覆盖全部60个source experts
每个source expert只映射到一个physical expert
forward logits和expert weights均为finite
```

## 运行顺序

1. 独占一台8卡机器，冻结Teacher、calibration与MC8 evaluator manifests，完成hash检查。
2. 计算或复用一次saliency、router profiles、expert activations和teacher statistics；禁止每个worker重复做Teacher前向。
3. 在单卡上完成`E12-P80/E12-Full`的20条prompt smoke与manifest自检；若复用Exp #6 checkpoint，则直接验证其hash和输出。
4. 按Wave 1映射同时启动8个worker，依次执行`压缩 → 结构断言 → 20条MC smoke → 完整MC8`。
5. Wave 1产物落盘后按Wave 2映射运行；Random Protect与Random Group只运行seed 42，不追加其它seed。
6. 汇总Protect比例曲线、组件消融表、逐任务结果、output distortion、失败样本与资源统计。

## 判定与结论边界

1. `P80`高于`P0`：支持在all-merge方案中加入saliency protection有效，但不能仅凭Exp #12声称优于纯剪枝；纯剪枝比较引用主实验的REAP-15。
2. MC8随ProtectSlotRatio从0%到80%提高后出现饱和或非单调变化：说明保护与残差覆盖之间存在trade-off；如果持续单调上升，只能报告在已测范围内更多保护更好，不能推断100%端点。
3. `Random Protect < Full`：支持saliency protection；`Random Group < Full`：支持functional grouping。
4. `No Alignment < Full`、`Uniform Fusion < Full`、`Centroid Router < Full`分别支持alignment、saliency weighting和grouped routing。如果某个消融与Full持平或更好，应简化方法或将该组件降为实现选择，不能声称其带来性能提升。
5. Random Protect与Random Group只有seed 42，因此只能作为受控组件对照，不能声称结果具有跨seed稳定性或统计显著性。
6. 本实验只验证Qwen1.5、75%压缩和MC8，不声称组件重要性已经跨模型、跨压缩率或跨Math/Code泛化。

## 必须保存的产物

```text
artifacts/exp12_ablation/
  data_manifest.json
  teacher_statistics_manifest.json
  protect_ratio/
    p0/
    p20/
    p40/
    p60/
    p80_full/
  components/
    random_protect/seed_42/
    random_group/seed_42/
    no_alignment/
    uniform_fusion/
    centroid_router/
  mc8_outputs/
  protect_ratio_summary.csv
  component_ablation_summary.csv
  resource_summary.csv
```

每个目录保存compression config、checkpoint hash、protected IDs、group assignments、alignment references、fusion weights、router mapping、逐任务原始输出、分数与资源日志。`p80_full`只允许存在一份canonical checkpoint，组件表与比例表通过同一hash引用。

**产出：** 五点Protect-slot曲线、Full SPRM与五项组件消融、完整MC8逐任务结果、output distortion、失败样本和资源统计。

**状态：** 已设计，未运行。执行顺序为`共享统计 → Full smoke → Wave 1 → Wave 2 → 汇总`；全部实验在同一台8卡机器上完成，不依赖Exp #11或任何Stage 2训练结果。

---

# Exp #13（2026-10-08）— FP8/W4A16量化模型上的75% Expert压缩兼容性

> **目标：** 直接从两个已量化的`Qwen3-30B-A3B-Instruct-2507` checkpoint出发，在75% routed-expert压缩下比较Original Quantized、REAP、REAM和SPRM，验证SPRM在FP8与W4A16部署格式中是否仍保留相对优势。本实验只做Stage 1，不训练、不运行LoRA/KD/OPD；8个主配置在同一台8卡机器上一卡一配置并行完成。

## 与Exp #9和Exp #11的区别

```text
Exp #9：BF16 expert压缩 → post-training W4A16量化
Exp #11：BF16 Qwen3主实验
Exp #13：从官方/公开量化checkpoint出发 → 在量化模型上执行expert压缩
```

Exp #13验证的是`Quantized checkpoint → Expert compression`兼容性，不复用Exp #9的`Compression → Quantization`结论。Exp #11只提供BF16参考；不得把Exp #11与Exp #13中不同管线产生的checkpoint当作同一模型。

## 固定模型与精度

| Precision track | Source checkpoint | 量化格式 |
|-----------------|------------------|----------|
| FP8 | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | E4M3、128×128 block-wise weight quantization、dynamic activation scheme |
| W4A16 | `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` | INT4 weights、BF16 activations、checkpoint自带compressed-tensors配置 |

两个checkpoint均对应48层、每层128个routed experts、Top-8 routing的Qwen3-MoE。启动前冻结各自的model revision、`config.json`、quantization config、tokenizer revision、chat template和checkpoint hash。禁止依据名称假设量化参数；实际重打包与重新量化必须读取并复用源checkpoint中的量化元数据。

## 固定压缩预算

只运行75% routed-expert压缩：

```text
N = 128
K = 32
ExpertCompression = 1 - 32 / 128 = 75%
TopK_active = 8
```

| Method | 最终结构 |
|--------|----------|
| Original Quantized | 128个量化experts，不压缩 |
| REAP | Top-32原始量化experts，删除其余96个 |
| REAM | 32个in-place merged experts |
| SPRM | `26U+6M`：26个bit-identical protected experts + 6个量化Super-Experts；剩余102个source experts均衡分为6组×17 |

所有压缩方法最终均存储32个物理experts/layer并保持Top-8 active physical experts/token。Router、embedding、normalization、attention和`lm_head`除方法定义所需的router映射外不得改变。

## 两类压缩管线

### REAP：直接量化剪枝

```text
Quantized checkpoint
  → 在量化模型上计算REAP saliency
  → 保留Top-32 expert的量化权重与scales
  → 删除其余96个experts及对应router rows
  → 不重新量化
```

REAP checkpoint中的保留expert tensor、scale和zero-point（如格式存在）必须与Original Quantized逐bit一致。

### REAM/SPRM：反量化、融合、按原格式重新量化

量化整数/FP8编码和原有scales不能直接做神经元排列与权重平均。REAM和SPRM统一使用：

```text
Quantized source expert weights
  → 仅将参与融合的expert projection反量化到BF16
  → BF16 neuron alignment与weight fusion
  → 按当前precision track的原始格式重新量化merged experts
  → 保存新的量化tensor与scales
```

具体约束：

1. FP8重新量化必须复用E4M3与128×128 weight block size，activation scheme保持dynamic；
2. W4A16重新量化必须复用源checkpoint的compressed-tensors scheme、group size、对称性、module coverage及scale/zero-point dtype；
3. REAM的32个centroid在吸收source experts后全部视为已修改，必须重新量化；
4. SPRM的26个protected experts必须连同量化tensor与scales逐bit复制，只有6个Super-Experts重新量化；
5. 非expert量化模块直接从Original Quantized复制，不得为REAM/SPRM重新量化整个模型；
6. 禁止直接平均INT4 codes、FP8 codes或量化scales来代替BF16 fusion。

本实验不增加ReQuant-Teacher主配置，以保持8卡矩阵。为区分融合损失和重新量化误差，REAM/SPRM worker必须在每层保存merged BF16权重重新量化前后的normalized weight MSE、max error、scale range，并对20条固定prompts保存重新量化前后next-token logits差异。若最终结果异常，再追加完整mixed-precision pre-requant evaluation；该诊断不属于第一轮主实验。

## Calibration与共享统计

两个precision track使用相同的任务无关数据样本，但分别在各自Original Quantized模型上计算saliency和teacher statistics：

```yaml
compression_calibration: C4
num_calibration_sequences: 2048
sequence_length: 2048
seed: 42
training: false
```

必须复用Exp #11冻结的C4样本ID、顺序、tokenized inputs和SHA256；如Exp #11尚未冻结，则在Exp #13创建一次并供两种精度共享。FP8和W4A16分别产生一份REAP saliency、router profiles和expert activation statistics，三种压缩方法在同一precision track内只读复用，禁止每个worker独立抽样。

W4A16 merged experts如果需要activation-aware重新量化，使用Exp #9冻结的128条C4量化校准子集与相同顺序；若不存在则从上述2048条compression calibration中固定前128条，单独保存`quant_calibration_manifest.json`。FP8 block-wise weight重新量化不额外使用任务测试数据。

任何GSM8K/MATH-500/HumanEval+/MBPP+/MC8 test样本都不得参与saliency、grouping、alignment、fusion、quantization calibration或超参数选择。

## Benchmark与评测口径

| Track | Benchmark | 汇总指标 |
|-------|-----------|----------|
| General | ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、MMLU、OpenBookQA、RTE、WinoGrande | MC8 Avg |
| Code | MBPP+、HumanEval+ | Code Avg |
| Math | GSM8K、MATH-500 | Math Avg |

沿用Exp #11冻结的数据revision、prompt、few-shot、答案抽取器、EvalPlus版本、代码沙箱、超时和generation config。所有方法在同一precision track必须使用同一后端；由于SPRM需要grouped log-sum-exp router，只有在一个后端同时正确支持Original、REAP、REAM和SPRM时才能形成主表，禁止Original/基线使用vLLM而SPRM单独使用HF后直接比较吞吐或分数。

第一轮生成统一为：

```yaml
do_sample: false
temperature: 0
max_input_length: 2048
max_new_tokens: 1024
```

如果某benchmark的冻结官方协议另有要求，对四种方法一致应用并写入manifest。主文分别报告MC8 Avg、Code Avg、Math Avg，不把任务样本直接pool成一个总分。

## 实验矩阵与单机8卡分配

Exp #13独占一台8-GPU机器；每个配置单卡、单进程运行。两种precision各4个配置，恰好占满8张GPU：

| GPU | ID | Precision | Method | Experts/layer |
|----:|----|-----------|--------|--------------:|
| 0 | E13-FP8-T | FP8 | Original Quantized | 128 |
| 1 | E13-FP8-RP | FP8 | REAP | 32 |
| 2 | E13-FP8-RM | FP8 | REAM | 32 |
| 3 | E13-FP8-S | FP8 | SPRM `26U+6M` | 32 |
| 4 | E13-W4-T | W4A16 | Original Quantized | 128 |
| 5 | E13-W4-RP | W4A16 | REAP | 32 |
| 6 | E13-W4-RM | W4A16 | REAM | 32 |
| 7 | E13-W4-S | W4A16 | SPRM `26U+6M` | 32 |

每个worker显式设置独立`CUDA_VISIBLE_DEVICES`、`output_dir`、`tmp_dir`、日志文件和端口，保持`WORLD_SIZE=1`。不得继承会自动占用整机的`torchrun`、Ray或tensor-parallel环境。两个source checkpoint必须在启动前完整下载到本地只读cache；禁止8个worker同时远程下载或写同一量化cache。

正式并发前分别对`E13-FP8-S`和`E13-W4-S`做单层`dequantize → align → fuse → requantize → reload`round-trip，自检通过后再启动整机。若单卡无法容纳Original Quantized与当前层融合workspace，先改为逐层CPU/BF16 staging并及时释放临时tensor；仍OOM时停止一卡一配置方案，不允许静默CPU offload后仍声称相同GPU效率。

## 必须断言

每个checkpoint保存前必须检查：

```text
compressed physical_expert_count == 32
active_physical_experts_per_token == 8
all quantized tensors/scales/zero-points are finite
quantized module coverage matches the source precision track
REAP retained expert quantized payloads are bit-identical to Original
SPRM 26 protected expert payloads are bit-identical to Original
SPRM groups cover exactly the other 102 source experts, 6 groups × 17
every source expert maps to exactly one SPRM physical expert
grouped Top-8 never selects the same physical expert twice
saved checkpoint reloads without silently dequantizing all experts to BF16
```

另外统计每层实际量化expert projections数量、scale tensor数量、量化后dtype分布与checkpoint字节数。任何方法静默跳过expert量化时，该run无效。

## 指标

除逐任务原始分数外，计算：

```text
CompressionDrop(method, task, precision)
  = Score(method, task, precision)
    - Score(OriginalQuantized, task, precision)

DeltaBestQ(task, precision)
  = Score(SPRM, task, precision)
    - max(Score(REAP, task, precision), Score(REAM, task, precision))
```

同时报告：

1. MC8 Avg、Code Avg、Math Avg；
2. checkpoint GB、加载后peak VRAM、压缩wall time与GPU hours；
3. REAM/SPRM merged weights重新量化前后的weight MSE与20-prompt logit error；
4. 空输出、乱码、重复循环、超时和代码执行失败数；
5. 实际量化模块覆盖、scales/zero-points数量和dtype分布。

## 结果表（完成后回填）

### FP8

| Method | Experts/layer | MC8 Avg | MBPP+ | HumanEval+ | Code Avg | GSM8K | MATH-500 | Math Avg | Checkpoint GB |
|--------|--------------:|--------:|------:|-----------:|---------:|------:|---------:|---------:|--------------:|
| Original FP8 | 128 | | | | | | | | |
| REAP FP8 | 32 | | | | | | | | |
| REAM FP8 | 32 | | | | | | | | |
| SPRM FP8 `26U+6M` | 32 | | | | | | | | |
| SPRM $-$ best baseline | 0 | | | | | | | | — |

### W4A16

| Method | Experts/layer | MC8 Avg | MBPP+ | HumanEval+ | Code Avg | GSM8K | MATH-500 | Math Avg | Checkpoint GB |
|--------|--------------:|--------:|------:|-----------:|---------:|------:|---------:|---------:|--------------:|
| Original W4A16 | 128 | | | | | | | | |
| REAP W4A16 | 32 | | | | | | | | |
| REAM W4A16 | 32 | | | | | | | | |
| SPRM W4A16 `26U+6M` | 32 | | | | | | | | |
| SPRM $-$ best baseline | 0 | | | | | | | | — |

## 运行顺序

1. 独占一台8卡机器，冻结两个source revisions、量化配置、C4 manifests和benchmark manifests，并将两个source checkpoint预下载到本地。
2. 分别在FP8和W4A16 Original上计算一次REAP saliency与共享teacher statistics，保存precision-specific hash；禁止跨precision混用saliency。
3. 对FP8-SPRM和W4A16-SPRM完成单层round-trip、checkpoint reload、量化module coverage和20条prompt smoke；任一失败时先修复对应precision管线。
4. 自检通过后按GPU表同时启动8个worker。Original worker直接评测；REAP worker执行直接量化剪枝后评测；REAM/SPRM worker完成逐层融合、重新量化、reload后评测。
5. 每个worker依次运行`MC8 → MBPP+ → HumanEval+ → GSM8K → MATH-500`，持续保存断点状态；单个worker失败不得终止其他有效worker。
6. 八个配置齐全后生成FP8/W4A16两张主表、逐任务`CompressionDrop`、`DeltaBestQ`及资源/量化误差表。

## 判定与结论边界

1. SPRM在FP8与W4A16的三个domain averages均高于同precision的REAP和REAM：支持SPRM兼容两种量化格式，并在75% expert压缩下保持相对优势。
2. SPRM只在FP8领先、W4A16不领先：只能声称FP8兼容；优先检查W4 super-expert重新量化误差和异常scale，不能泛化到4-bit。
3. REAM/SPRM的pre/post-requant误差很大且二者同步下降：应将问题归因于merge后的重新量化瓶颈，而不是直接归因于保护—融合结构。
4. REAP明显领先合并方法：说明直接删除量化tensor在该格式更稳定；必须如实报告，不为单个方法调节量化参数。
5. Original Quantized本身在统一评测下异常低或出现大量生成失败：先修复加载、chat template、量化kernel与evaluator，暂停解释压缩结果。
6. 本实验比较的是最终已量化部署checkpoint，不证明`Quantize→Compress`与`Compress→Quantize`两种顺序等价；顺序比较仍属于Exp #9/后续专门实验。

## 必须保存的产物

```text
artifacts/exp13_quantized_compression/
  data_manifest.json
  quant_calibration_manifest.json
  fp8/
    original/
    reap32/
    ream32/
    sprm_26u6m/
  w4a16/
    original/
    reap32/
    ream32/
    sprm_26u6m/
  eval_outputs/
  quantization_error/
  benchmark_scores.json
  domain_summary.csv
  resource_summary.csv
```

每个压缩目录必须保存source revision、quantization config、compression config、checkpoint hash、expert计数、router manifest、量化module清单、dtype/scale统计、round-trip误差与逐任务原始输出。Original目录保存不可变source manifest，不重复复制远端checkpoint。

**产出：** FP8和W4A16各4个配置的75% expert压缩兼容性结果、6个压缩checkpoint、2个Original参考、MC8/Code/Math完整评测、重新量化误差和资源统计。

**状态：** 已设计，未运行。执行顺序为`两种precision共享统计 → 两个SPRM round-trip smoke → 8卡并行压缩/评测 → 汇总`；全部主实验在同一台8卡机器上完成，不依赖任何Stage 2训练结果。

---

# Exp #14（2026-10-08）— 论文Stage 1统一实验：4台8×H20并行执行计划

> **目标：** 按本节统一协议完成主实验、保护比例/组件消融、已量化模型兼容性与压缩效率测量。本节是当前待跑实验的唯一调度清单，旧Exp保留为历史记录，不自动加入本次队列。所有配置只运行Stage 1，不运行SFT、LoRA、KD或OPD。
>
> **资源：** 4台机器，每台8张H20，共32张GPU。优先一卡一配置；每个配置单进程、独立输出目录。总计42个压缩/消融配置，另有4个未压缩模型参考评测。已配套独立运行代码和CPU测试，正式作业仍须数据访问与完整模型H20/Docker验收；本节不代表实验已运行。

## 1. 全局协议与配置数量

| 实验部分 | 模型 | Calibration tracks | Expert压缩率 | 方法/配置 | 压缩配置数 |
|----------|------|--------------------|-------------|-----------|-----------:|
| Main | Qwen3-30B-A3B-Instruct-2507 BF16 | G、X | 67%、75% | REAP、HC-SMoE、REAM、SPRM | 16 |
| Ablation | Qwen1.5-MoE-A2.7B-Chat BF16 | X | 75% | 五档保护比例+六个组件配置，Full/P80共用 | 10 |
| Quantized compatibility | Qwen3 FP8、W4A16 | G、X | 75% | REAP、REAM、SPRM | 12 |
| Efficiency | Qwen1.5-MoE-A2.7B-Chat BF16 | G | 75%（本计划默认，待用户核对） | REAP、HC-SMoE、REAM、SPRM | 4 |
| Total | | | | | **42** |

原始模型参考为Qwen3 BF16、Qwen1.5 BF16、Qwen3 FP8和Qwen3 W4A16共4个。原始模型不做压缩，不按calibration track复制；Qwen3 BF16/FP8/W4A16各评测G和X，Qwen1.5 BF16评测X作为消融参考。原始模型参考不计入42个压缩配置。

第一轮所有压缩及随机消融只使用seed 42；不额外运行43、44，不增加保护比例100%的条件。一次seed结果不声称跨seed稳定性或统计显著性。

## 2. 模型、精度与结构预算

| ID | 完整model ID | Precision | 层数 | 原始routed experts/layer | Active experts/token |
|----|---------------|-----------|-----:|-------------------------:|---------------------:|
| Q3 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | BF16 | 48 | 128 | 8 |
| Q15 | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | BF16 | 24 | 60 | 4 |
| F8 | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | 源checkpoint的FP8格式 | 48 | 128 | 8 |
| W4 | `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` | 源checkpoint的W4A16格式 | 48 | 128 | 8 |

实际加载时核对config并记录revision/hash；架构不符时先修正配置。Qwen1.5的shared expert保持原样，不计入压缩比例。

```text
ExpertCompression = 1 - K / E
ProtectSlotRatio = P / K
K = P + C
P = protected expert数；C = Super-Expert数
```

保护比例分母为最终物理expert预算K，不是原始expert数E。所有主方法在同一模型和压缩条件下具有相同K，保留原Top-k设置，并报告逻辑router slots和唯一物理experts数量。路由选择若采用逻辑slots，必须报告去重后的实际物理专家调用数；不将不同路由语义悄悄改成同一种方法。

| Model/condition | 标称压缩率 | K | 实际压缩率 | Full SPRM | Residual groups |
|-----------------|-----------:|--:|-----------:|-----------|-----------------|
| Q3-C67 | 67% | 42 | 67.1875% | `34U+8M` | 剩余94个，8组：6组12个+2组11个 |
| Q3-C75 / F8 / W4 | 75% | 32 | 75% | `26U+6M` | 剩余102个，6组×17个 |
| Q15-C75 | 75% | 15 | 75% | `12U+3M` | 剩余48个，3组×16个 |

显式传入K，避免浮点舍入导致不同方法使用不同expert预算。主实验保护分配沿用当前约80% protected slots、20% merged slots；不根据test结果分别优化保护比例。

## 3. 两份Calibration与数据冻结

只有两个压缩calibration tracks。**X是一份Math+Code混合校准集，不拆成Math/Code两个压缩checkpoint。**

| Track | Calibration | 样本数 | 单样本长度上限 | Seed | 对应Evaluation |
|-------|-------------|-------:|----------------:|-----:|----------------|
| G / General | C4 | 3,072 | 512 tokens | 42 | MC8 |
| X / Math+Code | NuminaMath 1,536 + The-Stack-Smol 1,536 | 3,072 | 512 tokens | 42 | MBPP+、HumanEval+、LiveCodeBench、GSM8K、MATH-500 |

两份校准集的token预算上限均为1,572,864 tokens；同时记录实际非padding tokens。1:1是样本数比例，不声称实际有效tokens恰好1:1。不使用OpenThought作为压缩校准集。

### 建议冻结的预处理设置

```yaml
calibration:
  source_split: train
  seed: 42
  num_sequences: 3072
  max_sequence_length: 512
  batch_size: 1
  split_by_category: false
  truncate: true
  padding_tokens_excluded_from_statistics: true
  cross_sample_packing: false
  mix_X:
    NuminaMath: 1536
    The-Stack-Smol: 1536
    merge_order: concatenate_then_shuffle_seed_42_and_save
  training: false
```

上面是本次拟定设置，需要实际入口显式传入并验收，不能依赖仓库的默认batch_size=8、model_max_length=2048或truncate=false。X分别固定抽取1536条，再合并并用seed 42固定打乱一次，将最终顺序保存；这比已有composite loader的按组件顺序拼接多一个冻结的shuffle步骤，不宣称旧实验已经如此执行。

C4、NuminaMath、The-Stack-Smol的准确HF dataset ID、revision、subset、文本字段、过滤与渲染模板在启动前填写data_manifest。NuminaMath拟使用题目+参考解答，代码拟使用代码正文，C4使用正文；各源具体字段及chat/plain-text模板仍需按实际dataset核对。同一模型同一track的所有方法必须使用完全相同的tokenized inputs。

Math/Code校准数据对本次benchmark进行去重并保存规则/hash；不把benchmark test题目、答案或生成输出加入校准。两个模型可共享源样本ID，各自tokenizer编码后分别保存hash；不得声称不同tokenizer得到完全相同tokens。

## 4. 方法定义、共享统计与实现状态

| Method | 必须采用的定义 | 当前验收要求 |
|--------|----------------|--------------|
| REAP | 在对应track计算REAP saliency，保留Top-K原始experts并删除其余权重及对应router rows | 核对最终K、保留权重不变、reload正确 |
| HC-MoE（代码名HC-SMoE） | expert output characteristic activation距离、average-linkage层次聚类、按routing frequency融合 | 核对论文算法/对齐设置，补齐物理去重与reload，不能仅复制相同权重后声称参数压缩 |
| REAM | 完整saliency-centroid、pseudo-pruning分配、神经元对齐和saliency-weighted in-place fusion；按方法进行sequential layer-wise recalibration | 冻结实现与参数；不得以SPRM分组或简单平均替代REAM |
| SPRM | Top-P saliency保护；对全部剩余experts做balanced grouping，组内对齐后融合为C个Super-Experts；grouped router聚合source routing mass | protected权重不变、residual全覆盖、C个独立expert、router与reload数值检查 |

SPRM拟沿用现有设计：0.5×router-profile cosine + 0.5×gated-output cosine做残余分组相似度；组内选择saliency最高的expert作为对齐reference，按归一化组内saliency融合。对齐cost的归一化、activation/weight系数、clustering迭代与tie-break必须在implementation manifest冻结；文档不能代替实际实现验收。

旧HC入口没有启用permutation，且把merged权重复制回原expert slots。Exp14独立实现保留其frequency融合/无permutation设定，但改为K份独立FFN和原logical-router映射，CPU保存/重新加载测试已通过。不能仅设置save_as_tied_params后未经验证声称物理压缩。

共有8个可独立冻结的source-model/calibration统计组合：

```text
Q3-G, Q3-X, Q15-G, Q15-X,
F8-G, F8-X, W4-G, W4-X
```

在同一source revision、precision、track及observer实现下，REAP saliency、routing frequency和可复用的activation profiles只计算一次；Q3同track的C67/C75可复用。不同precision/track不得混用统计。HC的专用统计按其方法采集；REAM层间模型变化所需的重新校准不能由静态Teacher统计替代。

所有方法必须存储K份物理expert权重，避免“融合成K组，但仍存储E份拷贝”。SPRM grouped router与HC逻辑slot映射均保存到manifest，并分别报告原Top-k参数及实际unique-expert调用分布。

## 5. Evaluation统一设置

### Benchmark与指标

| Track | Benchmark | Split/冻结项 | 主指标 |
|-------|-----------|--------------|--------|
| G | ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、MMLU、OpenBookQA、RTE、WinoGrande | 固定lm-eval版本、task配置及各task默认evaluation split | MC8任务定义下accuracy指标 |
| X Math | GSM8K | official test | 固定flexible answer extraction的accuracy |
| X Math | MATH-500 | official test | answer accuracy |
| X Code | HumanEval+、MBPP+ | EvalPlus固定dataset revision | expanded-test pass@1 |
| X Code | LiveCodeBench | 启动前填写固定release与日期区间 | code-generation pass@1 |

MC8明确用RTE，不用PIQA；旧文档/旧分数若任务列表或metrics不同不能复用。MC8沿用本次冻结lm-eval task配置的accuracy键：对有acc_norm的任务固定采用acc_norm，其余采用acc；manifest保存每个task的准确metric key。MMLU先按固定harness口径汇总，再作为MC8中的一项；MC8对8个任务等权平均，不按题数pool。

```yaml
evaluation:
  backend: hf_transformers_with_verified_method_adapters
  seed: 42
  model_mode: eval
  mc8_num_fewshot: 0
  generation:
    do_sample: false
    temperature: 0
    num_return_sequences: 1
    max_input_length: 2048
    max_new_tokens: 1024
  bf16_dtype: bfloat16
  quantized_dtype: preserve_source_quantization
  initial_batch_size: 1
  code_pass_k: 1
```

2048/1024是沿用此前的第一轮草案，需用户核对；运行前用独立smoke prompts检查长度，再冻结。benchmark特定的chat template、stop tokens、代码模板、few-shot及答案抽取在task manifest里固定；同benchmark所有方法一致。LCB超过输入长度的样本不得静默删除；截断规则和比例必须记录，若改生成预算则对该benchmark全部模型统一修改后再正式运行。batch_size可在正式运行前对所有同组方法共同验证后统一提高，禁止遇到OOM只对某方法改变评测协议而不记录。

保存HumanEval/MBPP基础分数可作为附表，但主表必须是HumanEval+/MBPP+。LCB不是EvalPlus任务，单独保存runner/grader版本、release、日期范围及执行限制。

数学/代码汇总：

```text
MathAvg = mean(GSM8K, MATH-500)
CodeAvg = mean(MBPP+, HumanEval+, LiveCodeBench)
```

所有分数统一转换到0--100再汇总。G和X由不同压缩checkpoint产生，分表报告，不把两者描述成同一个压缩模型同时获得的分数。

当前LCB入口在非server模式下会报错，要求vLLM server；而SPRM grouped router尚未证明能被该server正确执行。正式LCB前必须实现并验证HF生成/LCB grading入口，或验证一个同精度组全部方法均支持的共同后端。HF方案是本计划默认；不在文档中声称适配已完成。代码执行任务设定统一timeout/memory限制，控制CPU并发。

## 6. 保护比例与组件消融：10个唯一配置

两部分都用Q15 BF16、X校准、75%压缩、K=15、Top-4、seed42，并评测完整X五个benchmark。组件消融固定P=12、C=3。

### 保护比例：五档

| ID | ProtectSlotRatio | P | C | Residual grouping |
|----|-----------------:|--:|--:|-------------------|
| AB-P00 | 0% | 0 | 15 | 全部60个分成15组×4 |
| AB-P20 | 20% | 3 | 12 | 57个分12组：9组5个+3组4个 |
| AB-P40 | 40% | 6 | 9 | 54个分9组×6 |
| AB-P60 | 60% | 9 | 6 | 51个分6组：3组9个+3组8个 |
| AB-P80 / AB-Full | 80% | 12 | 3 | 48个分3组×16 |

0%是SPRM的无保护版本，不自动等同于REAM。**不运行100%/15U+0M。** 不用test分数反向选择主实验保护率。

### 组件：六个逻辑行，其中Full复用AB-P80

| ID | Setting | 相对Full的唯一变更 |
|----|---------|------------------|
| AB-Full | Full SPRM | 基准；复用AB-P80 |
| AB-RProtect | Random Protect | seed42随机选12个protected，其余48个重新按Full方式分组；保存选择结果 |
| AB-RGroup | Random Group | protected与Full相同；seed42将剩余48个随机均衡分成3组×16 |
| AB-NoAlign | No Alignment | 复用Full protected、groups、reference与fusion weights，只取消permutation |
| AB-Uniform | Uniform Fusion | 复用Full protected、groups与alignment，将组内权重改为1/16 |
| AB-Centroid | Centroid Router | 专家权重及分组与Full完全一致；Super-Expert改用组内reference/centroid的单独router logit，取消组内routing mass聚合 |

Random Protect/Group只运行seed42。NoAlign/Uniform/Centroid复用Full的不可变分组等产物，避免同时改变多个因素。Centroid Router直接复用Full expert权重，仅改router，不必再次融合。保护比例五档+组件六行-Full重复一次=10个唯一配置；不计原始Q15参考。

## 7. 量化兼容性：12个压缩配置

```text
(F8 or W4) × (G or X) × (REAP or REAM or SPRM)
compression = 75%, E=128, K=32
SPRM = 26U+6M
```

每种量化precision都有G/X两份压缩checkpoint；Original量化模型在两个评测track共用。

从已量化source出发进行压缩。REAP直接删除expert，保留的量化payload/scales/zero-points不变；REAM/SPRM仅对参与融合的expert projections反量化到BF16，执行对齐和融合，再按源checkpoint格式重新量化。

SPRM的26个protected量化experts逐bit保留，仅6个Super-Experts重新量化。REAM按修改后的权重重新量化。非expert模块保持source payload；不得平均INT4/FP8编码或scale代替浮点权重融合。

FP8沿用源checkpoint的format/block-size/activation scheme；W4沿用源compressed-tensors scheme、group-size、对称性、module coverage和scale/zero-point dtype，不根据模型名硬编码。对每个precision/track保存merged BF16权重在重新量化前后的误差，并做save/reload检查。

如果W4重新量化需要activation calibration，本轮采用对应track压缩校准集中的固定128条、最多512 tokens：G来自C4，X为64 NuminaMath+64 The-Stack-Smol。子集样本ID、顺序与hash对REAM/SPRM共享；这是本次建议值，启动前验证expert coverage，不能只给单个方法临时扩大。源量化算法如需要其他设置，先冻结实现并更新此项；不声称128条已获充分验证。FP8 weight-only重新量化如无需activation calibration则标为不适用。

重新量化误差属于该兼容性管线成本，主表不额外增加ReQuant-Teacher配置；异常时补诊断，不将重新量化误差自动解释为融合方法失败。

## 8. 四机运行计划与逐卡分配

机器用A/B/C/D逻辑名称，真实hostname、GPU UUID、H20显存容量、CPU cores/RAM、local SSD空间在启动前填写。下面是默认首发分工，完成后以空闲GPU继续补尾部任务；不是要求整个wave完成才放行下一项。

### 第一批

| GPU | A：Q3主实验X | B：Q3主实验G | C：Q15消融X | D：量化X与Original |
|----:|--------------|--------------|------------|--------------------|
| 0 | MAIN-X-C67-REAP | MAIN-G-C67-REAP | AB-P00 | Q-F8-X-REAP |
| 1 | MAIN-X-C67-HC | MAIN-G-C67-HC | AB-P20 | Q-F8-X-REAM |
| 2 | MAIN-X-C67-REAM | MAIN-G-C67-REAM | AB-P40 | Q-F8-X-SPRM |
| 3 | MAIN-X-C67-SPRM | MAIN-G-C67-SPRM | AB-P60 | Q-W4-X-REAP |
| 4 | MAIN-X-C75-REAP | MAIN-G-C75-REAP | AB-Full/P80 | Q-W4-X-REAM |
| 5 | MAIN-X-C75-HC | MAIN-G-C75-HC | AB-RProtect | Q-W4-X-SPRM |
| 6 | MAIN-X-C75-REAM | MAIN-G-C75-REAM | AB-RGroup | Original-F8：G+X评测 |
| 7 | MAIN-X-C75-SPRM | MAIN-G-C75-SPRM | AB-NoAlign，依赖Full产物 | Original-W4：G+X评测 |

C GPU7在Full分组/对齐产物可用后启动；依赖未完成时可先做Q15 Original参考评测或其他独立检查，不强制空等，也不重新计算一份不同Full分组。

每个X配置运行五个benchmark，G配置只跑MC8。Original-F8/W4各自只评测一次全部G/X，不因两份calibration重复运行Original。

### 后续队列与尾部利用

| 机器/阶段 | GPU分配建议 | 工作 |
|-----------|-------------|------|
| B的MC8配置完成后 | GPU0--5 | F8-G的REAP/REAM/SPRM，以及W4-G的REAP/REAM/SPRM，共6配置 |
| B的空闲槽位 | GPU6 | Original-Q3 BF16的G+X参考 |
| B的空闲槽位 | GPU7 | Original-Q15 BF16的X参考；若已在C完成且hash相同则跳过 |
| C有空闲卡且Full产物已验证 | 最先空闲的2卡 | AB-Uniform、AB-Centroid |
| A/C/D逐卡完成后 | 空闲卡 | 接管仍未完成的独立benchmark评测，优先预计耗时最长的X任务 |
| 一台机器完成其正常队列后，默认B | 固定同一GPU，其余卡不运行干扰作业 | Efficiency四个方法按随机冻结顺序依次测量 |

B无需等8个主配置都结束：单卡空闲且量化G统计准备完成即可接下一配置。跨机接管先完整复制已验证checkpoint与manifest到目标local SSD，不覆盖原产物。不为了利用GPU空槽而同时在同一卡驻留两个大模型。

Efficiency的机器内其他GPU、CPU重负载和checkpoint写入作业暂停；其他三台继续正常队列。该窗口单卡串行是为了获得可比较的资源测量，而不是整个集群停工。

### 启动前置与共享统计

1. 四机安装并冻结同一环境/git revision及dirty diff；把各自需要的模型提前下载到本地只读cache。Q3 BF16优先放A/B，Q15放C及Efficiency机器，量化source放D并同步到B。
2. 冻结G/X source manifests，按模型tokenizer保存tokenized manifests；核对数据量及非padding tokens。
3. 分配最多8个独立source/track统计任务，各占1卡；统计完成的分支立即放行，不等8份全部完成。A负责Q3-X，B负责Q3-G及Q15-G，C负责Q15-X，D负责F8/W4的G/X四份。
4. 统计缓存只在hash匹配时复用；REAM/HC独有步骤仍在各自worker执行。未完成统计时的其他GPU可跑原模型参考或smoke。
5. 对SPRM、HC以及两种量化融合分别做单层、save/reload和20条固定非test prompts检查；已经通过验收的分支进入正式队列。
6. 逐步从1 worker增加到4、再到8，检查GPU/CPU内存与local SSD吞吐；单卡内存不足先逐层staging。若仍不能单卡完成，标记该配置资源例外并显式重排，不能静默占用另一worker的GPU。

不能在8个CPU-offload worker中各自无控制地驻留一份完整BF16模型；同时检查host RAM、融合临时workspace和共享磁盘压力。先测速再给每个worker固定CPU线程和code-grader并发预算。

## 9. Worker设置、断点与失败恢复

拟定最小worker约束：

```yaml
execution:
  gpus_per_worker: 1
  world_size: 1
  cuda_visible_devices: exactly_one_assigned_gpu
  processes_per_gpu: 1
  initial_calibration_batch_size: 1
  initial_eval_batch_size: 1
  unique_output_dir: true
  unique_log_and_port: true
  shared_statistics: read_only_hash_verified
  teacher_statistics_recompute: false_if_matching_cache_exists
  model_download_before_launch: true
  cpu_threads_and_grader_workers: fixed_after_machine_probe
```

只让worker看到指定GPU，避免device_map=auto跨卡。不得继承整机torchrun/Ray/tensor-parallel环境。每个配置有唯一run ID，包含model、precision、track、K、method、ablation及seed，保存source revision、代码hash和数据hash。不按目录存在判断完成。

按阶段记录：

```text
pending → statistics_ready → compressed → reload_verified
        → benchmark_1_done → ... → eval_complete
```

每个benchmark完成立即保存原始输出、grader结果和状态；仅在输出完整并且hash正确时跳过。生成到一半可按sample ID续跑；保存前写临时文件，校验后原子标记完成。worker失败不终止其他worker。

本轮“断点”指压缩阶段/benchmark/样本级恢复，不涉及训练optimizer。逐层压缩只有在保存了层产物、方法需要的中间模型状态与manifest且实现验证通过时才可续跑；否则从干净source重启该压缩任务，仍复用有效校准缓存。不声称现有脚本已经支持这些功能。

配置稳定后，先用少量样本测量每个方法/benchmark的吞吐与长度，保存estimated remaining time；尾部空闲卡先接预计最久的评测。可将同checkpoint的不同benchmark分到不同空闲GPU；不重复生成checkpoint，不改变prompt/grader。主表不承诺预先估计的完成小时数。

## 10. Efficiency测量：4个配置

本轮默认Q15 BF16、G C4 3072×512、seed42、75%压缩、K15。SPRM为12U+3M；三个基线同K。此压缩率是对用户Efficiency清单缺项的建议，需本次check确认。

每个方法包含完整且独立的校准统计采集和压缩流程。源checkpoint预下载，本地读取；统一运行条件，记录：

| Metric | 范围 |
|--------|------|
| calibration_seconds | 该方法所需统计采集及方法要求的额外校准 |
| compression_seconds | 分组/选择、对齐、融合、结构处理 |
| checkpoint_save_seconds | 实际checkpoint写出 |
| total_wall_seconds | 从加载source开始到保存checkpoint完成；排除下载和benchmark评测 |
| GPU hours | 实际占用GPU数×上述时间/3600，单GPU运行时为total_wall_seconds/3600 |
| peak_allocated/reserved_VRAM | 重置CUDA峰值统计后记录整个范围；同时采样进程设备显存，避免遗漏非PyTorch allocations |
| final_checkpoint_GB | 实際模型权重、router及必要量化/映射元数据大小 |

同时给出“统计缓存已存在时”的compression+save时间，不能拿一个方法的cached时间与另一个方法的含calibration时间比较。REAM重复校准计入其成本，HF加载与CPU staging不静默扣除。保存每阶段GPU同步时间、线程数、host RAM峰值及后台负载；测量seed42一轮，避免未申请的额外重复。

Efficiency测量必须实际执行相应方法，即使已有其他配置checkpoint也不能把文件复制耗时当压缩耗时。保存reload/structure结果，但不追加完整benchmark评测。效率结论不从不同后端的吞吐推导。

## 11. 结果表与产物

| 文件/表 | 内容 |
|---------|------|
| main_general_C67_C75 | Q3 Teacher+4方法，MC8逐任务/平均；分别记录K42/K32 |
| main_math_code_C67_C75 | Q3 Teacher+4方法，X五任务及MathAvg/CodeAvg |
| protect_ratio | 五档保护比例，X五任务及两个domain averages |
| component_ablation | Full及五种组件变更，X五任务及相对Full差值 |
| quant_FP8_general / quant_FP8_math_code | FP8 Original+REAP/REAM/SPRM，G/X分表 |
| quant_W4_general / quant_W4_math_code | W4 Original+REAP/REAM/SPRM，G/X分表 |
| efficiency_Q15_C75 | 四方法的GPU hours、分阶段时间、峰值显存及checkpoint大小 |

每个结果必须可追溯到明确checkpoint+data/eval manifests；复用旧结果时全项匹配才接受。保持历史结果原始协议，不把混合校准分数改名为C4校准分数。

```text
artifacts/exp14_stage1_unified/
  manifests/
    experiment_plan.json
    data_G.json
    data_X.json
    tokenized_by_model/
    benchmark_protocols.json
    implementation_status.json
    hardware_by_host.json
  shared_statistics/<source_precision>/<track>/
  main/<track>/<K>/<method>/
  ablation/<setting>/
  quant/<precision>/<track>/<method>/
  original/<source_precision>/
  efficiency/<method>/
  runs/<run_id>/
    config.json
    status.json
    timings.json
    logs/
    checkpoint_manifest.json
    eval/<benchmark>/
  tables/
```

上面是计划中的产物约定，本次文档编辑不创建这些运行目录。run记录包含唯一GPU UUID、hostname、开始结束时间、数据与checkpoint hash、完整命令及异常事件。

## 12. 用户check与开跑前待冻结项

已按用户要求固定：四台8×H20、主模型Q3、主压缩67%/75%、两个calibration tracks均3072×512、量化75%、五档Protect、不跑100%、Random仅seed42、X评测包含LiveCodeBench、只Stage1。

请核对以下建议/待补全值：

| 项目 | 本次草案 |
|------|----------|
| 保护比例消融压缩率 | 75%，K15 |
| Efficiency压缩率 | 75%，K15 |
| Full SPRM主结构 | Q3：34U+8M / 26U+6M；Q15：12U+3M |
| Gen长度 | input2048/new1024，greedy，初始batch1 |
| MC8 | RTE而非PIQA，0-shot，逐task metric key冻结 |
| X混合顺序 | 各1536样本，合并后seed42固定shuffle |
| W4额外requant calibration | 本次实现为weight-only RTN，同原packed格式；不需额外activation calibration，不能声称复用了原GPTQ/MSE量化recipe |
| Source/dataset revisions | 运行前填写，当前不编造版本 |
| LCB release/date及code timeout | release_v6，filter=2025-01-01至2025-07-31，timeout=120秒；v6实际仅覆盖至2025-04，记录实际题目日期与数量，不声称包含5–7月 |
| 对齐/分组的完整实现参数 | 已冻结在reap.exp14.protocol和experiments/exp14/README.md |
| HC物理压缩与对齐 | 已实现K份FFN、原logical-slot router映射；frequency fusion，无permutation；CPU保存/加载一致性测试通过 |
| HF LCB、量化grouped router | 已实现HF生成+官方grader隔离适配，FP8/W4格式往返和路由CPU测试通过；完整模型CUDA/Docker验收待集群执行 |
| 样本级/逐层resume | 已实现身份校验、原子保存、逐样本生成与逐层压缩恢复；REAM恢复前缀及下一层hidden trajectory，CPU中断测试通过 |

**实现状态（2026-10-08）：** 已补齐独立Stage-1入口、四种方法、五档Protect和五项组件变化、原生量化payload、统一评测、四机独立卡队列/可选跨机器补尾、资源测量、断点恢复与结果汇总。仅调度本节42个配置及4个Original参考，不启动训练或旧Exp。方法/校准/路由/保存加载/恢复、自适应评测并发与原有REAP统计测试共37项通过，含真实MC8 adapter的冻结合成数据运行。**尚未执行完整30B/H20跑分及Docker隔离判分，未产生新实验分数；不能将CPU测试当作集群验收。**

实际实现和操作以[运行说明](experiments/exp14/README.md)为准，入口为bash scripts/run_exp14.sh，公开子命令为plan、prepare、prefetch、stats、job、launch、aggregate；自适应队列内部调用eval-helper和grade-task。模型与数据revision在prepare时真实解析并冻结，未编造版本；正式运行前需要本人确认/接受数据集访问协议，并在四台H20上预下载源模型、构建grader镜像和测量RAM/VRAM/SSD峰值。量化使用明确记录的reference dequant-GEMM后端，证明的是格式/功能兼容，不声称优化量化kernel的吞吐优势。与旧通用评测入口相比，数学/LCB异常不再被吞掉。

冷启动Efficiency测量若中断，不伪造可恢复的时间：旧部分产物移动到interrupted-*审计目录，再从干净source冷测量，其他压缩/评测保留有效层和样本继续。未删除任何用户结果；paper/和已有results/不纳入本次实现提交。

## 14. 自适应评测补尾（2026-10-08追加）

推荐四台机器的launch命令均同时启用--steal --adaptive-eval，其他科学setting不变。共享root需支持POSIX文件锁，每台预下载全部四种源模型；本机独立root可启用--adaptive-eval，但不能跨机--steal。

1. 空闲GPU优先领取尚未启动且依赖已就绪的压缩配置；没有可运行的新配置时，协助正在生成的Math/Code评测。
2. 固定32条样本/chunk，primary与helper互斥认领。同一checkpoint全局最多2个helper（尾部最多primary+2 helper=3张卡，每进程仍1卡）；不改变样本、prompt、精度、greedy、2048/1024长度或grader，不删慢题、不缩短生成。
3. 以已完成chunk的平均每题耗时×剩余题数估计尾部时间，优先协助预计剩余时间较长的任务；无观测时按剩余样本数排序。副本加载成本与共享存储开销单独记录，不能保证任意短尾都加速。
4. GPU完成生成即退出释放卡；数学判分与隔离代码grader进入独立CPU队列，每机最多2个grader，代码容器每个4 CPU/8 GiB。某benchmark所有chunks生成完成即可判分，不等其他benchmark。
5. 样本ID、prompt hash、完整实验identity、chunk锁与原子保存保证不重样。异常退出后重跑同一launch命令，复用partial chunk内有效样本；launcher被打断时子进程仍持有claim，防止重复认领。
6. status=waiting_evaluation不等于完成：primary GPU阶段退出且所有benchmark结果完整/身份匹配后才complete；helper/grader失败显式报错并保留已完成兄弟任务。MC8不拆分。
7. Efficiency不启用helper；A机清空本机GPU/CPU队列后单卡冷测量，测量期间不派发新helper或grader，其他三台继续并行。

每机命令示例（A替换为B/C/D）：

    bash scripts/run_exp14.sh launch --root /shared/exp14 --host A --gpus 0,1,2,3,4,5,6,7 --steal --adaptive-eval

调度参数记录在scheduler-<host>.json，逐chunk进度、原始sample、CPU判分秒数和helper GPU hours单独保存。本策略不能用primary GPU时间代表全部评测GPU成本。正式多机吞吐与Docker安全隔离仍须在H20集群验收，CPU并发测试不等于已运行完整评测。
