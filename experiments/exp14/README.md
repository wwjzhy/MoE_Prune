# Exp14 — Stage-1 implementation and four-node execution

Status: implemented and CPU contract-tested; no new benchmark score is claimed.
The full 30B/H20 run and Docker-isolated grading still require cluster acceptance.
The paper/ and results/ directories are not part of this implementation commit.

## Protocol

| Part | Model | Calibration | K / methods | Unique configs |
|---|---|---|---|---:|
| Main | Qwen3 BF16 | G and X | K42/K32; REAP, HC, REAM, SPRM | 16 |
| Protect / component ablation | Qwen1.5 BF16 | X | K15; P0/3/6/9/12 and five component changes | 10 |
| Quant compatibility | Qwen3 FP8 / W4A16 | G and X | K32; REAP, REAM, SPRM | 12 |
| Efficiency | Qwen1.5 BF16 | G | K15; REAP, HC, REAM, SPRM | 4 |

Four original-model references are additional jobs. No 100% protect condition,
no seed 43/44, and no training, LoRA, KD or OPD is launched.

G = allenai/c4, en/train, 3072 texts.
X = AI-MO/NuminaMath-1.5 train (cn_k12 / olympiads, problem + solution) 1536
and bigcode/the-stack-smol train (content, all source languages) 1536.
Both use seed 42, stream shuffle buffer 10000, minimum 32 tokens for intake,
max 512 tokens, no cross-document packing. X is concatenated and shuffled once.
Sequences are not padded in the layerwise replay; effective token counts are saved.
All methods consume the same frozen inputs per source model and track.

Calibration revisions, raw texts/IDs, tokenizer revision and token hashes are
saved before jobs run. Exact normalized 13-word question overlap and duplicate
texts are rejected; this is not a claim of exhaustive semantic decontamination.
MC8/evaluation questions participate in the filter. Dataset access agreements
(e.g. The Stack) must be accepted by the human owner; preparation fails visibly
when access is missing.

G evaluates the eight tasks in Exp.md (including RTE, not PIQA), zero-shot,
acc_norm where present, else acc; MMLU is aggregated once.
X evaluates HumanEval+ and MBPP+ expanded tests, LiveCodeBench, GSM8K,
MATH-500. Generation is greedy, batch 1, input max 2048 / new max 1024.
GSM uses last-number flexible extraction; MATH uses math-verify 0.8.0.
Prompts and answer extractors are frozen and shared by teacher and methods.

LCB is release_v6 with the requested 2025-01-01 through 2025-07-31 filter.
That release only contains problems through April 2025: this evaluates the
intersection, NOT additional May-July problems. Preparation saves actual min/max
dates, problem IDs and count. Use a new protocol/root if another release is desired.

## Exact method conventions

- REAP: conditional mean of top-k-renormalized router-weighted output norms,
  top-K original experts; remove their router rows.
- HC-SMoE: raw expert output means, Euclidean average-linkage, centroid nearest
  cluster mean, routing-frequency fusion, no neuron alignment. Original logical
  router slots/top-k are preserved and mapped to K physical experts; repeated
  hits are combined before executing a physical expert once.
- REAM: saliency centroids, greedy pseudo-pruning with group_size 16 including
  the centroid; averaged router-distance/output-cosine similarity; joint
  activation/weight Hungarian alignment; saliency-weighted fusion; centroid
  router; sequential replay through every already-compressed layer.
  Its saliency intentionally follows the reference implementation's gated-output
  then probability multiplication, distinct from the repository's REAP statistic.
  See THIRD_PARTY_NOTICES.md for the frozen reference revision.
- SPRM: protect top-P REAP saliency experts unchanged. Choose C saliency anchors
  from the residual set and greedily assign similarity pairs under balanced
  capacities. Similarity = 0.5 router-probability-profile cosine + 0.5 cosine of
  mean gated output. Highest-saliency member is each alignment reference.
  Gate/up rows and down columns are permuted together, then FP32 saliency fusion
  is performed. All residual experts are covered exactly once.
  Source router logits are combined by logsumexp per group, then top-k is applied
  over physical groups (NOT the HC logical-slot semantics).

Alignment uses the same deterministic token indices across experts, targeting
32768 samples/layer (10 per sequence for 3072 sequences, possibly fewer for
short inputs); neuron activation vectors are row-normalized. Weight signatures
concatenate gate row, up row and down column, row-normalize, joint PCA rank64
(seed0, official projected singular-value scaling), then row-normalize.
Cost is Euclidean activation distance plus Euclidean projected-weight distance
with coefficients 1:1. Alignment distance calculation runs on the assigned GPU;
Hungarian assignment and streaming FP32 fusion use host staging. Cached teacher
weight features are reused; REAM's activation trajectory is recalibrated.

Component changes are one at a time: Random Protect / Random Group use only
seed42; No Alignment, Uniform Fusion and Centroid Router reuse Full's exact
groups. Uniform changes only fusion coefficients; Centroid changes only routing.
Q3 Full is 34U+8M (K42) or 26U+6M (K32); Qwen1.5 Full is 12U+3M (K15).
P0 is a balanced pure-merge endpoint, not an alias for REAM.

## Quantization scope

Native FP8 E4M3 block128x128 / dynamic input activation and compressed-tensors
symmetric group128 packed W4A16 payloads are supported. Protected expert payloads
and REAP-selected experts are not changed. Only merged experts are dequantized,
aligned/fused and re-encoded. FP8 uses block max scaling; W4 uses signed RTN.
This preserves the storage format, NOT the original GPTQ/MSE optimization
recipe. RTN is weight-only and does not need the conditional 128-example
activation recalibration proposed in the initial plan.

Inference uses the explicitly recorded reference dequant-GEMM backend, with
dynamic FP8 input quantization where required. No permanent full-model BF16
conversion, integer-code averaging or silent unsupported-format fallback occurs.
This is functional/storage compatibility, NOT evidence of an optimized FP8/W4
CUDA kernel or quantized throughput advantage. All comparisons within a
precision use this same backend. Unsupported metadata fails rather than drifting
to another experiment. Quant compatibility checkpoints should be evaluated with
the adapter below, not unmodified vLLM/AutoModel loading.

## Setup (on each H20 node)

Use the same Python 3.12 environment on all four nodes. This entry point does
not require the old OPD/vLLM environment.

    python -m pip install -r experiments/exp14/requirements.txt
    git clone https://github.com/mklasby/LiveCodeBench.git /opt/exp14-lcb
    git -C /opt/exp14-lcb checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
    export PYTHONPATH="$PWD/src:/opt/exp14-lcb"
    docker build -f experiments/exp14/Dockerfile.grader -t moe-exp14-grader:1 .

Docker grading has no network, read-only root, no capabilities, limited PIDs,
8 GiB RAM and four CPUs. It receives only frozen benchmark data and generated
samples, not host credentials or arbitrary host directories.
Docker and model/dataset access are explicit cluster prerequisites.
Generated code is never executed by the host worker.

## Prepare and launch

Run preparation ONCE after committing the code and then keep the implementation
and environment frozen. Models/datasets are resolved to immutable revisions.
Using an existing prepared root with changed revisions/settings is rejected.

    bash scripts/run_exp14.sh plan
    bash scripts/run_exp14.sh prepare --root /shared/exp14

Prefetch on each node. With cross-node tail stealing, each node must prefetch
all models; otherwise specify its host to reduce local checkpoint copies.

    bash scripts/run_exp14.sh prefetch --root /shared/exp14
    bash scripts/run_exp14.sh launch --root /shared/exp14 --host A --dry-run

The root must be shared and support POSIX advisory file locking when using
--steal. Each node uses its own local HF cache. Calibration/evaluation artifacts
can be copied to identical node-local roots for higher disk throughput, but then
use fixed host queues WITHOUT --steal and copy jobs/ back to the central root
for aggregation. Do not share eight independent model download writers.

Recommended: warm Q3-X on A, Q3-G on B, Q15-X on C, and the four FP8/W4 tracks
on distinct D cards. These seven reusable caches plus Q15-G efficiency's cold
statistics cover all eight source-track combinations. REAM still computes its
own sequential statistics. Example:

    CUDA_VISIBLE_DEVICES=0 bash scripts/run_exp14.sh stats --root /shared/exp14 --model Q3 --track X

After per-node prefetch/acceptance, run one command on each node:

    bash scripts/run_exp14.sh launch --root /shared/exp14 --host A --gpus 0,1,2,3,4,5,6,7 --steal --adaptive-eval
    bash scripts/run_exp14.sh launch --root /shared/exp14 --host B --gpus 0,1,2,3,4,5,6,7 --steal --adaptive-eval
    bash scripts/run_exp14.sh launch --root /shared/exp14 --host C --gpus 0,1,2,3,4,5,6,7 --steal --adaptive-eval
    bash scripts/run_exp14.sh launch --root /shared/exp14 --host D --gpus 0,1,2,3,4,5,6,7 --steal --adaptive-eval

A first prioritizes Q3-X, B Q3-G, C the Qwen1.5 ablations, D quant-X and quant
references; B then quant-G and BF16 references. Dependencies wait only for Full's
compressed checkpoint, not its entire evaluation. Tail stealing claims any
ready, non-efficiency configuration once; failed configurations are not
automatically stolen/retried forever. A's efficiency runs are single-card and
wait for its local workers/GPUs to become idle. The other three nodes continue.

One process sees exactly one H20. Four CPU threads/worker and four grader CPUs
are the initial budgets. The scheduler reserves not-yet-resident host memory
(Q3 95, Q15 65, F8 60, W4 45 GiB/job) and may run fewer than eight workers when
RAM is insufficient. These are conservative budgets, not measured peaks.
Do not remove memory guards to force eight workers into an undersized host.
Disk planning is also necessary: full activation statistics and layer resume
copies can consume several TB (roughly 4 TB shared-root provision initially);
measure and provision local SSD/NFS bandwidth before all 32 workers start.

Inspect logs/ and jobs/<id>/status.json. After interruption repeat the launch
command: valid layer/statistics/benchmark/sample records are resumed under strict
identity hashes. REAM restores the compressed prefix AND the next-layer hidden
trajectory. Cold efficiency attempts cannot resume their timing fairly, so
their partial artifacts are moved into interrupted-* audit folders and a clean
cold attempt is made. No user output is deleted. Benchmark errors make the
job fail visibly; completed siblings remain available.

    bash scripts/run_exp14.sh aggregate --root /shared/exp14

Outputs include summary.csv/json, per-layer groups and router mode, checkpoint
checksums, raw generations and official grader details, effective tokens,
unique-physical-expert routing histograms, stage timings, GPU hours, CUDA peak
allocated/reserved memory, sampled process GPU peak and host RSS peak.
GPU hours means one reserved GPU's elapsed wall-time including host staging.
Cold and shared-statistics/cached costs are labelled separately; REAM's required
sequential recalibration is not falsely treated as a removable shared cache.

## Adaptive code/math evaluation

With --adaptive-eval, a GPU worker only compresses/generates, then releases its
card. Independent CPU workers run math verification and isolated official code
grading. Default: at most TWO graders/node; each code container still gets
four CPUs / 8 GiB. Grader failure is visible, not converted to zero or ignored.

Ready compression configurations have priority over evaluation helpers.
Otherwise an idle GPU loads the SAME checkpoint and claims missing generation
chunks, 32 benchmark samples/chunk. Candidates are ordered by estimated remaining
generation time (completed chunk time/sample times remaining samples; sample
count fallback before measurements). At most TWO helper replicas/checkpoint
across all nodes: primary plus helpers can use three cards at the tail, but each
process still sees exactly one card. This does not alter K, P, prompts, precision,
greedy sampling, input/new-token limits or grader definitions. There is no
early stopping, truncated benchmark or shortened generation limit.

The primary and helpers use shared chunk locks, canonical sample IDs, atomic
outputs and the same frozen identity. A killed helper releases its locks;
completed samples inside a partial chunk are reused without generating them
again. Children retain ownership claims if the launcher alone is interrupted.
Restart with the same launch command. These guarantees require a POSIX-locking
shared filesystem; without --steal the same mechanism works inside one node's
local root, but another node cannot help that local root.

CPU grading starts when ALL chunks of a benchmark are complete; a job remains
waiting_evaluation until the primary GPU phase has exited AND every required
benchmark has a validated score. MC8 remains on its original GPU worker.
Finished benchmark results do not wait for slower siblings to be saved.
Helper resource records include load time/GPU hours and routing calls separately;
do not interpret primary GPU time alone as total adaptive evaluation cost.
Replica loading and shared-storage traffic can offset the benefit for short
tails; this policy is a throughput opportunity, not a guaranteed speedup.

Cold Efficiency is exempt: A drains its own jobs and CPU graders, launches a
single isolated card, and refuses new helpers/graders during that measurement.
The other nodes can continue working. Scheduling defaults are recorded in
scheduler-A/B/C/D.json; changing scientific settings requires a new prepared root.

## Checkpoint reload and verification

Physical expert counts are K, while config keeps source E so metadata can build
the correct source-router shape. exp14.json defines physical topology and
checksums. Use the supported loader:

    from reap.exp14.checkpoint import load
    model, tokenizer, path = load("/path/to/checkpoint", device="cuda")

Local verification:

    PYTHONPATH=src TOKENIZERS_PARALLELISM=false python -m pytest tests/test_exp14.py tests/test_exp14_adaptive.py tests/test_pruning_metrics.py -q -p no:cacheprovider
    bash -n scripts/run_exp14.sh

Before formal H20 release, build the grader image and run the same tests there,
verify source generation on all four models, test a small calibration subset
through layer compression/reload, then one full configuration per precision.
Check finite logits, source quantization layout, actual peak RAM/VRAM and disk.
Only then release all 32 slots. CPU tests do not establish full-model GPU
throughput, actual benchmark accuracy, or the cluster's Docker availability.
