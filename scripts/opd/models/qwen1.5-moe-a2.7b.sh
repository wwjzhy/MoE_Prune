# Megatron MODEL_ARGS for Qwen1.5-MoE-A2.7B / Qwen2MoeForCausalLM.
# HF: hidden=2048, 24 layers, 16/16 attn heads (no GQA), 60 routed experts top-4,
# shared expert 5632 + shared_expert_gate. Qwen2-style QKV bias, no QK-norm.

NLAYERS=24
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# Qwen2-MoE: no GQA (16/16), QKV bias, no QK-norm. Shared expert + gate.
MODEL_ARGS=(
   --swiglu
   --num-layers 24
   --hidden-size 2048
   --ffn-hidden-size 5632
   --num-attention-heads 16
   --kv-channels 128
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --untie-embeddings-and-output-weights
   --position-embedding-type rope
   --rotary-percent 1.0

   --moe-ffn-hidden-size 1408
   --moe-shared-expert-intermediate-size 5632
   --moe-shared-expert-gate
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 4
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 60
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-aux-loss-coeff 0
)
