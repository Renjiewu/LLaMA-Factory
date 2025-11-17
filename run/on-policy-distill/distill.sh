# [INFO:swift] model: Qwen3ForCausalLM(
#   (model): Qwen3Model(
#     (embed_tokens): Embedding(151936, 2560)
#     (layers): ModuleList(
#       (0-35): 36 x Qwen3DecoderLayer(
#         (self_attn): Qwen3Attention(
#           (q_proj): Linear(in_features=2560, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
#           (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
#           (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
#         )
#         (mlp): Qwen3MLP(
#           (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
#           (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
#           (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
#         (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
#       )
#     )
#     (norm): Qwen3RMSNorm((2560,), eps=1e-06)
#     (rotary_emb): Qwen3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=2560, out_features=151936, bias=False)
# )

# --train_type full \
# --train_type lora \
# --lora_rank 8 \
# --lora_alpha 32 \
# --target_modules up_proj,down_proj,gate_proj \
# --attn_impl flash_attn \

NCCL_P2P_DISABLE=1 \
NPROC_PER_NODE=4 \
OMP_NUM_THREADS=8 \
VLLM_USE_MODELSCOPE=False \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-4B-Base \
    --teacher_model Qwen/Qwen3-32B \
    --use_hf true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 512 \
    --target_modules all-linear \
    --dataset open-thoughts/OpenThoughts3-1.2M#10000 \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --report_to wandb \
    --max_length 16000 \
    --truncation_strategy right \
    --max_completion_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --attn_impl flash_attn \
    --dataset_num_proc 16 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000