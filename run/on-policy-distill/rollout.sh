# --vllm_data_parallel_size 2 \

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 VLLM_USE_MODELSCOPE=False \
OMP_NUM_THREADS=8 \
swift rollout \
    --model Qwen/Qwen3-4B-Base \
    --use_hf true \
    --vllm_max_model_len 24192 \
    --vllm_tensor_parallel_size 2 \
    --port 8000