
###
 # @Author: moyueheng moyueheng@126.com
 # @Date: 2024-12-21 00:39:56
 # @LastEditors: moyueheng moyueheng@126.com
 # @LastEditTime: 2024-12-21 22:50:50
 # @FilePath: /LLaMA-Factory/run/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# # --served-model-name "my_model_name"
# "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
# "BAAI/bge-m3"
PYTHONPATH=/app CUDA_VISIBLE_DEVICES=7 NCCL_P2P_DISABLE=1 vllm serve \
    "Qwen/Qwen3-Embedding-0.6B" \
    --task embedding \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.4 \
    --max_num_seqs 48 \
    --host 0.0.0.0 \
    --port 8001