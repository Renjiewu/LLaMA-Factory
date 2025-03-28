
###
 # @Author: moyueheng moyueheng@126.com
 # @Date: 2024-12-21 00:39:56
 # @LastEditors: moyueheng moyueheng@126.com
 # @LastEditTime: 2024-12-21 22:50:50
 # @FilePath: /LLaMA-Factory/run/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# export HF_HOME=/jinx/cache/huggingface
# export LD_LIBRARY_PATH='/jinx/tools/yes/envs/llama-factory-wrj/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH'

# PYTHONPATH=/jinx/wrj/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/qwen2vl_lora_sft_img_cot_r64_ep3.yaml

# rm -rf output/qwen2_vl-7b/lora/sft-r128-split-img-noise/checkpoint-*

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/merge_lora/lora_r256_ep3.yaml

# rm -rf output/qwen2_vl-7b/merge-lora/lora_r256_ep3/checkpoint-*

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3.yaml

# rm -rf output/qwen2_vl-7b/lora/lora_r128_img_ep5/checkpoint-*

# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora-256.yaml

# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/2b/merge_dora_img.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/2b/merge_dora_dia.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/2b/merge_dora-256_img.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/2b/merge_dora-256_dia.yaml


# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/2b/dora/dora_r32_ep3_explain_img.yaml

# rm -rf output/qwen2_vl-2b/dora/dora_r32_ep3_explainv2_img/checkpoint-*

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/2b/dora/dora_r32_ep3_explain_dia.yaml

# rm -rf output/qwen2_vl-2b/dora/dora_r32_ep3_explainv2_dia/checkpoint-*

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/2b/dora/dora_r256_ep3_explain_img.yaml

# rm -rf output/qwen2_vl-2b/dora/dora_r256_ep3_explainv2_img/checkpoint-*

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/2b/dora/dora_r256_ep3_explain_dia.yaml

# rm -rf output/qwen2_vl-2b/dora/dora_r256_ep3_explainv2_dia/checkpoint-*

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory python scripts/vllm_infer_format.py
# PYTHONPATH=/jinx/wrj/LLaMA-Factory python scripts/vllm_infer_format.py

# PYTHONPATH=/root/autodl-tmp/LLaMA-Factory python scripts/vllm_infer_format.py

# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/full/qwen2vl_full_sft.yaml

# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3_explain_with_full.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep1_explain_with_full.yaml

# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_full_sft.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora_full-ep3.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora_full-ep1.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3_explain_with_add_dataset.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora_small.yaml

# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora-mask.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora-noise.yaml

# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3_explain_with_extra_data.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3_explainv3_with_extra_data.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3_explainv3_with_extra_datav2.yaml
# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025/dora/dora_r32_ep3_explainv3.yaml

# PYTHONPATH=/app CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora_v3.yaml


# shutdown -h now
# "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit" \ bitsandbytes
# "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ" \ awq
# "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF" \ gguf
# "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2" \ gptq
# "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
# "Qwen/QwQ-32B-AWQ" \ qwen
# --tensor_parallel_size 2 \
# --pipeline_parallel_size 2 \
#    -pp 4 \
#    -tp 1 \
# --num-scheduler-steps 8 \
# NCCL_P2P_DISABLE=1
PYTHONPATH=/app CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 vllm serve \
    "Qwen/QwQ-32B-AWQ" \
    --load-format auto \
    --quantization awq \
    --max-model-len 32000 \
    --dtype float16 \
    --gpu-memory-utilization 0.99 \
    --max_num_seqs 32 \
    -tp 4 \
    --host 0.0.0.0 \
    --port 8000