
###
 # @Author: moyueheng moyueheng@126.com
 # @Date: 2024-12-21 00:39:56
 # @LastEditors: moyueheng moyueheng@126.com
 # @LastEditTime: 2024-12-21 22:50:50
 # @FilePath: /LLaMA-Factory/run/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

PYTHONPATH=/app CUDA_VISIBLE_DEVICES=5 NCCL_P2P_DISABLE=1 vllm serve \
"ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2" \
--quantization gptq \
--dtype float16 \
--max-model-len 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.99 \
--max_num_seqs 4 \
--host 0.0.0.0 \
--port 7860