#/bin/base -e
###
 # @Author: moyueheng moyueheng@126.com
 # @Date: 2024-12-20 21:02:02
 # @LastEditors: moyueheng moyueheng@126.com
 # @LastEditTime: 2024-12-21 14:55:31
 # @FilePath: /LLaMA-Factory/init.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
INSTALL_BNB=false
INSTALL_VLLM=true
INSTALL_DEEPSPEED=true
INSTALL_FLASHATTN=true
INSTALL_LIGER_KERNEL=false
INSTALL_HQQ=false
INSTALL_EETQ=false
PIP_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
#PIP_INDEX=https://pypi.org/simple
# pip list
# source ~/miniconda3/etc/profile.d/conda.sh
# conda info --envs
conda activate llama-factory-wrj
pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

export LD_LIBRARY_PATH="/jinx/tools/yes/envs/llama-factory-wrj/lib/python3.12/site-packages/nvidia/nvjitlink/lib":$LD_LIBRARY_PATH
export HF_HOME=/jinx/cache/huggingface
EXTRA_PACKAGES="metrics"; \
if [ "$INSTALL_BNB" == "true" ]; then \
    EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes"; \
fi; \
if [ "$INSTALL_DEEPSPEED" == "true" ]; then \
    EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
fi; \
if [ "$INSTALL_LIGER_KERNEL" == "true" ]; then \
    EXTRA_PACKAGES="${EXTRA_PACKAGES},liger-kernel"; \
fi; \
if [ "$INSTALL_VLLM" == "true" ]; then \
    EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
fi; \
if [ "$INSTALL_HQQ" == "true" ]; then \
    EXTRA_PACKAGES="${EXTRA_PACKAGES},hqq"; \
fi; \
if [ "$INSTALL_EETQ" == "true" ]; then \
    EXTRA_PACKAGES="${EXTRA_PACKAGES},eetq"; \
fi; \
pip install -e ".[$EXTRA_PACKAGES]"
pip uninstall -y transformer-engine flash-attn && \
    if [ "$INSTALL_FLASHATTN" == "true" ]; then \
        pip uninstall -y ninja && pip install ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi
pip install webdav4