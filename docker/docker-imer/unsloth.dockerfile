# Default use the NVIDIA official image with PyTorch 2.3.0
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
# ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
# ARG BASE_IMAGE=vllm/vllm-openai:v0.7.3
# ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.8.4
# ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.8.5
# ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.9.0
# ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.9.1
# ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.9.2
# ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.10.0
ARG BASE_IMAGE=docker.1ms.run/vllm/vllm-openai:v0.10.2
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]
RUN ln -s /usr/bin/python3 /usr/bin/python

# Define environments
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TORCH_CUDA_ARCH_LIST=8.0+PTX
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /app


# Install the requirements

RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install --upgrade pip && \
    python -m pip install unsloth llmcompressor webdav4 wandb



# Set up volumes
VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
