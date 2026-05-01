#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# pip install flash-attn
# pip install trl
# rm -rf /root/.cache/huggingface/hub/*
# ls -ld ~/.cache/huggingface/hub/
# rm -rf ~/.cache/huggingface/hub/models--kernels-community--triton_kernels
# cp -r /root/models--kernels-community--triton_kernels ~/.cache/huggingface/hub/models--kernels-community--triton_kernels
# pip install torchvision==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip uninstall -y vllm
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# # 这个变量是关键，它告诉程序直接去哪里找 triton_kernels
# export KERNELS_LOCAL_DIR="/root/triton_kernels"

python tagger/train/train_qlora.py \
  --model_name "${MODEL_NAME:?set MODEL_NAME}" \
  --data_path "${DATA_PATH:?set DATA_PATH}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR}"
