#!/usr/bin/env bash

# 在这里设置你想使用的 GPU 编号，比如"0"或"0,1"等
export CUDA_VISIBLE_DEVICES="2,1"

# 计算有多少张 GPU
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Number of GPUs: $NUM_GPUS"

# 启动服务：tp-size 会根据 GPU 张数自动确定
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --tp-size ${NUM_GPUS} \
  --port 40000 \
  --mem-fraction-static 0.6