#!/bin/bash

# 첫 번째 인자를 GPU 번호로 사용
GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    echo "사용법: $0 <GPU_ID>"
    echo "예시:  $0 0"
    echo "       $0 1"
    exit 1
fi

echo "사용할 GPU: $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID ./isaaclab.sh \
    -p scripts/dofbot/train_rl.py \
    --num_envs 64 \
    --headless \
    --enable_cameras
