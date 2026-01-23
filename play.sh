#!/bin/bash
# 기본 라우트에 사용되는 로컬 IP 가져오기
PUBLIC_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}')

if [ -z "$PUBLIC_IP" ]; then
    echo "로컬 IP를 찾지 못했습니다."
    exit 1
fi

echo "사용할 PUBLIC_IP: $PUBLIC_IP"

LIVESTREAM=1 PUBLIC_IP=$PUBLIC_IP ./isaaclab.sh \
    -p scripts/dofbot/play_rl.py \
    --num_envs 16 \
    --enable_cameras
