#!/bin/bash

echo "Isaac Lab 관련 프로세스를 검색 중..."

# 1. 'isaaclab.sh' 문구가 포함된 모든 프로세스의 PID 추출
# 2. 해당 PID와 그 PID를 부모(PPID)로 가지는 모든 프로세스를 찾아 kill
PIDS=$(ps -ef | grep -E 'isaaclab.sh|isaac_sim' | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "종료할 Isaac Lab 프로세스가 없습니다."
else
    echo "다음 PID들을 종료합니다: $PIDS"
    # -9 옵션으로 강제 종료
    echo $PIDS | xargs kill -9 2>/dev/null
    echo "모든 관련 프로세스가 종료되었습니다."
fi

# 좀비 프로세스(defunct) 확인용
ZOMBIES=$(ps -ef | grep defunct | grep -v grep | awk '{print $2}')
if [ ! -z "$ZOMBIES" ]; then
    echo "남아있는 좀비 프로세스 정리 시도..."
    echo $ZOMBIES | xargs kill -9 2>/dev/null
fi
