#!/bin/bash

echo "Isaac Lab 관련 프로세스를 검색 중..."

# isaaclab 관련 프로세스 PID 수집
PIDS=$(ps -ef | grep -E 'isaaclab.sh|isaac_sim' | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "종료할 Isaac Lab 프로세스가 없습니다."
else
    # PID 중 가장 큰 값 선택(동시에 여러 프로세스 실행 중일 때, 제일 최근에 실행한거만 종료)
    TARGET_PID=$(echo "$PIDS" | sort -n | tail -1)

    echo "종료할 PID (가장 최근 것으로 추정): $TARGET_PID"
    kill -9 "$TARGET_PID" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "PID $TARGET_PID 종료 완료."
    else
        echo "PID $TARGET_PID 종료 실패."
    fi
fi

# 좀비 프로세스(defunct) 확인 (부모가 살아있다면 의미 없음)
ZOMBIES=$(ps -ef | grep defunct | grep -v grep | awk '{print $2}')
if [ -n "$ZOMBIES" ]; then
    echo "남아있는 좀비 프로세스가 있습니다 (직접 부모 프로세스 확인 필요):"
    echo "$ZOMBIES"
fi
