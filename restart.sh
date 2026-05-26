#!/bin/bash
# 精准 kill，避免杀掉 shell 自身
PID_FILE=/tmp/va.pid

if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID"
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

# 也清理残留（只杀同路径进程，排除 grep/bash 自身）
pkill -f "uvicorn src.voice_assistant.web" 2>/dev/null
sleep 1

cd /home/void/Projects/voice-assistant
nohup env PYTHONUNBUFFERED=1 uv run python -u -m uvicorn src.voice_assistant.web:app \
  --host 0.0.0.0 --port 8765 > /tmp/va.log 2>&1 &

echo $! > "$PID_FILE"
echo "Started PID=$(cat $PID_FILE)"
sleep 2
curl -s http://localhost:8765/health && echo ""
