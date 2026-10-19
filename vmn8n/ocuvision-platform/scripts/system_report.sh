#!/usr/bin/env bash
echo "=== SYSTEM ==="
uname -a
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu,temperature.gpu --format=csv 2>/dev/null || echo "no nvidia-smi"
free -h
df -h /
nproc
ps aux | grep -v grep | grep -c uvicorn || true
