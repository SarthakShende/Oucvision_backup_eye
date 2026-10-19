#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null
export OCUVISION_API_TOKEN="${OCUVISION_API_TOKEN:-change_me}"
uvicorn automation.vm_agent:app --host 127.0.0.1 --port 8000 --workers 1
