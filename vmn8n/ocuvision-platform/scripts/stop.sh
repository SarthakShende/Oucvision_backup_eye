#!/usr/bin/env bash
pkill -f "uvicorn automation.vm_agent" 2>/dev/null || true
echo "stopped"
