#!/bin/zsh
cd "$(dirname "$0")"
LOG="../logs/wfcap-$(date +%Y%m%d-%H%M%S).log"
mkdir -p ../logs
./.venv/bin/python -m wfcap.record "$@" 2>&1 | tee "$LOG"
cp "$LOG" ../logs/wfcap-latest.log
