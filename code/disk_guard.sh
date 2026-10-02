#!/bin/zsh
# Stops this study's training streams (all resumable) if free disk space falls below 10 GB.
cd "$(dirname "$0")/.."
while true; do
  FREE=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
  echo "$(date -u +%FT%TZ) free_gb=$FREE" >> experiments/logs/disk_guard.log
  if [ "$FREE" -lt 10 ]; then
    echo "$(date -u +%FT%TZ) STOPPING streams: free_gb=$FREE < 10" >> experiments/logs/disk_guard.log
    tmux kill-window -t crisis:conv 2>/dev/null
    pkill -f "code/run_confirm.sh"
    pkill -f "code/run_extras.sh"
    pkill -f "code/run_convergence.sh"
    sleep 2
    pkill -f "code/run_protocol.py"
    pkill -f "code/convergence.py"
    exit 0
  fi
  sleep 300
done
