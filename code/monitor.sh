#!/bin/zsh
# Read-only status monitor for the tmux training session. Prints a status block every INTERVAL seconds and an
# "ALERT" line when the session is gone, no training process is alive while runs remain, a stream log shows a
# failure, a stream has written nothing for STALL minutes, or free disk falls below 20 GB.
#   zsh code/monitor.sh [interval_seconds]
cd "$(dirname "$0")/.."
INTERVAL=${1:-600}
STALL=45
L=experiments/logs
while true; do
  echo "================ $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
  if tmux has-session -t crisis 2>/dev/null; then
    echo "tmux session 'crisis': $(tmux list-windows -t crisis -F '#W' | tr '\n' ' ')"
  else
    echo "ALERT tmux session 'crisis' is not running"
  fi
  NPROC=$(ps aux | grep -E "code/run_protocol.py|code/convergence.py" | grep -v grep | wc -l | tr -d ' ')
  echo "training processes alive: $NPROC"
  for f in $L/confirm_*_*.log $L/extras_*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c 'done exit=0' "$f")
    fails=$(grep -c 'FAILED' "$f")
    last=$(grep -E 'run=' "$f" | tail -1 | sed -E 's/.*run=([^ ]+).*/\1/')
    age=$(( ( $(date +%s) - $(stat -f %m "$f") ) / 60 ))
    printf "  %-40s done=%-4s last_run=%-45s log_age=%smin\n" "$(basename $f)" "$n" "$last" "$age"
    [ "$fails" -gt 0 ] && echo "ALERT $(basename $f) reports $fails FAILED run(s)"
    if [[ "$(basename $f)" == confirm_* ]] && [ "$age" -gt "$STALL" ] && ! grep -q "stream .* finished" "$f"; then
      echo "ALERT $(basename $f) has written nothing for ${age} min"
    fi
  done
  echo "  convergence epochs logged: $(grep -c '"epoch"' $L/convergence.log 2>/dev/null)"
  FREE=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
  echo "free disk: ${FREE} GB   swap: $(sysctl -n vm.swapusage | awk '{print $6}') used"
  [ "$FREE" -lt 20 ] && echo "ALERT free disk ${FREE} GB < 20 GB"
  if [ "$NPROC" -eq 0 ] && ! grep -q "extras roberta finished" $L/extras_roberta.log 2>/dev/null; then
    echo "ALERT no training process alive but work remains"
  fi
  sleep $INTERVAL
done
