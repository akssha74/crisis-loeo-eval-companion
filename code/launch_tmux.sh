#!/bin/zsh
# Launch every remaining training job in a detached tmux session so it survives editor/terminal restarts.
# All jobs are resumable (finished runs are skipped).
#   zsh code/launch_tmux.sh            attach later with: tmux attach -t crisis
set -e
cd "$(dirname "$0")/.."
S=crisis
tmux has-session -t $S 2>/dev/null && { echo "session $S already running"; exit 1; }
L=experiments/logs
tmux new-session -d -s $S -n dH "zsh code/run_confirm.sh distilbert humaid19 >> $L/confirm_distilbert_humaid19.log 2>&1; zsh code/run_extras.sh roberta >> $L/extras_roberta.log 2>&1"
tmux new-window -t $S -n rH "zsh code/run_confirm.sh roberta humaid19 >> $L/confirm_roberta_humaid19.log 2>&1"
tmux new-window -t $S -n dC "zsh code/run_confirm.sh distilbert crisislext26 >> $L/confirm_distilbert_crisislext26.log 2>&1; zsh code/run_extras.sh distilbert >> $L/extras_distilbert.log 2>&1"
tmux new-window -t $S -n rC "zsh code/run_confirm.sh roberta crisislext26 >> $L/confirm_roberta_crisislext26.log 2>&1"
tmux new-window -t $S -n conv "zsh code/run_convergence.sh >> $L/convergence.log 2>&1"
tmux new-window -t $S -n guard "zsh code/disk_guard.sh"
tmux list-windows -t $S
