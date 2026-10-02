#!/bin/zsh
# Start the RoBERTa four-epoch replicate (HumAID-19, seed 42) once fewer than 4 of this study's training
# processes are running. The extras chain skips any run this job has claimed or finished.
cd "$(dirname "$0")/.."
while [ "$(pgrep -f 'code/run_protocol.py' | wc -l | tr -d ' ')" -ge 4 ]; do
  echo "$(date -u +%FT%TZ) waiting for a free slot"; sleep 180
done
echo "$(date -u +%FT%TZ) starting RoBERTa four-epoch replicate"
HF_HUB_OFFLINE=1 ../../.venv/bin/python code/run_protocol.py --model data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b --corpus humaid19 --seed 42 --epochs 4 --plan confirm_ep4 --node_id n-a2-ep4-roberta
echo "stream roberta ep4 finished"
