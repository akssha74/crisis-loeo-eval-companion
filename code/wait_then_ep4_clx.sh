#!/bin/zsh
# Exploratory X1: DistilBERT four-epoch replicate on CrisisLexT26, seed 42; starts once fewer than 3 of this
# study's training processes are running.
cd "$(dirname "$0")/.."
while [ "$(pgrep -f 'code/run_protocol.py' | wc -l | tr -d ' ')" -ge 3 ]; do
  echo "$(date -u +%FT%TZ) waiting for a free slot"; sleep 180
done
echo "$(date -u +%FT%TZ) starting DistilBERT four-epoch replicate on CrisisLexT26"
HF_HUB_OFFLINE=1 ../../.venv/bin/python code/run_protocol.py --model data/models/distilbert-base-uncased/12040accade4e8a0f71eabdb258fecc2e7e948be --corpus crisislext26 --seed 42 --epochs 4 --plan confirm_ep4 --node_id n-x1-ep4-clx
echo "stream distilbert ep4 crisislext26 finished"
