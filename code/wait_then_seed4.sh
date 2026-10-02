#!/bin/zsh
# Start RoBERTa HumAID-19 seed 4 once both RoBERTa CrisisLexT26 processes have finished (keeps GPU concurrency flat).
cd "$(dirname "$0")/.."
while pgrep -f "run_confirm.sh roberta crisislext26" >/dev/null || pgrep -f "corpus crisislext26 --model data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b --seed 2" >/dev/null; do
  echo "$(date -u +%FT%TZ) waiting for RoBERTa CrisisLexT26 to finish"; sleep 120
done
echo "$(date -u +%FT%TZ) starting RoBERTa HumAID seed 4 helper"
HF_HUB_OFFLINE=1 ../../.venv/bin/python code/run_protocol.py --corpus humaid19 --model data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b --seed 4 --plan confirm --node_id n-confirm-humaid19
echo "stream roberta humaid19 seed4 finished"
