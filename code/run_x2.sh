#!/bin/zsh
# Exploratory X2: four-epoch HumAID-19 runs of the conditions behind E0-E4 and MC for one further seed.
#   zsh code/run_x2.sh distilbert 1        zsh code/run_x2.sh roberta 1 --after-full-pool
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
case $1 in
  distilbert) M=data/models/distilbert-base-uncased/12040accade4e8a0f71eabdb258fecc2e7e948be ;;
  roberta)    M=data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b ;;
  *) echo "usage: $0 distilbert|roberta SEED [--after-full-pool]"; exit 2 ;;
esac
if [[ $3 == --after-full-pool ]]; then
  while pgrep -f "^../../.venv/bin/python code/run_protocol.py.*loeo_full:" > /dev/null; do sleep 120; done
fi
../../.venv/bin/python code/run_protocol.py --corpus humaid19 --model $M --seed $2 --epochs 4 --plan confirm_ep4 \
  --only id_random,id_temporal,id_random_dedup,id_temporal_dedup,loeo: --node_id n-x2-ep4
echo "X2 $1 seed $2 finished"
