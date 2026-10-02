#!/bin/zsh
# One confirmation stream (model x corpus), resumable: finished runs are skipped.
#   zsh code/run_confirm.sh distilbert humaid19     zsh code/run_confirm.sh roberta crisislext26
set -e
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1
case $1 in
  distilbert) M=data/models/distilbert-base-uncased/12040accade4e8a0f71eabdb258fecc2e7e948be ;;
  roberta)    M=data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b ;;
  *) echo "usage: $0 distilbert|roberta humaid19|crisislext26"; exit 2 ;;
esac
case $2 in
  humaid19)     SEEDS=(42 1 2 3 4) ;;
  crisislext26) SEEDS=(42 1 2) ;;
  *) echo "usage: $0 distilbert|roberta humaid19|crisislext26"; exit 2 ;;
esac
for s in $SEEDS; do
  $PY code/run_protocol.py --corpus $2 --model $M --seed $s --plan confirm --node_id n-confirm-$2
done
if [[ $1 == distilbert && $2 == crisislext26 ]]; then
  $PY code/run_protocol.py --corpus humaid19 --model $M --seed 42 --epochs 4 --plan confirm_ep4 --node_id n-confirm-ep4
fi
echo "stream $1 $2 finished"
