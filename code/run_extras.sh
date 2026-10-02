#!/bin/zsh
# Amendment A2 added arms for one encoder, in the registered order; resumable.
#   zsh code/run_extras.sh distilbert      zsh code/run_extras.sh roberta
set -e
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1
case $1 in
  distilbert) M=data/models/distilbert-base-uncased/12040accade4e8a0f71eabdb258fecc2e7e948be ;;
  roberta)    M=data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b ;;
  *) echo "usage: $0 distilbert|roberta"; exit 2 ;;
esac
R() { $PY code/run_protocol.py --model $M --plan confirm "$@"; }
for c in humaid19 crisislext26; do R --corpus $c --seed 42 --only id_random_dedup,id_temporal_dedup --node_id n-confirm-$c; done
for s in 42 1 2; do for c in humaid19 crisislext26; do R --corpus $c --seed $s --extras fuzzy --only id_random_fuzzydedup --node_id n-a2-fuzzy; done; done
for c in humaid19 crisislext26; do R --corpus $c --seed 42 --extras divk --only loeok: --node_id n-a2-divk; done
for s in 42 1; do for c in humaid19 crisislext26; do R --corpus $c --seed $s --extras mask --only id_random_mask,loeo_mask: --node_id n-a2-mask; done; done
if [[ $1 == distilbert ]]; then
  for c in crisislext26 humaid19; do R --corpus $c --seed 42 --extras full --only id_random_full,loeo_full: --node_id n-a2-full; done
else
  $PY code/run_protocol.py --model $M --corpus humaid19 --seed 42 --epochs 4 --plan confirm_ep4 --node_id n-a2-ep4-roberta
fi
echo "extras $1 finished"
