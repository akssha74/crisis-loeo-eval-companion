#!/bin/zsh
# Source-validation learning curves (L26): development-visible folds only, no target scoring.
set -e
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=code
for M in data/models/distilbert-base-uncased/12040accade4e8a0f71eabdb258fecc2e7e948be \
         data/models/roberta-base/e2da8e2f811d1448a5b465c236feacd80ffbac7b; do
  $PY code/convergence.py --corpus humaid19 --model $M \
      --events hurricane_harvey_2017,puebla_mexico_earthquake_2017,srilanka_floods_2017
  $PY code/convergence.py --corpus crisislext26 --model $M --events 2013_boston_bombings,2012_typhoon_pablo
done
echo "convergence finished"
