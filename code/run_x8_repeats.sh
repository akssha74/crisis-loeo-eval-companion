#!/bin/zsh
# Exploratory X8(a): identical-seed repeats of the two-epoch DistilBERT HumAID-19 runs behind the protected E2
# (seed 42), trained twice more with the same run keys, so that differences between replicates are MPS
# non-determinism only.
#   zsh code/run_x8_repeats.sh
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
M=data/models/distilbert-base-uncased/12040accade4e8a0f71eabdb258fecc2e7e948be
ONLY=id_random,loeo:california_wildfires_2018,loeo:hurricane_dorian_2019,loeo:hurricane_florence_2018,loeo:kerala_floods_2018,loeo:midwestern_us_floods_2019,loeo:pakistan_earthquake_2019
for rep in x8_repeat_a x8_repeat_b; do
  ../../.venv/bin/python code/run_protocol.py --corpus humaid19 --model $M --seed 42 --epochs 2 --plan $rep \
    --only $ONLY --node_id n-x8-repeat
done
echo "X8_REPEATS_DONE"
