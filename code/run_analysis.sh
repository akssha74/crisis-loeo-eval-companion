#!/bin/zsh
# Frozen confirmation analyses, in order; each step logs to experiments/logs/analysis_<step>.log
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
set -e
$PY code/analyze_v2.py --plan confirm --skip_incomplete_extras 2>&1 | tee experiments/logs/analysis_v2_confirm.log
$PY code/analyze.py --plan confirm 2>&1 | tee experiments/logs/analysis_registered.log
$PY code/diagnostics.py --plan confirm 2>&1 | tee experiments/logs/analysis_diagnostics.log
$PY code/per_seed_table.py --plan confirm 2>&1 | tee experiments/logs/analysis_per_seed.log
$PY code/analyze_classes.py --plan confirm 2>&1 | tee experiments/logs/analysis_classes.log
echo ANALYSIS_DONE
