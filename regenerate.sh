#!/bin/sh
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY code/test_companion_eval.py
$PY code/verify_training_ids.py
$PY code/revision_x9.py
$PY code/run_zeroshot.py score
$PY code/revision_x11.py
$PY code/revision_x12.py
$PY code/revision_x13.py
$PY code/check_note_values.py
