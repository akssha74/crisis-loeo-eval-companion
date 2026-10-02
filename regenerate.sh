#!/bin/sh
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY code/verify_training_ids.py
$PY code/revision_x9.py
$PY code/run_zeroshot.py score
$PY code/check_note_values.py
