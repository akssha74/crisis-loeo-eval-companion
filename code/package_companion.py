"""Assemble Online Resource 2, the evaluation companion, under release/online-resource-2 and zip it. The package
holds no tweet text: metadata, split manifests, whole-event LOEO predictions and run records, code, pinned
environments, the derived files the note reads, and the values printed in the note.

    python code/package_companion.py
"""
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

OUT = os.path.join("release", "online-resource-2")
FILE_GLOBS = ["code/*.py", "code/*.sh", "environment/*.txt", "experiments/meta/*.tsv", "experiments/splits/*.json",
              "experiments/derived/**/*.json", "experiments/run-ledger.jsonl", "research/preregistration.md",
              "research/pilot-preregistration.md", "research/metric-practice-audit-2026-09-29.*"]
RUN_GLOBS = ["experiments/runs/confirm/*/*/seed*/loeo__*", "experiments/runs/confirm/*/*/seed*/loeo_full__*",
             "experiments/runs/confirm_ep4/*/*/seed*/loeo__*", "experiments/runs/x3_tfidf/*/*/seed*/loeo__*",
             "experiments/runs/x10_zeroshot/*/*"]
TEXT_COLUMNS = {"text", "text_norm"}

README = """# Evaluation companion for event-held-out HumAID and CrisisLexT26 (Online Resource 2)

Companion to "A Reproducible Evaluation Companion for Event-Held-Out Crisis-Message Classification" (Sharma and
Prasad, Language Resources and Evaluation, Project Note). It contains no tweet text.

## Contents
- `experiments/meta/`: tweet identifier, event, hazard and label of every message (99,859 messages, 45 events).
- `experiments/splits/`: split manifests (event lists, random and temporal test identifiers, corpus hashes).
- `experiments/runs/`: whole-event leave-one-event-out predictions (`preds.tsv.gz`: tweet_id, event, y_true,
  y_pred, class probabilities) and run records (`run.json`: settings, versions, SHA-256 of the training
  identifiers) of every reference system: `confirm` (two epochs, and `loeo_full` without the 6,000 cap),
  `confirm_ep4` (four epochs), `x3_tfidf` (TF-IDF + logistic regression), `x10_zeroshot` (BART-large-MNLI).
- `code/companion_eval.py`: the evaluator. `code/verify_training_ids.py`: rebuilds every whole-event training
  sample from the metadata and checks its hash. `code/revision_x9.py`, `code/run_zeroshot.py score`: the note's
  analyses. `code/check_note_values.py`: compares recomputed values with `note_values.json`, the values printed
  in the note. The remaining scripts rebuild the corpora from the providers (`fetch_*.py`, `prepare_*.py`),
  train the reference systems (`run_protocol.py`, `run_tfidf.py`, `run_zeroshot.py predict`) and produce the
  analyses of Online Resource 1.
- `experiments/derived/`: outputs of every analysis; `experiments/run-ledger.jsonl`: one record per training run.
- `research/`: the analysis plan with its amendments, exploratory additions and deviations log, and the coded
  literature reading.
- `environment/`: pinned Python environments.

## Scoring a prediction file
    python code/companion_eval.py --convention P --out out/ \\
        --meta experiments/meta/corpus_crisislext26_meta.tsv --preds "my_run/*.tsv.gz"

The convention (P: classes present in the event; D: classes in the true labels or predictions; O: every class)
has no default. Files covering only some events need `--partial`. Outputs: `scores.json` and
`event_class_counts.tsv`.

## Regenerating the note
    sh regenerate.sh

## Licences
Code: MIT (`LICENSE`). Label-bearing files (`experiments/meta`, `experiments/runs`): CC BY-NC-SA 4.0, the licence
of the HumAID-events release; the CrisisLexT26 labels come from the MIT-licensed CrisisLex repository. Tweet text
must be obtained from the providers under their terms.
"""

REGENERATE = """#!/bin/sh
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY code/verify_training_ids.py
$PY code/revision_x9.py
$PY code/run_zeroshot.py score
$PY code/check_note_values.py
"""

LICENSE = """MIT License

Copyright (c) 2026 Akshay Sharma and Lalji Prasad

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def copy(src):
    dst = os.path.join(OUT, src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):
        if name != ".git":
            p = os.path.join(OUT, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for pattern in FILE_GLOBS:
        for f in sorted(glob.glob(pattern, recursive=True)):
            copy(f)
    n_runs = 0
    for pattern in RUN_GLOBS:
        for d in sorted(glob.glob(pattern)):
            for name in ("preds.tsv.gz", "run.json"):
                if os.path.exists(os.path.join(d, name)):
                    copy(os.path.join(d, name))
            n_runs += 1
    for f in glob.glob(os.path.join(OUT, "experiments", "meta", "*.tsv")):
        header = set(open(f).readline().rstrip("\n").split("\t"))
        assert not header & TEXT_COLUMNS, f"text column in {f}"
    open(os.path.join(OUT, "README.md"), "w").write(README)
    open(os.path.join(OUT, "regenerate.sh"), "w").write(REGENERATE)
    open(os.path.join(OUT, "LICENSE"), "w").write(LICENSE)
    subprocess.run([sys.executable, "code/check_note_values.py", "--write", "--values",
                    os.path.join(OUT, "note_values.json")], check=True)
    files = [f for f in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)) if os.path.isfile(f)]
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as m:
        for f in files:
            m.write(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {os.path.relpath(f, OUT)}\n")
    with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + [os.path.join(OUT, "MANIFEST.sha256")]:
            z.write(f, os.path.join("online-resource-2", os.path.relpath(f, OUT)))
    print(f"packaged {n_runs} run directories into {OUT} and {OUT}.zip")


if __name__ == "__main__":
    main()
