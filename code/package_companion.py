"""Assemble Online Resource 2, the evaluation companion, under release/online-resource-2 and zip it. The package
holds no tweet text: metadata, split manifests, whole-event LOEO predictions and run records, code, pinned
environments, the derived files the note reads, and the values printed in the note.

    python code/package_companion.py
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

OUT = os.path.join("release", "online-resource-2")
INVENTORY = os.path.join("experiments", "derived", "companion_inventory.json")
FILE_GLOBS = ["code/*.py", "code/*.sh", "environment/*.txt", "experiments/meta/*.tsv", "experiments/splits/*.json",
              "experiments/systems.tsv", "experiments/derived/**/*.json", "experiments/run-ledger.jsonl",
              "research/preregistration.md", "research/pilot-preregistration.md",
              "research/metric-practice-audit-2026-09-29.*", "research/prelock/corpora_metadata.json"]
GROUPS = (("Event metadata", lambda f: f.startswith("experiments/meta/corpus_")),
          ("Duplicate lists", lambda f: f.startswith("experiments/meta/") and "duplicates" in f),
          ("Split manifests", lambda f: f.startswith("experiments/splits/")),
          ("System index", lambda f: f == "experiments/systems.tsv"),
          ("Predictions", lambda f: f.endswith("preds.tsv.gz")),
          ("Run records", lambda f: f.endswith("run.json") or f.endswith("run-ledger.jsonl")),
          ("Code and tests", lambda f: f.startswith("code/")),
          ("Derived results", lambda f: f.startswith("experiments/derived/")),
          ("Analysis plan and literature reading", lambda f: f.startswith("research/")),
          ("Environments", lambda f: f.startswith("environment/")))
RUN_GLOBS = ["experiments/runs/confirm/*/*/seed*/loeo__*", "experiments/runs/confirm/*/*/seed*/loeo_full__*",
             "experiments/runs/confirm_ep4/*/*/seed*/loeo__*", "experiments/runs/x3_tfidf/*/*/seed*/loeo__*",
             "experiments/runs/x10_zeroshot/*/*"]
TEXT_COLUMNS = {"text", "text_norm"}

README = """# Evaluation companion for event-held-out HumAID and CrisisLexT26 (Online Resource 2)

Companion to "A Reproducible Evaluation Companion for Event-Held-Out Crisis-Message Classification" (Sharma and
Prasad, Language Resources and Evaluation, Project Note). Version 1.2.0. It contains no tweet text.

## What it is for
Leave-one-event-out (LOEO) results can be summarised as one macro-F1 over all held-out messages (pooled) or as
the mean of per-event macro-F1, and per-event scores depend on which classes enter each event's average (the
label list). The evaluator scores a set of predictions under a label list that must be named, optionally after
removing event-class cells below a size threshold, reports both summaries, their difference A and its split, and
prints a signature for each of the three scores, for example

    loeo-macro-f1|summary:pooled|labels:P|zero-div:0|events:26|min-cell:0|meta:META12|v:1.2.0
    loeo-macro-f1|summary:per-event|labels:P|zero-div:0|events:26|min-cell:0|meta:META12|v:1.2.0
    loeo-macro-f1|summary:difference|labels:P|zero-div:0|events:26|min-cell:0|meta:META12|v:1.2.0

`zero-div:0` records that an empty class (no messages and no predictions of it) scores 0.

Label lists: P, the classes present in the event; D, the classes in the true labels or the predictions
(scikit-learn's default); O, every class of the corpus.

## Contents and file formats
- `experiments/meta/corpus_{humaid19,crisislext26}_meta.tsv`: one row per message, columns `tweet_id`, `event`,
  `hazard`, `label`. 76,484 HumAID messages in 19 events and 10 classes; 23,375 CrisisLexT26 messages in 26
  events and 6 classes.
- `experiments/meta/{cross,within}_event_duplicates_*.tsv`: `tweet_id`, `event` of messages whose normalised text
  also occurs in another event, or repeats an earlier message of the same event (computed from tweet text, which
  is not included).
- `experiments/splits/confirm_*.json`: event lists and corpus hashes.
- `experiments/systems.tsv`: the reference systems (`system`, `corpus`, `description`, `seeds`, `preds`, `meta`).
- `experiments/runs/<plan>/<corpus>/<model>/seed<k>/loeo__<event>/`: one directory per LOEO training run, with
  `preds.tsv.gz` (columns `tweet_id`, `event`, `y_true`, `y_pred`, then one probability column per class) and
  `run.json` (settings, code and data hashes, package versions, the SHA-256 of the sorted training identifiers).
  Plans: `confirm` (two epochs; `loeo_full__` runs train on 80% of the corpus), `confirm_ep4` (four epochs),
  `x3_tfidf` (TF-IDF + logistic regression), `x10_zeroshot` (BART-large-MNLI, one file per corpus).
- `experiments/run-ledger.jsonl`: one record per training run, failed runs included.
- `experiments/derived/`: outputs of every analysis in the note and Online Resource 1;
  `companion_inventory.json` counts the files and bytes of each part.
- `code/`: `companion_eval.py` (evaluator), `test_companion_eval.py` (tests), `verify_training_ids.py` (rebuilds
  every training sample from the metadata and checks its hash), `systems_index.py`, the analyses
  (`revision_x9.py`, `revision_x11.py`, `revision_x12.py`, `revision_x13.py`, `run_zeroshot.py score`),
  `check_note_values.py`
  (compares recomputed values with `note_values.json`, the values printed in the note), and the scripts that
  rebuild the corpora from the providers (`fetch_*.py`, `prepare_*.py`) and train the reference systems
  (`run_protocol.py`, `run_tfidf.py`, `run_zeroshot.py predict`).
- `research/`: the analysis plan with its exploratory additions and deviations log, the literature reading with
  its evidence, and corpus metadata. `environment/`: pinned Python environments.

The reference systems were trained for an earlier, unpublished experiment on cross-event transfer, whose plan is
also in `research/preregistration.md`; the companion holds their whole-event LOEO runs.

## Scoring
A reference system:

    python code/companion_eval.py --corpus crisislext26 --system tfidf-lr --seed 42 --convention P --out out

A new system: write one prediction file per held-out event (or one file for all), with at least the columns
`tweet_id`, `event`, `y_true` and `y_pred`, then

    python code/companion_eval.py --meta experiments/meta/corpus_crisislext26_meta.tsv \\
        --preds "my_run/*.tsv.gz" --convention P --out out

The evaluator rejects duplicated or unknown identifiers, events or true labels that disagree with the metadata,
predictions outside the corpus classes, and incomplete events; files covering only some events need
`--partial`, and the output is then marked partial. With `--min-cell 15`, the messages of every event-class
cell with fewer than 15 messages are removed before scoring (predictions of a removed class for other messages
still count as its false positives), and the threshold is recorded in the signatures. Outputs: `scores.json`
(pooled and per-event macro-F1, A, the split of A^P into T1 to T4 and its other orderings T2alt/T3alt and
T3c/T4c, the label-list shift, per-event values, the signatures and a hash of the inputs) and
`event_class_counts.tsv` (TP, FP, FN, support, precision, recall and F1 per event and class), from which any
other label list or summary can be computed. To add a reference system to the index, add its runs under
`experiments/runs` and a row to `experiments/systems.tsv`.

Other multi-event corpora can be scored in the same way: write a metadata file with the columns `tweet_id`,
`event`, `hazard` and `label` and pass it with `--meta`. Nothing in the evaluator is specific to the two corpora.

## Tests and regeneration
    python code/test_companion_eval.py      # or: python -m pytest code/test_companion_eval.py
    sh regenerate.sh                        # recompute every number of the note and compare

The tests check the three label lists against hand-computed values and scikit-learn, the closed-form shifts
between them, that T1 + T2 + T3 + T4 = A^P in every ordering, cell removal, the input validation and the
signatures.

## Versions and maintenance
Releases are tagged (`v1.0.0`, `v1.1.0`, `v1.2.0`) and archived in Software Heritage. A changed number in a release is
recorded in `CHANGELOG.md`; the evaluator version is printed in every signature. Problems can be reported
through the issue tracker of https://github.com/akssha74/crisis-loeo-eval-companion, which the first author
maintains.

## Licences
Code: MIT (`LICENSE`). Files that carry labels (`experiments/meta`, `experiments/runs`): CC BY-NC-SA 4.0. The
Hugging Face release `QCRI/HumAID-events` declares CC BY-NC-SA 4.0 in its metadata and CC BY-NC 4.0 in its card
text; the share-alike licence meets the terms of both. The CrisisLexT26 labels come from the CrisisLex repository
(https://github.com/sajao/CrisisLex, commit d67cddd5), released under the MIT licence; its copyright and permission
notice is reproduced in `LICENSE-CrisisLex`. Tweet text must be obtained from the providers under their terms.
"""

CHANGELOG = """# Changelog

## v1.2.0 (2026-10-02)
- Evaluator: one signature per score, naming the summary (`summary:pooled`, `summary:per-event`,
  `summary:difference`) and the cell threshold (`min-cell:`); the v1.1.0 signature named neither the summary nor
  a threshold, and its `zero-division:0` field is now written `zero-div:0`. No score changes.
- Evaluator: `--min-cell t` removes event-class cells with fewer than t messages before scoring.
- Evaluator: the split of A^P also reports the ordering that applies class weights first (T3c, T4c).
- Analyses X13 (`code/revision_x13.py`): the third ordering, a fixed-rate model, comparisons across summaries,
  and a check of `--min-cell 15` against the X11 threshold analysis.

## v1.1.0 (2026-10-02)
- Evaluator: `--corpus/--system/--seed` scores a reference system listed in `experiments/systems.tsv`; every result
  carries a signature (label list, zero-division rule, events, metadata hash, evaluator version).
- Tests (`code/test_companion_eval.py`) and a file inventory (`experiments/derived/companion_inventory.json`).
- Four-epoch HumAID seed 4 completed (27 runs that were missing from v1.0.0; see the deviations log in
  `research/preregistration.md`); every four-epoch HumAID value now averages five seeds.
- Analyses X11 and X12 (`code/revision_x11.py`, `code/revision_x12.py`) and the within-event duplicate lists.

## v1.0.0 (2026-10-02)
- First release.
"""

CRISISLEX_LICENSE = """The MIT License (MIT)

Copyright (c) 2014 Alexandra Olteanu, Carlos Castillo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

REGENERATE = """#!/bin/sh
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


def planned():
    files = {f for pattern in FILE_GLOBS for f in glob.glob(pattern, recursive=True)} - {INVENTORY}
    n_runs = 0
    for pattern in RUN_GLOBS:
        for d in sorted(glob.glob(pattern)):
            files |= {os.path.join(d, n) for n in ("preds.tsv.gz", "run.json") if os.path.exists(os.path.join(d, n))}
            n_runs += 1
    return sorted(files), n_runs


def inventory(files):
    groups = []
    for name, match in GROUPS:
        sel = [f for f in files if match(f)]
        groups.append({"name": name, "files": len(sel), "bytes": sum(os.path.getsize(f) for f in sel)})
    assert sum(g["files"] for g in groups) == len(files), "a packaged file belongs to no inventory group"
    return {"groups": groups, "total_files": len(files), "total_bytes": sum(os.path.getsize(f) for f in files),
            "prediction_files": sum(f.endswith("preds.tsv.gz") for f in files)}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):
        if name != ".git":
            p = os.path.join(OUT, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    files, n_runs = planned()
    json.dump(inventory(files), open(INVENTORY, "w"), indent=1)
    for f in files + [INVENTORY]:
        copy(f)
    for f in glob.glob(os.path.join(OUT, "experiments", "meta", "*.tsv")):
        header = set(open(f).readline().rstrip("\n").split("\t"))
        assert not header & TEXT_COLUMNS, f"text column in {f}"
    meta12 = hashlib.sha256(open("experiments/meta/corpus_crisislext26_meta.tsv", "rb").read()).hexdigest()[:12]
    open(os.path.join(OUT, "README.md"), "w").write(README.replace("META12", meta12))
    open(os.path.join(OUT, "CHANGELOG.md"), "w").write(CHANGELOG)
    open(os.path.join(OUT, "regenerate.sh"), "w").write(REGENERATE)
    open(os.path.join(OUT, "LICENSE"), "w").write(LICENSE)
    open(os.path.join(OUT, "LICENSE-CrisisLex"), "w").write(CRISISLEX_LICENSE)
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
