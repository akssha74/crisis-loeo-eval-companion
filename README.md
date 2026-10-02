# Evaluation companion for event-held-out HumAID and CrisisLexT26 (Online Resource 2)

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
    python code/companion_eval.py --convention P --out out/ \
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
