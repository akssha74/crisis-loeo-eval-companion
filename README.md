# Evaluation companion for event-held-out HumAID and CrisisLexT26 (Online Resource 2)

Companion to "A Reproducible Evaluation Companion for Event-Held-Out Crisis-Message Classification" (Sharma and
Prasad, Language Resources and Evaluation, Project Note). Version 1.1.0. It contains no tweet text.

## What it is for
Leave-one-event-out (LOEO) results can be summarised as one macro-F1 over all held-out messages (pooled) or as
the mean of per-event macro-F1, and per-event scores depend on which classes enter each event's average (the
label list). The evaluator scores a set of predictions under a label list that must be named, reports both
summaries, their difference A and its split, and prints a signature that records the convention, for example

    loeo-macro-f1|labels:P|zero-division:0|events:26|meta:7d042873c1f4|v:1.1.0

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
  (`revision_x9.py`, `revision_x11.py`, `revision_x12.py`, `run_zeroshot.py score`), `check_note_values.py`
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

    python code/companion_eval.py --meta experiments/meta/corpus_crisislext26_meta.tsv \
        --preds "my_run/*.tsv.gz" --convention P --out out

The evaluator rejects duplicated or unknown identifiers, events or true labels that disagree with the metadata,
predictions outside the corpus classes, and incomplete events; files covering only some events need
`--partial`, and the output is then marked partial. Outputs: `scores.json` (pooled and per-event macro-F1, A,
the split of A^P into T1 to T4, the label-list shift, per-event values, the signature and a hash of the inputs)
and `event_class_counts.tsv` (TP, FP, FN, support, precision, recall and F1 per event and class), from which any
other label list or summary can be computed. To add a reference system to the index, add its runs under
`experiments/runs` and a row to `experiments/systems.tsv`.

Other multi-event corpora can be scored in the same way: write a metadata file with the columns `tweet_id`,
`event`, `hazard` and `label` and pass it with `--meta`. Nothing in the evaluator is specific to the two corpora.

## Tests and regeneration
    python code/test_companion_eval.py      # or: python -m pytest code/test_companion_eval.py
    sh regenerate.sh                        # recompute every number of the note and compare

The tests check the three label lists against hand-computed values and scikit-learn, the closed-form shifts
between them, that T1 + T2 + T3 + T4 = A^P, the input validation and the signature.

## Versions and maintenance
Releases are tagged (`v1.0.0`, `v1.1.0`) and archived in Software Heritage. A changed number in a release is
recorded in `CHANGELOG.md`; the evaluator version is printed in every signature. Problems can be reported
through the issue tracker of https://github.com/akssha74/crisis-loeo-eval-companion, which the first author
maintains.

## Licences
Code: MIT (`LICENSE`). Files that carry labels (`experiments/meta`, `experiments/runs`): CC BY-NC-SA 4.0. The
Hugging Face release `QCRI/HumAID-events` declares CC BY-NC-SA 4.0 in its metadata and CC BY-NC 4.0 in its card
text; the share-alike licence meets the terms of both. The CrisisLexT26 labels come from the CrisisLex repository
(https://github.com/sajao/CrisisLex, commit d67cddd5), released under the MIT licence; its copyright and permission
notice is reproduced in `LICENSE-CrisisLex`. Tweet text must be obtained from the providers under their terms.
