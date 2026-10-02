# Changelog

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
