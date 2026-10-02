# Changelog

## v1.1.0 (2026-10-02)
- Evaluator: `--corpus/--system/--seed` scores a reference system listed in `experiments/systems.tsv`; every result
  carries a signature (label list, zero-division rule, events, metadata hash, evaluator version).
- Tests (`code/test_companion_eval.py`) and a file inventory (`experiments/derived/companion_inventory.json`).
- Four-epoch HumAID seed 4 completed (27 runs that were missing from v1.0.0; see the deviations log in
  `research/preregistration.md`); every four-epoch HumAID value now averages five seeds.
- Analyses X11 and X12 (`code/revision_x11.py`, `code/revision_x12.py`) and the within-event duplicate lists.

## v1.0.0 (2026-10-02)
- First release.
