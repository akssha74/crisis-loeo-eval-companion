# Pilot preregistration — matched-estimand decomposition of the cross-event gap

Status: FROZEN 2026-09-28T16:49Z, before any outcome-bearing run of this study.
Role: disclosed exploratory pilot (pilot admission, not contract lock). A pilot never supports a paper claim.

## Why this pilot exists

The rejected SNCS manuscript (`AI_Disaster_IEEE/sn_computer_science/sncs_paper.tex`, SNCS-D-26-05757)
reported a "lab-to-deployment gap" as the pooled macro-F1 of a random 80/20 split minus the mean
per-event leave-one-event-out (LOEO) macro-F1. The reviewer objected that these are different test
populations and that LOEO is not temporally prospective. This pilot measures, on HumAID, how much of
that gap survives when the in-distribution and cross-event models are scored on the same tweets, and
whether training on events that happened after the target event changes cross-event scores.

## Prior outcome exposure (disclosed)

HumAID set1 labels are fully outcome-exposed: the rejected study reported pooled ID, per-event LOEO,
leave-one-hazard-out and mitigation results on this exact corpus. This pilot is therefore
development-visible, not a protected evaluation. Any confirmatory claim needs a corpus whose
individual outcomes this programme has not inspected (to be chosen after the data search).

## Data and frozen splits

- Corpus: HumAID event-wise set1, 13 events, 43,409 tweets, 10 labels
  (`AI_Disaster_IEEE/humaid_set1.tar.gz`, sha256 `5407c09c…1b44`).
- Canonical table `data/humaid_canonical.tsv` sha256 `d02ce1727ed75d467276b3c04792eb08bc5fd0b72c1502a2e873806a1f5b9671`
  (built by `code/prepare_humaid.py` sha256 `8815974d…0c5b`). Posting time decoded from each tweet ID
  (Twitter snowflake). Metadata: `research/prelock/humaid_metadata.json`.
- HumAID is already near-deduplicated by its creators: 0 within-event normalized-text duplicates,
  7 texts shared across events. A duplicate-removal condition is therefore uninformative on HumAID and
  is not run in the pilot (it is retained for corpora with retweet duplicates).
- Split manifest `experiments/splits/pilot_splits.json` sha256 `b2e0a8105ce77ae37740f85e7534eef293de83be3fe68ba558a583a7eaeefde6`
  (built by `code/make_splits.py` sha256 `cd0ee924…80f0`; labels never read; split seed 20260928).
  Per event e: `R_e` = random 20 %; `T_e` = latest 20 % by posting time.
  Chronological pool for e = tweets of other events posted strictly before e's first tweet.

## Training conditions (all: training set capped at 6,000 tweets by seeded random sampling)

| Condition | Training pool | Evaluated on |
|---|---|---|
| `id_random` | all tweets not in any R_e | ∪ R_e |
| `id_temporal` | all tweets not in any T_e | ∪ T_e |
| `loeo:e` (13) | tweets of all other events | all tweets of e |
| `chrono:e` (11; pool ≥ 3,000) | tweets posted before e's first tweet | all tweets of e |
| `loeon:e` (3) | LOEO pool subsampled to the chrono pool size (only where that is < 6,000) | all tweets of e |

Model: DistilBERT-base-uncased, Hub revision `12040accade4e8a0f71eabdb258fecc2e7e948be`
(safetensors sha256 `5e3f1108…0063`, verified against the Hub). AdamW, lr 2e-5, weight decay 0.01,
batch 32, 10 % linear warm-up, 2 epochs, max length 96, seed 42 — the rejected study's configuration,
kept so the pilot measures the protocol, not a new recipe. The cap is sampled uniformly rather than
label-stratified (the rejected study stratified by label).
Runner `code/finetune.py` sha256 `fa953952b5bbc8e8bdb6b6ab14e1b08f6d86563af1166151e0f62931f4b094cc`.
Environment: Apple M2 Max, MPS; torch 2.13.0, transformers 5.17.0, scikit-learn 1.9.0, scipy 1.18.1,
numpy 2.5.2, pandas 3.0.5, tokenizers 0.23.2.

## Estimands (analysis `code/analyze_pilot.py` sha256 `29631e4ff22ddb1927481153b482da2e1677a2e1e87e75bd2256712a3820f202`)

Macro-F1; per-event values averaged with equal event weight.

- E0 prior-study gap: F1pool(id_random, ∪R) − mean_e F1(loeo_e, e)
- E1 aggregation: F1pool(id_random, ∪R) − mean_e F1(id_random, R_e)
- E2 matched event exposure: mean_e [F1(id_random, R_e) − F1(loeo_e, R_e)]
- E3 test subsample: mean_e F1(loeo_e, R_e) − mean_e F1(loeo_e, e); E0 = E1 + E2 + E3 exactly
- E2p pooled matched: F1pool(id_random, ∪R) − F1pool(loeo predictions on the same ∪R)
- E4 matched temporal: mean_e [F1(id_temporal, T_e) − F1(loeo_e, T_e)]
- E5 future-event leakage: mean over chrono events [F1(size-matched loeo, e) − F1(chrono_e, e)]

Primary macro-F1 convention: scikit-learn default (labels in y_true ∪ y_pred), for continuity with the
rejected study. Secondary: classes present in y_true only.
Uncertainty: event-level bootstrap (10,000 resamples, equal event weight) and Wilcoxon signed-rank
across events for E2/E4/E5; instance bootstrap stratified by event (2,000) for E2p. One seed, so no
training-variance estimate; the pilot is descriptive.

## Materiality

0.02 macro-F1: about four times the rejected study's across-seed SD of the LOEO mean (0.004–0.006).
Data-informed (from the rejected study), not outcome-blind.

## Modal-outcome pre-mortem (recorded before running)

Most likely result: E0 ≈ 0.04–0.06 (reproduces the prior gap); E1 ≈ 0.02–0.04 (per-event macro-F1 on
small events with rare classes is lower than pooled macro-F1); E2 ≈ 0.02–0.03 (a real but smaller
matched gap); E4 ≈ E2; E5 ≈ 0–0.02. Under that outcome the modal paper is defensible (the headline
gap roughly halves under a matched estimand) but not great on its own; it needs the method-reproduction
arm and a second corpus.

## Framing decision rule (branch selection, not a study kill)

- Branch A, "estimand artefact": E0 ≥ 0.03 and (E2 < 0.02 or E2's event-bootstrap CI includes 0).
- Branch B, "real matched gap": E2 ≥ 0.02 with CI excluding 0 → matched/prospective benchmark plus
  head-to-head reproduction of published cross-event methods.
- KILL of the decomposition framing: E0, E2, E4 all < 0.02 and |E5| < 0.02 (nothing to decompose).
- E5 ≥ 0.02 with CI excluding 0 adds "future-event leakage" as a reported component in either branch.

## Failed-run handling

An execution crash before predictions are written is an implementation failure: record it in the run
ledger as failed, fix without changing any scientific field, and replay the same run once. No
outcome-informed reruns, no alternate seeds, no split changes.

## Compute budget

≤ 2 h wall clock on the local M2 Max (≈ 29 runs). No paid or cloud compute.

## Protected-evaluation access log

None. No protected evaluation set exists for this study yet.

## Pilot outcome (appended 2026-09-28T18:20Z, after the frozen analysis ran)

Source: `experiments/derived/pilot_summary_distilbert_s42.json` (29/29 runs succeeded; analysis code
unchanged, sha256 `29631e4f…f202`). Primary convention (scikit-learn default macro-F1):

| Estimand | Value | Notes |
|---|---|---|
| E0 prior-study gap | 0.069 | pooled ID 0.600 vs mean LOEO 0.532 |
| E1 aggregation | 0.006 | |
| E2 matched exposure | 0.029 (event bootstrap 95% CI 0.018–0.041) | 12/13 events positive, Wilcoxon p = 0.0007 |
| E3 test subsample | 0.033 | |
| E2p pooled matched | 0.045 (instance CI 0.037–0.052) | |
| E4 matched temporal | 0.023 (0.009–0.037) | 9/13 positive |
| E5 future-event leakage | −0.001 (−0.024–0.020) | 11 events |

Present-class convention: E0 = 0.020, E2 = 0.025 (0.015–0.036), E3 = −0.007, E5 = 0.015 (0.000–0.031).

Framing rule result: **Branch B** (E2 ≥ 0.02, CI excludes 0).

Unregistered, exploratory observation: E3 is not sampling noise. Under the default convention the
per-event macro-F1 of a whole held-out event averages over every class the model predicted, and LOEO
models predict on average 2.3 tweets per event into 0.77 classes absent from that event's labels (Italy:
4 predictions into 3 absent classes over 1,201 tweets). Each absent class adds a zero-F1 term. The
random 20 % subsets rarely contain those stray predictions (0.15 absent classes per event), so the term
appears as E3. This metric-convention component must be registered as a confirmatory estimand before
it can support any claim.

Modal pre-mortem scoring: E0 above the predicted 0.04–0.06; E1 far below the predicted 0.02–0.04
(miss); E2 within 0.02–0.03 (hit); E4 ≈ E2 (hit); E5 within 0–0.02 (hit). The unpredicted component is
the metric convention (E3), not aggregation.

## Deviations log

- 2026-09-28T17:05Z, execution only. Attempt 1 of `id_random` was killed by the host shell during
  epoch 2 (the job had been started with `nohup … &` from a tool shell that terminated its process
  group). No predictions were written and no evaluation outcome was observed; recorded as a failed run
  in `experiments/run-ledger.jsonl`, partial log moved to `experiments/failed/`. The run-directory
  naming in `code/finetune.py` was also fixed (outputs were landing under the model revision hash
  instead of the model name). No scientific field changed: same model, seed, splits, hyperparameters
  and analysis. Patched runner sha256 `ee410a5b8343c2b4f7299eb1b3b478cb4443199570f95a6d883068c0f346f039`.
  The full plan is replayed once from the start as a detached process.
