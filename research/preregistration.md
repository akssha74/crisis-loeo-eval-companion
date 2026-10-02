# Confirmation preregistration — decomposing the random-split vs cross-event gap

Status: FROZEN 2026-09-28T18:40Z, before any confirmation run. Commit recorded in git history.
Predecessor: disclosed exploratory pilot, `research/pilot-preregistration.md` (Branch B).

## Question

When a crisis-message classifier's in-distribution score (pooled random split) is compared with its
leave-one-event-out (LOEO) score, how much of the difference is (i) a genuine effect of having trained
on the target event, measured on identical tweets; (ii) aggregation (pooled vs per-event averaging);
(iii) the macro-F1 label-set convention; (iv) training on events that happened after the target; and
(v) near-duplicate leakage? The pilot answered this for one seed and one encoder on outcome-exposed
data. This confirmation tests the registered components on six HumAID events this programme has
never scored, a second corpus, two encoders and several seeds.

## Hypotheses and mechanism

- **H1 (real exposure effect).** On identical target-event tweets, a model whose training pool
  includes other tweets of the target event scores higher than a LOEO model: E2 > 0.
  Mechanism: event-specific vocabulary, entities and label priors are learnable from same-event tweets.
- **H2 (metric-convention component).** Scoring each whole held-out event with scikit-learn's default
  macro-F1 (average over classes in y_true ∪ y_pred) lowers LOEO scores relative to present-class
  macro-F1: MC > 0. Mechanism: LOEO models occasionally predict a class absent from the event's labels;
  each such class adds a zero-F1 term to the default average.
- **H3 (future-event leakage negligible).** Restricting training to tweets posted before the target
  event's first tweet, at matched training size, changes whole-event LOEO macro-F1 by less than the
  materiality margin: |E5| < 0.02 (equivalence).

## Data, exposure and frozen splits

| Corpus | Events | Tweets | Labels | Role | Exposure |
|---|---|---|---|---|---|
| HumAID set1 | 13 (2016–2019) | 43,409 | 10 | development | fully outcome-exposed (rejected study; this pilot) |
| HumAID set2 | 6 (Aug 2018–Sep 2019) | 33,075 | 10 | **protected confirmation** | metadata only; ~40 dataset-card preview rows and one dev row seen |
| CrisisLexT26 | 26 (2012–2013) | 23,375 | 6 | second-corpus replication | prior study saw pooled-ID and default-convention LOEO scores; no matched or present-class estimand ever computed |

- HumAID-19 table `data/corpus_humaid19.tsv` sha256 `b4c30b0b24ba9a773d510a11fd798fdf6fb87e014e7ce680b756e22ccad2e3d9`
  (set1 archive + Hugging Face `QCRI/HumAID-events` rev `2e7ea233…`, whose 13 set1 events were verified
  identical to the archive by tweet ID and event). Built by `code/prepare_set2.py` (`f2e4ed4d…`) and
  `code/prepare_corpora.py` (`63401260…`).
- CrisisLexT26 table `data/corpus_crisislext26.tsv` sha256 `83e4187f5333ab94a3987f8ba8e212f9b1d1f068bf6f693eb2435a8b8a0762fd`
  (GitHub `sajao/CrisisLex` commit `d67cddd5…`; information-type task; "Not labeled"/"Not applicable"
  dropped; 6 classes; one tweet ID labelled in two events dropped).
- Split manifests (labels never read; split seed 20260928; `code/make_splits.py` `cd0ee924…`):
  `experiments/splits/confirm_humaid19.json` `be1cd3ed0ac402a1ad62b7690b520593ee264e4136e1465d8eb552e1c5c8c000`,
  `experiments/splits/confirm_crisislext26.json` `7ed91531a1154c5c45d2f77a4f0619da2ca74fb45915c4e90dd9051cd866e785`.
  Per event: R_e random 20 %; T_e latest 20 % by posting time (decoded from tweet IDs).
- Near-duplicates: normalized-text identity (lower-case, URLs/users/punctuation removed, leading "RT
  @user:" stripped). HumAID has almost none (it works as a negative control); CrisisLexT26 has 3,014
  within-event duplicate rows.

## Training conditions (runner `code/run_protocol.py` sha256 `c574f92fceef89ca9139e1675ca3f438cc753fd17e137e3047235a8af2713d41`)

Every training set is capped at 6,000 tweets by seeded uniform sampling (run-key × seed hash).

| Condition | Training pool | Evaluated on |
|---|---|---|
| id_random | all tweets not in any R_e | ∪R |
| id_random_dedup | id_random pool minus tweets whose normalized text occurs in ∪R | ∪R |
| id_temporal | all tweets not in any T_e | ∪T |
| id_temporal_dedup | id_temporal pool minus normalized-text matches to ∪T | ∪T |
| loeo:e | all other events | all of e |
| chrono:e | tweets posted before e's first tweet (pool ≥ 3,000) | all of e |
| loeon:e | LOEO pool subsampled to the chrono pool size, where that is < 6,000 | all of e |

HumAID-19: 43 runs per model-seed (17 chrono-eligible events, 3 of which need a size-matched LOEO run).
CrisisLexT26: 55 runs (22 chrono-eligible events, 3 needing a size-matched LOEO run).

Models: DistilBERT-base-uncased rev `12040acc…` and RoBERTa-base rev `e2da8e2f…` (safetensors verified
against the Hub). Fixed recipe from the rejected study, no tuning and no target data used for any
selection: AdamW lr 2e-5, weight decay 0.01, batch 32, 10 % linear warm-up, 2 epochs, max length 96.
Seeds: HumAID-19 {42, 1, 2, 3, 4}; CrisisLexT26 {42, 1, 2}.

Recipe-sensitivity arm (secondary): DistilBERT, HumAID-19, seed 42, 4 epochs, all 43 conditions
(`--plan confirm_ep4`). Per-epoch training loss is logged for every run.

## Estimands (analysis `code/analyze.py` sha256 `4dbe8a50c440c93225ad526a377a70e3aa0b8924eed1a302645ec10889de3c3e`)

Validated before freezing: on the pilot predictions it reproduces the pilot E0/E1/E2/E3/E4/E5 exactly.
Definitions are in the analysis docstring; E0 = E1 + E2 + E3 holds exactly.

**Primary macro-F1 convention: present-class (P)**, because it is invariant to stray predictions of
absent classes. The scikit-learn default (D) is reported alongside for comparability with the literature
and the rejected study.

## Decision rules

Uncertainty: hierarchical bootstrap (resample events, then seeds within each event; 5,000 draws) of the
event-mean estimand; Wilcoxon signed-rank on seed-averaged event differences is descriptive only (six
protected events cannot reach Holm-corrected significance by construction). Materiality m = 0.02.

- **H1**, per model, on the 6 protected HumAID events, convention P: supported if the 95 % CI of E2
  excludes 0; "material" additionally if the mean ≥ 0.02. Also reported on set1, all 19 events and
  CrisisLexT26.
- **H2**, per model: supported on a corpus if the 95 % CI of MC excludes 0. Tested on CrisisLexT26 and
  on all HumAID-19 events; reported separately for protected events. Expected to be near 0 for events
  whose label set is complete (MC is zero by construction when no absent class is predicted).
- **H3**, per model and corpus, convention P, chrono-eligible events: supported if the 90 % CI of E5 lies
  within (−0.02, +0.02).
- Secondary: E4 > 0 (95 % CI); DUP and DUPT on CrisisLexT26 (expected > 0) and HumAID (expected ≈ 0,
  negative control); E1; the share of E0[D] carried by E1, E2 and E3; recipe sensitivity (4 epochs)
  changes E2, MC or E5 by less than 0.02.

## Refutation value

- H1 fails on protected events: the exposure effect does not carry to later events; the conventional gap
  is then mostly an evaluation artefact — a stronger message.
- H2 fails on CrisisLexT26: the convention effect is corpus-specific; reported as a HumAID-only
  observation, with guidance scoped accordingly.
- H3 fails: training on future events matters; reported as a registered component with its size.
Every branch yields a complete, reportable decomposition.

## Modal-outcome pre-mortem

Most likely: E2[P] on protected events 0.02–0.04 for both models (H1 supported); MC 0.03–0.08 on
CrisisLexT26 (many events lack one of 6 classes) and 0–0.02 on protected HumAID events (large events
with near-complete label sets); E5 within ±0.015 but the 90 % CI may cross ±0.02 on HumAID (H3 may be
inconclusive rather than refuted); DUP on CrisisLexT26 0.005–0.02. The modal paper is a
publishable evaluation-methodology contribution: the conventional gap mixes a real exposure effect with
a scoring artefact of similar size, and prospective training order barely matters.

## Failed-run handling

Execution crashes are recorded as failed runs; a semantics-preserving fix is replayed once for the same
run key and seed. No outcome-informed reruns, no extra seeds added after results are seen, no split
changes. A run that cannot complete after one replay is reported as missing and the estimand is
computed on the remaining seeds with the gap disclosed.

## Protected-evaluation access log

- 2026-09-28: set2 fetched and summarised at metadata level only (counts, posting times, duplicates).
- The protected predictions are written by the runner and first read by `code/analyze.py` after all
  HumAID-19 runs complete. No per-event protected result is inspected before that.
- 2026-09-29 (found by OPUS-R0-F9): the first HumAID learning-curve runs (amendment A1; DistilBERT folds
  Harvey, Puebla, Sri Lanka) drew training and validation tweets from pools containing protected events; their
  per-epoch aggregate source-validation macro-F1 (600 tweets, ~40–50% protected) was printed to a log. This is
  aggregate, non-estimand exposure: no protected event was scored as a held-out target, no per-event protected
  result exists, and no registered estimand can be computed from it. Outputs quarantined; reruns exclude
  protected tweets.
- 2026-09-30T07:45Z: all registered HumAID-19 and CrisisLexT26 base runs complete for both encoders (215 and
  165 per encoder). Protected predictions first read by `code/analyze_v2.py --plan confirm` (then `analyze.py`,
  `diagnostics.py`), with optional arms still running (RoBERTa CrisisLexT26 mask seed 1, DistilBERT HumAID
  full-pool, RoBERTa four-epoch replicate) left out of this pass and added when complete.

## Compute budget

≈ 430 HumAID-19 runs + 330 CrisisLexT26 runs + 43 sensitivity runs on the local M2 Max (MPS),
≤ 60 h wall clock, two concurrent streams. No paid or cloud compute.

## Amendment A1 (2026-09-29T19:15Z, before any confirmation outcome was read)

Secondary, recipe-justification only (lesson L26): source-validation learning curves. For five
development-visible LOEO folds (HumAID set1: Harvey, Puebla, Sri Lanka; CrisisLexT26: Boston, Pablo),
DistilBERT and RoBERTa-base, seed 42: 6,000 training and 600 disjoint validation tweets drawn from the
fold's source pool, 5 epochs with the registered recipe, source-validation loss and present-class
macro-F1 after each epoch (`code/convergence.py`). No target event is scored, so no protected or
registered estimand is exposed. Reported descriptively next to the registered four-epoch replicate.

## Amendment A2 (2026-09-29T22:00Z, before any confirmation outcome was read)

Source: independent pre-results design reviews `reviews/design-review-{gpt,opus,glm}-r00.json`; adjudication
in `reviews/adjudication-r00.md`. No confirmation prediction, summary or log was read by the authors or the
reviewers. The protected set is still unopened.

**Implementation changes (runner `code/run_protocol.py`).**
1. Paired dedup (GPT-R0-F1, OPUS-R0-F12): a dedup arm keeps every non-duplicate tweet of its base arm's
   training sample and tops up from the same deduplicated pool. The eight unpaired seed-42 dedup runs already
   trained were never read; they are superseded (ledger records) and rerun under the paired rule. On HumAID
   the paired dedup sample equals the base sample (no duplicate was drawn), so HumAID DUP is a pure
   run-to-run noise placebo.
2. Opt-in secondary arms (base plans unchanged, verified by reconstructing recorded training-ID hashes):
   - `fuzzy`: `id_random_fuzzydedup`, near-duplicates removed at char 3-5-gram TF-IDF cosine >= 0.8
     (threshold fixed after inspecting unlabelled text pairs; `code/fuzzy_dups.py`; 2,419 CrisisLexT26 and 266
     HumAID pool tweets), seeds 42, 1, 2, both encoders, both corpora.
   - `mask` (consequence arm; GPT-R0-F6, GLM-R0-F3): event-token masking following the masking baseline
     evaluated by Seeberger et al. (2025) — spaCy en_core_web_sm 3.8.0 entities replaced by their type,
     hashtags by [HASHTAG], digit strings by [NUM] (`code/mask_entities.py`, label-free) — applied to training
     and test text, with training samples identical to the unmasked arms: `id_random_mask` and `loeo_mask:e`,
     seeds 42 and 1, both encoders, both corpora.
   - `divk` (OPUS-R0-F4): diversity-matched control `loeok:e` = LOEO restricted to k randomly chosen other
     events (k = number of source events in chrono(e)'s pool; set fixed per event), trained on as many tweets as
     chrono(e); seed 42, both encoders, both corpora. E5 = E5_div + E5_time with
     E5_div = F(loeo_n) − F(loeok), E5_time = F(loeok) − F(chrono).
   - `full` (OPUS-R0-F11; SNCS letter uncapped item): `id_random_full` on its whole pool (61,181 HumAID; 18,689
     CrisisLexT26) and `loeo_full:e` subsampled to the same size; DistilBERT, seed 42, both corpora.
   - RoBERTa four-epoch replicate on HumAID-19, seed 42 (OPUS-R0-F18).
3. Learning curves (OPUS-R0-F9): HumAID convergence pools included protected tweets (aggregate validation
   macro-F1 over 600 source tweets of which ~40–50% protected; non-estimand exposure, now logged below). Pools
   are restricted to `protected == 0`; the three HumAID DistilBERT folds already run are quarantined in
   `experiments/failed/convergence-protected-exposure/` and rerun. Curves are described as plateau evidence under
   a five-epoch schedule, not as a check of the two-epoch end state.

**Analysis (`code/analyze_v2.py` primary; `code/analyze.py` retained and reported as the as-registered
analysis; `code/diagnostics.py` secondary).**
- Primary interval for per-event estimands: two-sided t-interval (df = events − 1) on seed-averaged event
  contributions (OPUS-R0-F3: the nested percentile bootstrap undercovers with six events). Equivalence: TOST
  with the 90% t-interval. Crossed bootstrap (one global seed vector per draw, all terms recomputed from
  per-class counts so the identity holds draw by draw) is secondary and gives intervals for E0, E1, E3,
  pooled E2p and component shares (GPT-R0-F2, F9).
- Decomposition reporting (OPUS-R0-F2, F14): the main identity is E0[D] = E1[P] + E2[P] + E3[P] + MC; both
  orderings (E0 = E1 + E2 + E3 and E0 = E2p + E1_loeo + E3), the ordering interaction I = E2p − E2 and a
  symmetric two-path attribution are reported. Per-event E2 is primary because LOEO evaluation averages per
  event. The full decomposition is computed on every subset, including the protected events (GPT-R0-F5).
- Third convention O (fixed corpus ontology, absent classes scored 0) reported descriptively (GPT-R0-F7).
- Renaming (GPT-R0-F3, F4): E2 is the "matched ID-versus-LOEO contrast at the natural pooled share of the
  target event with n = 6,000", not an isolated exposure mechanism; E5 is the "chronological versus all-events
  LOEO contrast at matched size", decomposed by the divk control. Exposure dose (target tweets in the ID
  sample) and E5 hazard-composition distance are reported per event.
- Margin: 0.02 is anchored to published cross-event method gains (Seeberger et al. 2025: +2.1, +4.4, +3.7
  macro-F1 points on HumAID, CrisisLexT26, TREC-IS), i.e. an artefact of 0.02 is as large as a typical method
  improvement; conclusions are also shown at 0.01 and 0.03 (GPT-R0-F8, OPUS-R0-F17).

**Revised decision rules.**
- H1 (primary, protected): E2[P] on the six protected HumAID events, 95% t-interval excluding 0 (material if
  mean >= 0.02). H1b (replication): the same on CrisisLexT26, where matched E2 was never computed. A
  model-general claim requires both encoders to pass (OPUS-R0-F20). Leave-one-protected-event-out values are
  reported.
- H2 (magnitude, replaces the sign test, which holds by construction since MC >= 0; GPT-R0-F7, OPUS-R0-F1):
  MC >= 0.02 with 95% t-interval lower bound > 0, decided on CrisisLexT26 and on the protected events; set1 is
  discovery data. The share of events affected, MC by number of absent classes, and MC's share of E0[D] are
  reported.
- H3 (equivalence): TOST at ±0.02 on chrono-eligible events with future share >= 0.25 (future share = 1 −
  |chrono pool| / |LOEO pool|), decided on protected ∪ CrisisLexT26 events; events with future share < 0.05
  are reported as placebos (OPUS-R0-F4, F5).
- Inconclusive branches are stated as such in the abstract (GLM-R0-F5).
- Consequence estimands (GPT-R0-F6, OPUS-R0-F7, GLM-R0-F3): (i) masking's change in the conventional gap
  (ΔE0[D]) versus in the matched contrast (ΔE2[P]) and its LOEO benefit under D versus P; (ii) Kendall τ of
  per-event LOEO difficulty rankings under D versus P, events moving >= 3 ranks, and whether the hardest event
  and hardest hazard change; (iii) the RoBERTa − DistilBERT LOEO difference under D versus P and under
  conventional versus matched scoring. (Pilot, disclosed: τ = 0.62; hardest hazard floods under D, wildfires
  under P.)
- Secondary diagnostics (`code/diagnostics.py`): oracle label-prior matching (upper bound on the share of E2
  explained by label priors; EM re-estimation diverged on the pilot and is not used), E2 by near-duplicate
  similarity stratum, classes-with->=5-instances sensitivity, tweet-weighted E2/E4, stray classes on matched
  instances, per-class breakdown.
- Literature audit (OPUS-R0-F7): at least ten cross-event crisis papers and their code are tabulated for the
  metric, per-event versus pooled averaging and label-list use; prevalence claims in the paper are limited to
  what the audit shows.

**Run order for added arms** (two extra streams, one per encoder): paired seed-42 dedup reruns → fuzzy →
divk → mask → (DistilBERT) full-pool / (RoBERTa) four-epoch replicate; learning-curve reruns in parallel.
All added arms are reported whether or not they change a conclusion.

## Post-results exploratory additions (not confirmatory)

- X1 (2026-09-30T08:30Z, after reading the confirmation results and the registered DistilBERT four-epoch
  replicate on HumAID-19). The four-epoch replicate changed the HumAID composition (E0[D] 0.042 → 0.129 at seed
  42, mostly through E1 and MC; E2 0.021 → 0.026), so the registered recipe-sensitivity criterion (E2, MC, E5
  change < 0.02) fails for MC. To learn whether the CrisisLexT26 composition is equally recipe-dependent we add a
  DistilBERT four-epoch replicate of every CrisisLexT26 condition (seed 42, 55 runs, `--plan confirm_ep4`). It is
  reported as exploratory, whatever it shows.
- X2 (2026-09-30T14:38Z, commit 79838676, before any X2 run started; after reading review round 1). At two epochs `missing_or_found_people` (present in 4 of 19
  HumAID events) is never predicted by any DistilBERT seed or by RoBERTa seeds 1 and 3; both seed-42 four-epoch
  replicates learn it. To learn whether the four-epoch HumAID composition holds across seeds we add four-epoch
  HumAID-19 runs of the conditions behind E0–E4 and MC (`id_random`, `id_temporal`, both dedup variants and the
  19 `loeo` runs; 23 runs per seed) for the next seeds in the registered order: DistilBERT seeds 1 and 2 and
  RoBERTa seed 1 (69 runs, `--plan confirm_ep4`, `code/run_x2.sh`). E5 stays a seed-42 quantity. Reported as
  exploratory, whatever it shows.
- X3 (after reading review round 1; recorded in the commit that adds `code/run_tfidf.py`, before any X3 run).
  To learn whether the composition is specific to fine-tuned encoders, a TF-IDF + logistic-regression classifier
  goes through every base condition of both corpora (the 43 HumAID-19 and 55 CrisisLexT26 run keys of
  `--plan confirm`) with the registered seeds (HumAID 42, 1, 2, 3, 4; CrisisLexT26 42, 1, 2). Training samples are
  those of the transformer runs (same run key and seed; the runner asserts equal `train_ids_sha256`). Fixed,
  untuned settings: word 1–2-grams, lowercase, `min_df=2`, sublinear tf; L2 logistic regression, C = 1, lbfgs,
  `max_iter=2000`. Analysed with `code/analyze_v2.py --plan x3_tfidf`. Reported as exploratory, whatever it shows.
- X4 (2026-10-01T09:10Z, before any X4 run started; after reading TMLR referee round 1, which asked for the
  HumAID decomposition under a recipe that is not under-fitted, with at least three seeds per encoder). In the
  source-only learning-curve folds (Appendix F) present-class macro-F1 at epoch 4 is within 0.02 of its five-epoch
  maximum in all six HumAID folds, and source-validation macro-F1 peaks at epoch 3, 4 or 5, so the four-epoch
  recipe is the source-validated plateau, applied identically to every condition. X2 gave DistilBERT three
  four-epoch seeds (42, 1, 2) but RoBERTa two (42, 1). We add RoBERTa seed 2, the next seed in the registered order,
  with the X2 run list (23 runs, `--plan confirm_ep4`, `code/run_x2.sh roberta 2`). Reported as exploratory,
  whatever it shows.
- X5 (2026-10-01T09:10Z; evaluation-time analyses of existing predictions, no training, recorded before they were
  computed; `code/revision_tmlr.py`). (a) The gap that each audited practice produces on these data (pooled
  versus pooled on matched tweets; pooled in-distribution versus pooled whole held-out events; per-event versus
  per-event on different test tweets; the composite E0) and its relation to the matched contrast. (b) The HumAID
  decomposition with `missing_or_found_people` merged into `injured_or_dead_people` in labels and predictions, the
  merged scheme of Seeberger et al. (2025). (c) Intervals for the per-event estimands (E2, MC, E5) from a crossed
  random-effects (events × seeds) variance estimate with Satterthwaite degrees of freedom, as a sensitivity
  analysis beside the registered t intervals; registered decisions are unchanged. Reported as exploratory.
- X6 (2026-10-01T12:10Z; evaluation-time analyses of existing predictions, no training, recorded before they were
  computed; after reading TMLR referee round 3). (a) The different-events comparison of the audited papers
  (Kersten et al. 2019; development-set proxies): for each target event, a pooled in-distribution score on the
  held-out random fifths of all other events minus the whole-event LOEO score, averaged over target events, under
  conventions P and D. The in-distribution model stands in for the LOEO model on its own training events (it also
  saw the target's natural share; no model weights were kept). Crossed bootstrap over events and seeds as in X5a.
  (b) For MC, a percentile bootstrap over events of seed-averaged event contributions (non-negative by
  construction) and the number of events with MC > 0, beside the registered t interval. (c) A source-only epoch
  rule for the four-epoch arms: the epoch with the highest mean source-validation present-class macro-F1 over the
  learning-curve folds of each encoder (earliest on ties). Reported as exploratory.
- X7 (2026-10-01T12:56Z, before any X7 run started and before any X7 analysis was computed; after reading TMLR
  referee round 4). (a) Four-epoch HumAID-19 runs of the X2 run list (23 runs per seed) for the remaining
  registered seeds 3 and 4 of both encoders (92 runs; `code/run_x2.sh distilbert 3`, `… 4`, `… roberta 3`,
  `… 4`), so that every four-epoch HumAID arm has the five registered seeds. The four-epoch results are reported
  for the five seeds, and for seeds 42, 1 and 2 as previously reported, whatever they show. (b) Evaluation-time
  analyses of existing predictions: crossed events × seeds intervals (as in X5c) for DUP, DUP_F and E3, per-seed
  DUP values, and two one-sided tests of DUP against ±0.01 and ±0.02 with those intervals; pooled-minus-per-event
  macro-F1 of the same LOEO predictions on whole events (LOEO aggregation) with crossed-bootstrap intervals; per-seed
  protected E2 for the four-epoch arms; a Holm adjustment over the ten registered decisions (H1, H1b, H2 on the
  protected events, H2 on CrisisLexT26 and H3, for each encoder) from the registered t tests; H1 materiality judged
  on the 95% lower bound; the loss-based epoch rule (epoch of minimum mean source-validation loss) beside the
  macro-F1 rule; per-class recall of whole-event LOEO predictions for the HumAID classes `missing_or_found_people`,
  `injured_or_dead_people` and `requests_or_urgent_needs`, pooled and per-event mean. Reported as exploratory.
- X8 (committed in 34ae3932d at 2026-10-01T16:25:49Z, before any X8 run started and before any X8 analysis was computed; after reading TMLR
  referee round 5). (a) Identical-seed repeats, to separate MPS non-determinism from seed variance: the
  two-epoch DistilBERT HumAID-19 runs `id_random` and `loeo:` of the six protected events at seed 42, trained
  twice more with the same run keys and seed (`--plan x8_repeat_a`, `--plan x8_repeat_b`; 14 runs; same training
  samples by construction). Reported: per-event and mean protected E2 and per-event F1 of each replicate beside
  the original run, the standard deviation over the three seed-42 replicates, and the standard deviation over
  the five registered seeds. (b) Evaluation-time analyses of existing predictions (`code/revision_x8.py`): on
  CrisisLexT26, the matched contrast against the deduplicated in-distribution model,
  F(`id_random_dedup`, R_e) − F(`loeo(e)`, R_e), with the registered t interval and the crossed interval of X5c;
  the H3 rule (90% t interval of E5 inside ±0.02, events with future share ≥ 0.25) applied per corpus
  (CrisisLexT26; the protected HumAID events; all eligible HumAID events); for every arm (two and four epochs,
  capped and uncapped, TF-IDF) the whole-event LOEO aggregation under convention P with a 95% crossed-bootstrap
  interval (event bootstrap for single-seed arms) and E2 with its 95% t interval; per-seed E0[D], E1, E2 and MC of
  the five-seed four-epoch HumAID arms, with the number of seeds that predict every class. No ratio of
  aggregation to E2 is computed. Reported as exploratory.
- X9 (written at 2026-10-01T21:45Z, before any X9 quantity was computed; after reading LRE referee round 6 of the
  Project Note). Evaluation-time analyses of existing whole-event LOEO predictions of the X8 arms; no model is
  trained (`code/companion_eval.py`, `code/revision_x9.py`). (a) LOEO aggregation A under conventions P, D and O
  for every arm, with the X7 crossed bootstrap (2,000 draws, seed 20261001). (b) The split
  A = W + N, where W = Σ_e w_e F_e − mean_e F_e with w_e = n_e / Σ n (event-size weighting) and
  N = F_pool − Σ_e w_e F_e (non-additivity of macro-F1 across events), under P, with the same bootstrap.
  (c) A under P after removing every test message whose event–class cell holds fewer than 15 messages (the HumAID
  provider's per-event rule), applied to both corpora, with the numbers of cells and messages removed. (d) Spearman
  correlation between a class's share of an event and its per-class F1 in whole-event LOEO predictions, per seed,
  for every two-epoch arm, with the number of event–class cells. (e) Per-event n_e, number of present classes and
  seed-mean F_e for the two-epoch arms. (f) Per-seed A of the two-epoch arms. (g) Independent check: every
  per-event and pooled macro-F1 of the two-epoch arms recomputed from the prediction files with scikit-learn
  `f1_score(average='macro', zero_division=0)` and explicit label lists, and with `code/companion_eval.py`; the
  maximum absolute difference from the analysis values is reported. Reported as exploratory, whatever it shows.
- X10 (written at 2026-10-01T22:25Z, before the model was downloaded or run; after reading LRE referee round 6,
  which noted that no system other than the authors' own was scored). A public system that is trained on no event:
  zero-shot natural-language-inference classification with `facebook/bart-large-mnli` at Hub revision
  `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` (`code/run_zeroshot.py`). Every message of both corpora is paired with
  one hypothesis per corpus class, "This message is about {class}.", where {class} is the class identifier with
  underscores replaced by spaces; the prediction is the class with the largest entailment logit. Input is the
  text field used for training (`text`), truncated at 128 tokens. (Corrected at 2026-10-01T22:40Z, before the
  model was run: the first wording named `text_norm`, the deduplication key, which training does not use.) Predictions are written in the
  companion's prediction-file format and scored only through `code/companion_eval.py` under P, D and O, with A
  and an event-bootstrap 95% interval (2,000 draws, seed 20261001) under each convention. No prompt, template or
  label wording is tuned. Reported as exploratory, whatever it shows.
- X11 (written at 2026-10-02T07:20Z, before any X11 quantity was computed, and pushed to the public companion
  repository before computation; after reading LRE referee round 7). Evaluation-time analyses of the existing
  whole-event LOEO predictions of every X9 arm and of the X10 system; no model is trained (`code/revision_x11.py`).
  Notation: C_e classes present in event e, k_e classes absent from e but predicted for at least one of its
  messages, s_ec support of class c in e, E_c the events containing c. All intervals use the X7 crossed bootstrap
  (2,000 draws, seed 20261001; events only for one-seed arms and X10).
  (a) Closed forms. With per-class F1 set to 0 when its denominator is 0, F_e^O = (|C_e|/|C|) F_e^P and
  F_e^D = |C_e|/(|C_e|+k_e) F_e^P, so, when every class occurs in the pooled data, A^O = A^P + mean_e (1-|C_e|/|C|)
  F_e^P and A^D = A^P + mean_e k_e/(|C_e|+k_e) F_e^P. Checked numerically for every arm and seed (maximum absolute
  deviation reported); k_e summarised per arm; A^O of a perfect classifier, mean_e (1-|C_e|/|C|), reported per
  corpus. A^D and A^O reported for every arm and X10 with intervals. scikit-learn `f1_score` with labels = O and
  `zero_division=np.nan` compared with convention D per event.
  (b) Four-term split of A^P, every arm and X10, with intervals: T1 (false positives in events lacking the class)
  = mean_c F_c(all events) - mean_c F_c(E_c), where F_c(S) is F1 of counts summed over events S; T2 (non-additivity)
  = mean_c F_c(E_c) - mean_c sum_{e in E_c} (s_ec / sum s) F_ec; T3 (support weighting) = mean_c of the
  support-weighted mean minus mean_c of the equal-weighted mean of F_ec over E_c; T4 (class composition) =
  mean_c equal-weighted mean over E_c - mean_e F_e^P. A^P = T1+T2+T3+T4. Second ordering: T3' = mean_c F_c(E_c) -
  mean_c F_c^eq(E_c), where F_c^eq pools the counts of each event in E_c weighted by 1/s_ec, and T2' = mean_c
  F_c^eq(E_c) - mean_c equal-weighted mean; T1 and T4 unchanged. W and N of X9 are kept for continuity only.
  (c) Ranking agreement. Per corpus and convention, the systems (X9 arms by seed mean, and X10) ordered by pooled
  macro-F1 and by per-event mean; Kendall tau between the two orders and every pair whose order differs.
  (d) Small-cell threshold. A^P after removing every test message whose event-class cell holds fewer than t
  messages, t in {5, 10, 15, 20, 30}, every arm and X10, both corpora, with cells and messages removed.
  (e) Small cells. Event, class and support of every cell below 15 messages, with seed-mean per-class F1 of the
  two-epoch arms.
  (f) Within-class association. For each two-epoch arm, Spearman correlation between s_ec and seed-mean F_ec
  across the events containing c, per class, with the median over classes; the X9(d) pooled values labelled by
  arm.
  (g) Event jackknife. A^P of every arm and X10 with each event left out in turn (range, and the event whose
  removal moves A^P most); bootstrap quantiles 2.5, 25, 50, 75 and 97.5 of A^P; the share of bootstrap draws
  in which some class is absent from all drawn events.
  (h) Cross-event duplicates. Messages whose normalised text (`text_norm`) occurs in another event of the same
  corpus, counted per corpus; A^P of the two-epoch arms after removing those messages from the scored events.
  Reported as exploratory, whatever it shows.

## Deviations log

- 2026-09-28T18:40Z, execution only, before any outcome was read. The two streams (one per model)
  were restructured into four (model × corpus) to obtain a larger share of a GPU shared with other
  jobs. The two runs in flight (`humaid19/distilbert/seed42/id_temporal`,
  `humaid19/roberta/seed42/id_random_dedup`) were interrupted mid-training, recorded as failed
  executions, moved to `experiments/failed/`, and rerun from scratch with the same run key and seed.
  Seeds, conditions, recipe and analysis are unchanged; `code/run_confirm.sh` now takes the corpus as
  a second argument.
- 2026-09-29T23:48Z, execution only. The system volume filled (4 GB free of 926 GB; the largest recent
  growth was a 64.5 GB Cursor application state database outside this study, plus swap from unified-memory
  pressure). One confirmation run (`crisislext26/distilbert/seed42/chrono:2013_west_texas_explosion`)
  crashed before writing predictions (exit 120) and was recorded as failed and rerun. To relieve memory and
  swap pressure the two A2 extras streams and the learning-curve job were paused; their two in-flight runs
  (`humaid19/*/seed42/loeok:*`) were recorded as failed (no outcome) and will be rerun. Regenerable pip and
  uv caches (3.2 GB) were purged. No outcome was read; no scientific field changed.
- 2026-09-29, execution only. After a host terminal restart killed all background jobs (four in-flight runs
  recorded as failed, no outcome), every job was relaunched in a detached tmux session. To shorten the RoBERTa
  bottleneck, seeds 3 and 4 (HumAID-19) and seed 2 (CrisisLexT26) run in parallel helper processes; the runner
  now claims each run directory atomically so two processes never train the same run. Runs, seeds, splits and
  analysis are unchanged. The HumAID seed-3 and seed-4 RoBERTa helpers were stopped after ~4.5 h because the
  machine was oversubscribed (26 GPU processes across four studies, 16 GB swap) and their runs had stalled; their
  two in-flight runs were recorded as failed (no outcome) and are rerun by the main stream.
- 2026-09-30T08:05Z, execution only. The three long-lived streams still running optional arms (RoBERTa
  CrisisLexT26 mask seed 1, DistilBERT HumAID full-pool, RoBERTa four-epoch replicate) had accumulated 33 GB of
  cached MPS memory because the runner never released it between runs, which stalled the analysis. The runner now
  calls `gc.collect()` and `torch.mps.empty_cache()` after each run (no change to data, sampling, recipe or
  outputs); the streams were restarted and their three in-flight runs (none had written predictions) were
  recorded as failed and rerun. The frozen analysis script gained `--skip_incomplete_extras`, which leaves an
  optional arm out of a pass until all of its runs exist; registered base estimands are unaffected.
- 2026-09-30T10:54Z, execution only. The MPS backend reported "command buffer exited with error status
  (Internal Error)" in both running streams and the machine shut down and rebooted. The two in-flight runs
  (`confirm_ep4/crisislext26/distilbert/seed42/loeo:2013_bohol_earthquake`, X1;
  `confirm/humaid19/distilbert/seed42/loeo_full:hurricane_dorian_2019`) had written no predictions; they were
  recorded as failed, moved to `experiments/failed/mps-crash-2026-09-30/` and rerun with the same run key and seed.
  At relaunch (13:41Z) the remaining HumAID full-pool runs are shared by two worker processes on the same seed
  (atomic directory claim), which changes only the order in which runs execute. The host is an Apple M4 Max with
  64 GB of unified memory; earlier study notes that named an M2 Max were wrong.
- Timestamp erratum (2026-09-30, found by OPUS-R1-F12). Three hand-written times in this file disagree with the
  commits that introduced them; the commit times are authoritative and the text above is left unchanged. Freeze:
  "FROZEN 2026-09-28T18:40Z" was committed in 7de4ceb0 at 2026-09-28T18:24:49Z (first confirmation run started
  18:25:22Z). Amendment A1, headed 2026-09-29T19:15Z, was committed in a33b20e7 at 2026-09-28T19:10:09Z. Amendment
  A2, headed 2026-09-29T22:00Z, was committed in d49816ed at 2026-09-28T21:22:01Z. Both amendments therefore
  precede the first confirmation outcome read (2026-09-30T07:45Z, access log) by more than a day.
- 2026-09-30T16:58Z, execution only. The X2 RoBERTa seed-1 stream was started before the HumAID full-pool arm
  finished (its `--after-full-pool` wait was dropped) because memory allowed a sixth process; only the order in
  which runs execute changes.
- 2026-10-01T15:33Z, execution only, before any X7(a) outcome was read. Two further streams (`code/run_x2.sh
  roberta 3`, then `… roberta 4`; and `… roberta 4`) were started beside the two X7(a) streams, because the
  RoBERTa runs were the bottleneck and memory allowed four processes. Runs claim their output directory
  atomically, so no run is trained twice; seeds, run list, recipe and analysis are unchanged.
- 2026-10-01T22:27Z, execution only, before any X10 outcome was read. From the HumAID corpus onward, X10 batches
  the messages of each 2,048-message chunk in token-length order and restores corpus order before saving, to
  reduce padding. On the first 256 CrisisLexT26 messages (1,536 message-class pairs) the reordered logits equal
  the saved corpus-order logits exactly (`code/run_zeroshot.py check_order`: maximum absolute difference 0).
  Model, revision, hypotheses, truncation and prediction rule are unchanged.
