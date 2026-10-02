# Audit of evaluation-metric practice in cross-event crisis social-media text classification

Audit date: 2026-09-29. Companion file: `metric-practice-audit-2026-09-29.json` (one object per paper).

## 0. Purpose, rules and definitions

The study under preparation shows that scoring each held-out event with scikit-learn's default macro-F1 (`f1_score(average='macro')` with no `labels` argument) averages over every class found in `y_true` **or** `y_pred`. A few predictions of a class that does not occur in that event therefore add zero-valued terms. Averaging such per-event scores, and comparing the average with a pooled random-split score, inflates the reported cross-event gap. This audit checks how common each ingredient of that pattern is in published work, so that the paper claims only the prevalence the evidence supports.

Rules followed:
- Every fact below comes from a PDF, web page or code file opened in this session (2026-09-29).
- Anything not seen is marked **UNVERIFIED**.
- Code behaviour is reported only from lines actually read. Library semantics are quoted from the library documentation or source that was opened (scikit-learn 1.9.1 docs; torchmetrics v0.10.0 source).
- Quotes are verbatim from `pdftotext` output. Only line-break hyphenation and ligatures were repaired.

Library semantics used for the code column (opened this session):
- **scikit-learn `f1_score`**, `labels` parameter (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): "Labels not present in the data can be included and will be 'assigned' 0 samples. For multilabel targets, labels are column indices. By default, all labels in `y_true` and `y_pred` are used in sorted order."
- The same page, on zero division: "When `true positive + false positive + false negative == 0` (i.e. a class is completely absent from both `y_true` or `y_pred`), f-score is undefined. In such cases, by default f-score will be set to 0.0".
- **torchmetrics v0.10.0** (`src/torchmetrics/functional/classification/f_beta.py`, `_fbeta_compute`, line 777): `if average == AvgMethod.MACRO ...: cond = (tp + fp + fn == 0) | (tp + fp + fn == -3)`, then `num = num[~cond]`. Classes absent from both predictions and targets are dropped. A class absent from the targets but predicted (fp > 0) is kept with F1 = 0, which is the same effective behaviour as scikit-learn's default label set.
- `Metric.forward` (v0.10.0 `metric.py`, line 229): it "serves the dual purpose of both computing the metric on the current batch of inputs but also add[ing] the batch statistics to the overall accumulating metric state".

Aggregation vocabulary:
- **per-event**: a separate score for each held-out event (or source→target pair), not averaged.
- **mean of per-event**: the arithmetic mean of per-held-out-event scores. This is the target pattern when the metric is macro-F1.
- **pooled**: one score computed on the concatenation of several held-out events.
- **mean of pooled groups**: the mean over several held-out test sets, each of which pools several events (for example a held-out disaster type).
- **mean of per-batch**: the mean of scores computed on mini-batches of a test set.

## 1. Headline counts

22 papers were audited:
- 19 report cross-event results (held-out event, held-out event set, held-out type, or new-event/zero-shot). 18 of these were verified in full text; one (Livers et al. 2025) is abstract-only and UNVERIFIED.
- 3 are closely related but report **no** held-out-event classifier evaluation: Pramanick et al. 2022 (within-event temporal), CrisisBench (cross-dataset), and HumAID (in-event random splits). They are included because the task named them and they supply evidence on metrics and on class absence.

**Metrics reported in the text** (21 verified papers; a paper can fall in several rows):

| Metric | n | Papers |
|---|---|---|
| macro-F1 (multi-class, or macro over labels/information types) | 12 | Nguyen17 (arXiv version), MedinaMaza20, Seeberger22, Pramanick22, Seeberger25, Imran24, McDaniel24, CrisisMatch23, TREC-IS19, Alharbi22, DeCrisisMB23, CrisisTS25 |
| accuracy | 10 | Nguyen17 (arXiv version), CAST21, CrisisBench21, Imran24, CrisisMatch23, TREC-IS19, Krishnan20, Alharbi22, DeCrisisMB23, Yu19 |
| weighted-F1 | 6 | Alam18, CAST21, CrisisBench21, HumAID21, McDaniel24, Alharbi22 |
| binary / positive-class / per-class F1 as a reported quantity | 5 | MedinaMaza20 (Critical F1), Pramanick22 (binary F1 on Sandy), Kersten19 (per-class F1), McDaniel24 (binary F1), CrisisTS25 (per-class tables) |
| F1 whose averaging is not stated in the text | 5 | Sánchez23, Wiegmann20, Imran24 (Table 3 and event-wise Figure 9), Krishnan20 (the code shows weighted), Yu19 |
| AUC | 2 | Nguyen17, Alam18 |
| other | — | priority RMSE (TREC-IS19); misclassification rate on tranquil tweets (Wiegmann20) |

**Aggregation of cross-event results** (18 verified cross-event papers, primary scheme):

| Scheme | n | Papers |
|---|---|---|
| pooled over several held-out events | 7 | Seeberger22, Sánchez23, Seeberger25, Imran24 (overall and per type; per-event also shown), McDaniel24 (per source dataset), CrisisMatch23, TREC-IS19 (headline) |
| mean of per-held-out-event scores | 5 | MedinaMaza20 (macro-F1), Alharbi22 (macro- and weighted-F1), CAST21 (accuracy), Kersten19 (per-class F1), Krishnan20 (accuracy/F1) |
| per-event only (no cross-event average) | 3 | Nguyen17, Alam18, Yu19 |
| mean of pooled held-out groups | 2 | Wiegmann20, CrisisTS25 (macro-F1) |
| mean of per-mini-batch macro-F1 on a pooled test set | 1 | DeCrisisMB23 |

**Key counts**

- **Per-held-out-event macro-F1, averaged over events: 3 cross-event papers.** MedinaMaza20 and Alharbi22 do this in headline tables; Seeberger22 does it only in its Figure 4 analysis (mean ± s.d. of per-event F1 within an event type). Two more papers use the same pattern outside the strict definition:
  - Pramanick22 does it for within-event temporal splits, not cross-event.
  - CrisisTS25 averages macro-F1 over held-out *groups* of events or types.
- **Random-split or in-distribution score compared with a cross-event score on different test instances: 6.** These are CAST21 (CrisisT6), Seeberger22, Seeberger25, Kersten19, Imran24 and Yu19. Livers25 appears to do this from its abstract but is UNVERIFIED.
  - Comparisons made on the same instances: Nguyen17 (in-event vs out-of-event training), CAST21 (nepal_queensland), Wiegmann20 (type-specific vs generic; no random split), CrisisBench21 (cross-dataset) and Pramanick22 (within-event).
  - Not determinable: Alam18 and McDaniel24.
- **Released code computing default-label macro-F1** (no fixed class list, or equivalent semantics). Code was released and inspected for 10 papers.
  - It produces the reported macro-F1 in 2: Pramanick22 (scikit-learn, per event) and DeCrisisMB23 (torchmetrics 0.10 macro, averaged over mini-batches).
  - In 2 more (CAST21 and Sánchez23, both binary tasks), the default-label macro is computed but is not, or not verifiably, the reported number.
  - Seeberger22 calls `f1_score(average='macro')` without `labels`, but on multi-label indicator matrices, where the label set is fixed by the columns.
  - Two use a fixed class list: MedinaMaza20 (probable repository; custom macro over a fixed two-label map) and the TREC-IS official notebooks (fixed 25-type ontology).
- **Held-out events verifiably lacking some classes:**
  - HumAID's per-event experimental splits contain 6–10 of the 10 classes (Table 7 "# Cls"; Table 6 has "-" cells), because classes with fewer than 15 tweets in an event were removed. This bears on Pramanick22's per-event runs and Imran24's event-wise results. Whether those papers used exactly these splits is UNVERIFIED.
  - CrisisTS held-out types: zero-count cells in Tables 4–5.
  - TREC-IS per-event scoring: absent types produce "no true nor predicted samples" warnings in the rendered 2021 notebook.
  - MedinaMaza20 (repository data), Nguyen17, Kersten19, Wiegmann20, Seeberger25 (pooled), CrisisMatch23 and Sánchez23 verifiably do **not** lack classes, by counts or by construction.

**No audited paper shows the complete pattern.** The complete pattern is per-event default-label macro-F1, averaged, and compared with a pooled random-split score on different instances. The ingredients occur separately:
- Default-label per-event macro-F1 appears in Pramanick22, but its comparison is on the same test set.
- Different-instance random-vs-cross-event comparisons appear in Kersten19, Yu19 and Imran24, but these use per-class, binary or unstated F1, and in Kersten19 and Yu19 all classes are present.
- Averaged per-event macro-F1 appears in MedinaMaza20, but with a fixed two-label list and both classes present in every held-out event.

The paper should therefore claim that each ingredient is common, not that the combined artefact is documented in the literature.

## 2. Summary table

Abbreviations: PE = per-event; mPE = mean of per-event; PO = pooled; mPG = mean of pooled groups; mPB = mean of per-batch; ID = in-distribution; FT = full text.

| ID | Paper | Task (#classes) | Metrics reported | Cross-event aggregation | Random/ID vs cross-event; same instances? | Code; F1 call; `labels` passed? | Held-out event can lack classes? | Verified |
|---|---|---|---|---|---|---|---|---|
| P01 | Nguyen et al. 2017 (ICWSM; arXiv 1608.03902) | binary informative (ICWSM); 6-class humanitarian (arXiv) | AUC; accuracy and macro-F1 (arXiv) | PE (4 target events) | event-only vs out-of-event training on each event's 20% test split: same (implied) | none linked | no (every class ≥83 tweets per event, Table 1) | FT |
| P02 | Alam, Joty & Imran 2018 (ACL) | binary relevant (2) | weighted P/R/F1, AUC | PE (2 target events) | in-domain vs transfer; target test split **UNVERIFIED** (code takes `-T` file argument) | github.com/firojalam/domain-adaptation; support-weighted P/R/F1 parsed from `classification_report`; no macro; no `labels` | no (NEQ 5,527/6,141; QFL 5,414/4,619) | FT + code |
| P03 | Medina Maza et al. 2020 (Findings EMNLP) | binary critical (2) | macro-F1, Non-Critical F1, Critical F1 | **mPE** (LOEO within disaster type) | no ID comparison | probable repo salmedina/CriticalTweetsClassifier (paper link 404); custom macro over fixed 2-label map (`classifier.py` L325–339, L348) | no (all 25 held-out events have ≥13 critical; repo data) | FT + probable code |
| P04 | Wang, Nulty & Lillis 2021 CAST (ISCRAM) | binary (2) | weighted-F1 (nepal_queensland); accuracy (CrisisT6) | PE; mPE (CrisisT6 LOO average accuracy) | CrisisT6: in-domain 5-fold CV vs whole target event: **different**; nepal_queensland: same test file | github.com/wangcongcong123/CAST; `precision_recall_fscore_support(labels, preds, average='macro')`, **no `labels`** (on generated strings); macro not reported | no (both classes "well-balanced") | FT + code |
| P05 | Seeberger & Riedhammer 2022 (NLP4PI) | multi-label TREC-IS (25 low / 4 high) | macro-F1 over information types | PO (21 test events); Fig. 4 **mPE** per event type | dev = stratified 10% of train events vs test events: **different** | github.com/seebergerph/EMLM; `partial(f1_score, average='macro')`, **no `labels`**, on 2-D indicator arrays (column set fixed) | UNVERIFIED per event | FT + code |
| P06 | Pramanick et al. 2022 (Findings EMNLP) — within-event temporal | Sandy binary; T26 4-class; HumAID (11-way head in code) | binary F1; macro-F1 | PE; mean of per-event differences (Table 1) | CONTROL vs TEMPORAL on **same** test set | github.com/UKPLab/emnlp2022-temporal-adaptation; `f1_score(p.label_ids, preds, average='macro')`, **no `labels`**, per event | **yes** (HumAID per-event splits keep 6–10 of 10 classes, HumAID Tables 6–7; split identity UNVERIFIED) | FT + code |
| P07 | Sánchez et al. 2023 (ICWSM) | binary related (2) | "F1-score" (averaging unstated) | PO per target (language × domain) | monodomain vs cross-domain on same target test set; no random split | github.com/cinthiasanchez/Crisis-Classification; computes `average='binary'`, `'macro'` (no `labels`), `'micro'` | no (Table 4: both classes in every target) | FT + code |
| P08 | Seeberger, Freisinger, Bocklet & Riedhammer 2025 (Findings IJCNLP-AACL) | HumAID 9; CrisisLexT26 7; TREC-IS 7 (multi-label) | macro P/R/F1 | PO (4 / 6 / 9 test events) | probe on train-event encodings vs test-event Table 1: **different** | none found | no in pooled test sets (Table 4) | FT |
| P09 | Kersten et al. 2019 (ISCRAM) | binary related (2) | per-class P/R/F1 (± s.d.) | mPE (mean ± s.d. over held-out events) | random-excluded 10% vs unseen event: **different** | none linked | no (balanced by sampling unrelated tweets) | FT |
| P10 | Wiegmann et al. 2020 (ISCRAM) | binary related (2) | F1, P, R, MCR (averaging unstated) | mPG (per-type test sets pooling disasters; averaged over models/CV) | type-specific vs generic on **same** test sets; no random split | none linked | no (balanced with tranquil tweets) | FT |
| P11 | Alam et al. 2021 CrisisBench (ICWSM) — cross-dataset, not cross-event | informativeness 2; humanitarian 11 | weighted P/R/F1; accuracy | PO per source dataset (random 70/10/20) | in-dataset vs cross-dataset on **same** test set | github.com/firojalam/crisis_datasets_benchmarks; `metrics.f1_score(y_true,y_pred,average="weighted")`, no `labels` | per-dataset label sets differ; paper reports "only those classes the model is able to classify" | FT + code |
| P12 | Alam et al. 2021 HumAID (ICWSM) — no cross-event | humanitarian (10 after removal) | weighted P/R/F1 | PE, event-type and All on random splits | none | not checked | **yes** (# Cls per event 6–10 after removing classes with <15 tweets; "-" cells in Table 6) | FT |
| P13 | Imran et al. 2024 (arXiv 2412.10413) | 9 categories (ORI dropped) | macro-F1 (stated for type/class/hashtag analyses); F1/acc/P/R (Table 3, averaging unstated) | PE (Fig. 9) + PO (overall; per type) | LLM F1 vs "RoBERTa (F1=0.78)" from HumAID, which equals HumAID's RoBERTa "Average" row (0.781, weighted-F1 over 24 test sets), not its "All" row (0.760): **different** | none found | **yes** for HumAID per-event splits (Tables 6–7); event-level presence in Imran's test split not recomputed | FT |
| P14 | McDaniel, Scheele & Liu 2024 (arXiv 2410.00182) | informativeness 2; humanitarian 16 (prompt) | macro, weighted, binary F1 | PO per source dataset | LLM vs fine-tuned benchmark numbers from other papers; instance identity **UNVERIFIED** | none found | handled: "performance assessment of each dataset is based exclusively on the rankings of the labels present within that dataset" | FT |
| P15 | Zou, Zhou, Caragea & Caragea 2023 CrisisMatch (ISCRAM) | 7 | accuracy, macro-F1 | PO (Floods test, 970) | in-domain and OOD on different test sets; not directly contrasted | none found ("We will release…") | no (classes <100 discarded) | FT |
| P16 | Livers et al. 2025 (ICDM Workshops) | 4 SVM tasks | **UNVERIFIED** | LODO / "Disaster-Holdout Validation" (abstract) | abstract: random splits "inflate performance"; instances **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | abstract only |
| P17 | TREC-IS (McCreadie, Buntain & Soboroff 2020, ISCRAM; official eval notebooks 2019-A v2 and 2021-B v3.0) | multi-label, 25 information types | Info-type positive-class F1 macro over types (All/Actionable), accuracy, priority RMSE | PO: "micro-averaged over all events but macro-averaged over information types"; notebooks also emit PE | none | official notebooks: per-type `f1_score(..., average='binary')` summed over the fixed ontology ÷ `len(informationTypes2Index)` (fixed list) | **yes** in PE output (UndefinedMetricWarning "no true nor predicted samples") | FT + eval code |
| P18 | Krishnan, Purohit & Rangwala 2020 (ASONAM) | 4 binary tasks | accuracy, F1 | PE + mPE (AVG row) over 10 LOEO events | none | github.com/jitinkrishnan/Crisis-Tweet-Multi-Task-DA; `f1_score(y_true, y_pred, average='weighted')`, no `labels` | no (events trimmed to ≥10 positives per task) | FT + code |
| P19 | Alharbi & Lee 2022 (OSACT @ LREC) | binary relevancy; multi-class information types (≥5; exact count UNVERIFIED) | weighted-F1 + macro-F1; accuracy + macro-F1 | PE + **mPE** (7 LOO target events) | none | none found | rare, not absent (Cairo bombing: 700 relevant / **6** irrelevant) | FT |
| P20 | Zou, Zhou, Zhang & Caragea 2023 DeCrisisMB (Findings EMNLP) | 8 | accuracy, macro-F1 | **mPB** on pooled OOD test set | none | github.com/HenryPengZou/DeCrisisMB; `F1Score(num_classes=n_labels, average='macro')` (torchmetrics 0.10.0), batch values summed ÷ `len(loader)` | test set balanced (160 per class); per-batch coverage not checked | FT + code |
| P21 | Meunier et al. 2025 CrisisTS (ACL) | urgency 3; utility 2; humanitarian/intent 6+ | macro F-score, per-class F | **mPG** (3 held-out event sets × 3 runs; held-out types) | none | data only on HF (no evaluation code) | **yes** (zero cells for held-out types, Tables 4–5) | FT |
| P22 | Yu et al. 2019 (Int. J. Digital Earth) | 5 topics | P/R/F1 (averaging unstated); accuracy | PE (4 source→target pairs) | single-event random 80/20 vs cross-event whole target: **different** | none linked | no evidence of absence | FT |

## 3. Per-paper notes

### P01 — Nguyen, Al-Mannai, Joty, Sajjad, Imran & Mitra (2017). Robust Classification of Crisis-Related Data on Social Networks Using Convolutional Neural Networks. ICWSM 11(1).
Opened: https://ojs.aaai.org/index.php/ICWSM/article/download/14950/14800 (4 pp.) and https://arxiv.org/pdf/1608.03902 (10 pp., extended version).
- **Protocol.** "Given a particular event (e.g. Nepal earthquake), we use data from all other events plus All others (see Table 1) as out-of-event data. We divide each event dataset into train (70%), validation (10%) and test sets (20%)". "For each event under consideration, we train classifiers on the event data only, on the out-of-event data only, and on a combination of both."
- **Metrics.** The ICWSM version reports "Table 2: The AUC scores of non-neural and neural network-based classifiers". The arXiv version adds multi-class results: "Table IV summarizes the accuracy and macro-F1 scores", with values per event (e.g. Nepal Mout CNNI macro-F1 0.51 vs Mevent 0.57).
- **Aggregation.** Per event only (4 events).
- **Comparison.** Event-only vs out-of-event training, evaluated per event. That the test split is shared is implied by the design, not stated explicitly.
- **Absent classes.** No. Every event has ≥83 tweets in every class (Table 1; e.g. California Earthquake: Donations 83, Sympathy 83).
- **Code.** No repository linked in either PDF.

### P02 — Alam, Joty & Imran (2018). Domain Adaptation with Adversarial Training and Graph Embeddings. ACL 2018, pp. 1077–1087.
Opened: https://aclanthology.org/P18-1099.pdf and repository https://github.com/firojalam/domain-adaptation (commit 24b36d7).
- **Task.** "In all the experiments, the classification task consists of two classes: relevant and non-relevant." Split: "60% as training, 30% as test and 10% as development."
- **Metric.** "we use weighted average precision, recall, F-measure, and Area Under ROC-Curve (AUC) … The rationale behind choosing the weighted metric is that it takes into account the class imbalance problem."
- **Results.** Table 4 is per source→target pair, e.g. in-domain Nepal→Nepal F1 60.89 vs Queensland→Nepal 53.63.
- **Target test set.** Not stated in the text for the transfer rows. In `cnn/cnn_domain_adaptation.py` (L155–192) the target set is `domain_test_file` taken from option `-T`, so same-instance evaluation is **UNVERIFIED**.
- **Code.** `cnn/performance.py` `performance_measure_tf` (L53–75) computes AUC as `metrics.roc_auc_score(y_true,y_pred)` on hard labels. P/R/F1 come from `classifaction_report(report)` (L89–117), which support-weights the rows of `classification_report`. There is no macro-F1 and no `labels` argument. The README says: "We will make the code publicly available soon."
- **Absent classes.** No: NEQ 5,527 relevant / 6,141 non-relevant; QFL 5,414 / 4,619 (Table 1).

### P03 — Medina Maza, Spiliopoulou, Hovy & Hauptmann (2020). Event-Related Bias Removal for Real-time Disaster Events. Findings of EMNLP 2020, pp. 3858–3868.
Opened: https://aclanthology.org/2020.findings-emnlp.344.pdf; the footnote link https://salmedina.github.io/EventBiasRemoval/ returns HTTP 404 (checked today); the author repository https://github.com/salmedina/CriticalTweetsClassifier (commit e1f310c; created 2019-09-19, last push 2020-06-01; not linked from the paper, so it is a *probable* release).
- **Protocol.** "the training data consists of all the events of the same disaster type except one, as it is used for testing the model. We generated n splits for each event type … We evaluated the three models on each split obtaining the macro-F1 and the micro-F1 scores from the cr predictions. Finally, we calculated the mean of these metrics, which we can see in Table 2."
- **Result.** This is a **mean of per-held-out-event macro-F1**. Table 2 columns are "Macro F1 / Non-Critical F1 / Critical F1".
- **Code (probable repository).** `src/classifier.py` `calc_metrics` (L325–339) loops over every label in `label_map` and divides by `len(final_metrics)`. Counts are initialised as `{'correct': 0.0, 'gold': 0.0001, 'predicted': 0.0001}` for each label (L348). The label list is therefore fixed at two classes, not taken from the data.
- **Model selection.** The held-out event is listed under `valid:` in `data/experiments/*.yaml` (e.g. `earthquake_1.yaml` holds out `2012_Italy_earthquakes`). The best epoch is chosen on its Critical F1 (L265–279). This matches the paper: "considered as the best model the one which showed the highest Critical-F1 score".
- **Absent classes.** No. From `data/labeled_data.json`, all 25 held-out events contain the critical class; the minimum is 2014 Chile Earthquake with 13 critical of 311.

### P04 — Wang, Nulty & Lillis (2021). Crisis Domain Adaptation Using Sequence-to-Sequence Transformers (CAST). ISCRAM 2021, pp. 655–666.
Opened: https://lill.is/pubs/Wang2021a.pdf (the ISCRAM library returned 503) and https://github.com/wangcongcong123/CAST (commit 99a780e).
- **Metrics.** "the weighted F1 scores are reported for nepal_queensland (Alam et al. 2018) and the accuracy scores are reported for CrisisT6 (Liu et al. 2020)".
- **In-domain vs cross-domain.** "we use 5-fold cross-validation evaluation whenever the in-domain performance is reported for CrisisT6." The code tests cross-domain on the entire target event: `encoded["test"] = encoded_dataset["train"].filter(lambda example: example["event_name"] == EVENTSHORT2LONG[test_e])` (`train_t6.py` L237–245). In-domain uses `KFold(n_splits=5)` inside the event (L211–225).
  - The two sides are **different instances**, and Figure 3 shows them in one matrix: "the diagonals refer to the in-domain performance and the rest are cross-domain scores".
  - For nepal_queensland, the paper says "We use the standard train, validation and test splits from Alam et al. (2018)".
- **Aggregation.** "Table 4. Leave-one-out cross-domain adaptation accuracy using CrisisT6. The last row reports the average score."
- **Code.** `metric_scripts/cls.py` L34 calls `precision_recall_fscore_support(labels, preds, average='macro')` with **no `labels`**. Its inputs are decoded T5 output strings (`train_t6.py` L55–65), so the label set is whatever strings occur in either list. The macro value is logged but is not the reported metric.

### P05 — Seeberger & Riedhammer (2022). Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning. NLP4PI 2022, pp. 70–78.
Opened: https://aclanthology.org/2022.nlp4pi-1.9.pdf. The paper links https://github.com/th-nuernberg/crisis-tapt-hmc, whose README says "Repository moved to … https://github.com/seebergerph/EMLM"; that repository was inspected (commit 9f02ec3).
- **Protocol.** "we split the dataset into train and test events which corresponds to the TREC-IS 2020B task". Table 1: train events 1–52, test events 53–75 (21 events, 22,003 tweets).
- **Metric.** "We follow the TREC-IS evaluation scheme: macro-averaged F1-score across information types".
- **In-domain proxy.** Hyper-parameters use "a stratified split with a ratio of 90% for train and 10% for development data". Then: "we additionally analyzed the development set, as a proxy to estimate the in-domain event performance … Contrary to the test set, standard MLM increases the absolute LB performance by 2% whereas the E-MLM approach drops by 5%". This compares **different instances**.
- **Per-event analysis.** Figure 4: "Comparison across event types w.r.t. F1-score … We plot the mean and standard deviation for multiple events within a event type." This is a per-event score averaged within type; the caption does not restate the averaging.
- **Code.** `src/evaluation/metrics.py` L15: `'f1_macro': partial(f1_score,  average='macro')`, with **no `labels`**. `src/evaluation/scoring.py` `_filter_actionable` slices `preds[:, actionable_indices]`, so the inputs are 2-D indicator arrays, and scikit-learn fixes the label set to the columns. `src/eval.py` scores one pooled test-prediction file; no per-event scoring code was found.

### P06 — Pramanick, Beck, Stowe & Gurevych (2022). The challenges of temporal alignment on Twitter during crises. Findings of EMNLP 2022, pp. 2658–2672. *(Within-event temporal, not cross-event.)*
Opened: https://aclanthology.org/2022.findings-emnlp.195.pdf and https://github.com/UKPLab/emnlp2022-temporal-adaptation (commit 331bc3a).
- **Metric.** "We report binary-F1 Score for Sandy and macro-F1 score for multi-class classification task on T26 and Humaid datasets."
- **Same-test comparison.** CONTROL: "We use the same test set as in TEMPORAL setup". Table 1: "Averaged F1 performance difference of the CONTROL to TEMPORAL setting". Per-event results are in Tables 8–9.
- **Code.**
  - `Baseline.py` L69–75: "For multi-event datasets, loop over each event."
  - `Experiments.py` L234 (`compute_metrics`, used for test scores at L176–178): `'f1_macro': f1_score(p.label_ids, preds, average='macro')`, with **no `labels`**. L72 (`compute_metrics_da`) has the same call.
  - `CONST.py` L28: `N_LABELS = {"sandy":2, "clex":4, "humaid":11}`. L43 maps 11 strings for HumAID, one of which is the literal `'class_label': 6`.
- **Absent classes.** Yes, very likely. HumAID's per-event experimental splits keep 6–10 of the 10 classes (HumAID Table 7 "# Cls"; Table 6 "-" cells), while the head has 11 outputs. TEMPORAL tests on "a randomly sampled 50% of data from the second temporal half", a subset of the event. That Pramanick used exactly the HumAID released splits is UNVERIFIED; the code reads files from `datasets/humaid/humaid/<event>`.
- **Observation, not interpreted.** In Table 9, "Sri Lanka Floods (2017)" is 0.092 in all 12 CONT/TEMP cells.

### P07 — Sánchez, Sarmiento, Abeliuk, Pérez & Poblete (2023). Cross-Lingual and Cross-Domain Crisis Classification for Low-Resource Scenarios. ICWSM 17(1), 754–765.
Opened: https://ojs.aaai.org/index.php/ICWSM/article/download/22185/21964 and https://github.com/cinthiasanchez/Crisis-Classification (commit 1ea5d67).
- **Protocol.** "each event was distributed into a training or test set. … training sets prioritized events with the highest and most balanced number of instances". Test sets are pooled by target language and domain (Table 4). For negatives, "we opted to augment the negative class by including a random sample of negative instances from the complete testing set (i.e., not only from the test event for that scenario)".
- **Metric.** Only "F1-score"; the averaging is not stated in the text read.
- **Code.** `src/report.py` computes `f1_score(y_test, predictions, average='binary')` (L17). It also computes `compute_score_threshold(probs, y, f1_score, {'average':'macro'})` (L70–71) and `f1_score(y, (prob >= .5) *1., average='macro')` (L91), with no `labels`.
- **Which F1 is reported (UNVERIFIED).** `results/tabulated results.md` says "We show the F1-score below. This score is the average of 5 executions." The English Flood / Monodomain / LF cell is 0.84. In `results/files/RF_LF_1.csv`, `test_f1` (binary) is 0.8389 and `test_f1_macro` is 0.8283. That file may be a single run, so which averaging the paper reports remains UNVERIFIED.

### P08 — Seeberger, Freisinger, Bocklet & Riedhammer (2025). Generalizing to Unseen Disaster Events: A Causal View. Findings of IJCNLP-AACL 2025, pp. 28–37.
Opened: https://aclanthology.org/2025.findings-ijcnlp.2.pdf.
- **Protocol.** "we use a temporal split strategy and divide the events into disjoint sets according to the provided timestamps". Table 4 gives the train/validation/test event counts: HumAID 12/3/4, CrisisLex 15/5/6, TREC-IS 16/8/9.
- **Metric.** "We use macro-averaged precision (P), recall (R), and F1 scores". Table 1: "averaged and tested over the same five seeds".
- **Aggregation.** Pooled over test events. Every class has test instances in Table 4; the minimum is 74 for TREC-IS requests.
- **Different-instance comparison.** "the baseline model achieves the highest main task performance for train events but worse performance for test events (see Table 1)". The train-event side is a probe that fits "a shallow classifier using only a small subset of the samples (5%)" of training-set encodings.
- **Code.** No repository found (checked `seebergerph/*` and `th-nuernberg/*`).

### P09 — Kersten, Kruspe, Wiegmann & Klan (2019). Robust Filtering of Crisis-related Tweets. ISCRAM 2019.
Opened: https://downloads-cf.webis.de/publications/papers/kersten_2019.pdf.
- **Protocol.** "we exclude 10 % randomly sampled tweets from CV for independent testing. In order to test the the models' transferability, each of the events is successively excluded for training local and global models, and is instead used for testing."
- **Metric and aggregation.** Table 2: "Local and global models tested with randomly excluded and unseen event data". It reports per-class Precision/Recall/F1 for Related and Unrelated as mean ± s.d. For example, flood CNNl_re Related F1 is 0.915±0.039 vs CNNl_ue 0.875±0.054.
- **Different instances.** Randomly excluded data vs data from unseen events.
- **Absent classes.** No: "Perfect class balance is achieved by randomly sampling the required amount of unrelated tweets from the Events2012 data set."
- **Code.** None linked.

### P10 — Wiegmann, Kersten, Klan, Potthast & Stein (2020). Analysis of Detection Models for Disaster-Related Tweets. ISCRAM 2020, pp. 872–880.
Opened: https://elib.dlr.de/137213/1/2278_MattiWiegmann_etal2020.pdf.
- **Protocol.** "We trained a model on the sampled tweets for each disaster type and tested it against all tweets from all other disasters of the same type not selected for training … 6-fold Monte Carlo cross-validation." Also: "We tested all 8 same-type models and the generic model on each of the 10 test sets".
- **Metrics.** F1, precision, recall and MCR (Tables 3–5), reported as averages. The averaging of F1 is not stated in the text read.
- **Absent classes.** No: "we removed all negative examples from the datasets and filled them up to balance with tweets from the tranquil sample."
- **Code.** None linked (only TF-Hub model links).

### P11 — Alam, Sajjad, Imran & Ofli (2021). CrisisBench. ICWSM 2021. *(Cross-dataset, not held-out event.)*
Opened: https://ojs.aaai.org/index.php/ICWSM/article/download/18115/17918 and https://github.com/firojalam/crisis_datasets_benchmarks (commit 2dc70b4; the paper links https://crisisnlp.qcri.org/crisis_datasets_benchmarks.html, which was not opened).
- **Split and metric.** "We split data into train, dev, and test sets with a proportion of 70%, 10%, and 20%". "we use weighted average precision (P), recall (R), and F1-measure (F1)."
- **Same-instance comparison.** "there is 14.3% difference in F1 on CrisisNLP data using the CrisisLex model". This is CNLP→CNLP 0.832 vs CLex→CNLP 0.689 on the same CNLP test set.
- **Label-set handling.** "We report the results of those classes only for which the model is able to classify."
- **Code.** `bin/performance.py` L50–52: `F1=metrics.f1_score(y_true,y_pred,average="weighted")`, with no `labels`.

### P12 — Alam, Qazi, Imran & Ofli (2021). HumAID. ICWSM 15(1), 933–942. *(No cross-event experiments.)*
Opened: https://ojs.aaai.org/index.php/ICWSM/article/download/18116/17919.
- **Protocol.** "We removed low prevalent classes (i.e., number of tweets with a class label less than 15 …)". "The purpose of event and event type level experiments is to provide a baseline, which can be used to compare cross event experiments in future studies."
- **Metric.** "we use weighted average precision (P), recall (R), and F1-measure".
- **Class absence (used by P06 and P13).** Table 7: "The column # Cls reports number of class labels available after removing low prevalent class labels". It ranges per event from 6 (2016 Italy Earthquake) to 10 (2018 California Wildfires, 2019 Cyclone Idai); "All" has 10. Table 6 shows "-" (no train/dev/test tweets) for removed classes. RoBERTa is 0.760 on All; the "Average" row is 0.781.
- **Code.** Not checked.

### P13 — Imran, Ziaullah, Chen & Ofli (2024). Evaluating Robustness of LLMs on Crisis-Related Microblogs across Events, Information Types, and Linguistic Features. arXiv 2412.10413.
Opened: https://arxiv.org/pdf/2412.10413.
- **Data.** "We drop the 'other relevant information' class … We use the test split (N=15,160)".
- **Metrics.**
  - "Figure 2(a) shows the macro F1-scores" (per disaster type, pooled). "Figure 9 shows the event-wise F1-scores" (per event).
  - Table 3 gives "F1-score, Accuracy, Precision, and Recall" overall; averaging not stated.
- **Different-instance comparison.** "Figure 10 shows the F1-scores … and the SOTA supervised baseline (i.e., RoBERTa F1=0.78) as we report in [3]". In HumAID, RoBERTa is 0.760 on "All" and 0.781 on "Average", the mean of weighted-F1 over the 19 event, 4 type and All test sets. The 0.78 therefore matches the multi-test-set average, not the test split used by the LLMs, and it is a weighted-F1.
- **Absent classes.** Yes, for HumAID per-event splits (HumAID Tables 6–7). Presence per event in Imran's N=15,160 test split was not recomputed.
- **Code.** None found.

### P14 — McDaniel, Scheele & Liu (2024). Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models. arXiv 2410.00182.
Opened: https://arxiv.org/pdf/2410.00182.
- **Metrics.** "For the informativeness classification, we calculate macro (unweighted), weighted, and binary (only the positive class) F1 scores; for the humanitarian classification, we calculate weighted and macro (unweighted) class-averaged F1 scores."
- **Label-set handling.** "Given that some datasets in CrisisBench may cover only a subset of the 16 labels specified in the prompt … the performance assessment of each dataset is based exclusively on the rankings of the labels present within that dataset."
- **Comparison with benchmarks.** "weighted F1 of 0.828 (LLM: GPT-4o) vs. 0.883 (fine-tuned RoBERTa)" and "on CrisisLexT26 [11] (macro F1 of .492 vs .848)". The benchmark numbers come from other papers, and "Responses that were not valid were omitted from analysis", so whether the instances match is **UNVERIFIED**.
- **Code.** None found.

### P15 — Zou, Zhou, Caragea & Caragea (2023). Semi-Supervised Few-Shot Learning for Fine-Grained Disaster Tweet Classification (CrisisMatch). ISCRAM 2023.
Opened: https://arxiv.org/pdf/2310.14627.
- **Data.** "We use Earthquake and Wildfires datasets for in-domain evaluation and Floods dataset for out-of-domain evaluation." "we discard those classes with less than 100 data and use the 7-class version".
- **Metrics.** "We use accuracy and macro-F1 as our evaluation metrics." OOD results are in Table 5 (Floods test, 970 tweets, pooled).
- **Code.** "We will release our sampled and processed datasets". No CrisisMatch repository found under HenryPengZou (the repo list was checked).

### P16 — Livers, Johnson, Spurlock & Nasraoui (2025). When Splits Matter: Interpreting Disaster Tweet Classification Models. ICDM Workshops 2025, pp. 2844–2851.
Opened: https://exa.ai/library/publication/0qll0wvnbwr (abstract) and https://researchr.org/publication/LiversJSN25 (metadata).
- **Abstract.** "random data splits, can inflate performance by allowing disaster-specific vocabulary to leak … Disaster-Holdout Validation (DHV), extending the Leave-One-Disaster-Out (LODO) strategy".
- **Status.** Metrics, aggregation, instance identity and code are all **UNVERIFIED**; the full text was not accessible.

### P17 — TREC-IS. McCreadie, Buntain & Soboroff (2020). Incident Streams 2019: Actionable Insights and How to Find Them. ISCRAM 2020, pp. 744–760; plus the official evaluation notebooks.
Opened: the 2019 overview (text via https://eprints.gla.ac.uk/210955/1/210955.pdf); http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/TREC-IS_V2_2018Events_2019AFormat_Evaluation_Notebook.ipynb; and the rendered 2021-B notebook "TRECIS-EvaluationNotebook.v3.0.2021.all" at https://trec.nist.gov/pubs/trec30/appendices/incident/ucdcs-mtl.ens.html.
- **Headline metric (overview).** "we calculate metrics, like accuracy, micro-averaged over all events but macro-averaged over information types." Also: "Info. Type Positive F1, All … The 'All' version of this metric macro-averages over all information types". Each edition releases new events.
- **Overall cell (both notebooks).** "# Does not average across events (larger events have more impact)". It loops `for categoryId in informationTypes2Index.keys(): categoryF1 = f1_score(category2GroundTruth[categoryId], category2Predicted[categoryId], average='binary')` and divides by `numInformationTypes`. The ontology is fixed.
- **Per-event cell, 2019-A (cell 6).** Sums over all types and prints `avgF1/len(informationTypes2Index)`.
- **Per-event cell, 2021-B.**
  - It skips types with `if sum(event2groundtruth[eventId].get(categoryId)) == 0: continue` and prints `tavgF1/categoryCount`.
  - But it **writes** `tavgF1/len(informationTypes2Index)` to the per-event file (rendered-page lines 1533–1563).
  - The "Per Event F1 Graph" divides by `len(informationTypes2Index)` over all types (lines 1779–1785).
  - The rendered output shows "UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no true nor predicted samples".
  - So per-event TREC-IS values carry zero terms for types absent from the event, and the script prints and writes two different per-event values.

### P18 — Krishnan, Purohit & Rangwala (2020). Unsupervised and Interpretable Domain Adaptation to Rapidly Filter Tweets for Emergency Services. ASONAM 2020.
Opened: https://arxiv.org/pdf/2003.04991 and https://github.com/jitinkrishnan/Crisis-Tweet-Multi-Task-DA (commit 8a775df).
- **Protocol.** "we trimmed the events and tasks such that there are at least 10 positive samples for each task." Four binary tasks; leave one event out over 10 TREC 2018 events.
- **Results.** Table V gives per-event Acc and F1 with an "AVG" row, and "EACH REPORTED SCORE IS AN AVERAGE OF 10 INDEPENDENT RUNS".
- **Code.** `models.py` L238: `f1 = f1_score(y_true, y_pred, average='weighted')`, with no `labels`.

### P19 — Alharbi & Lee (2022). Classifying Arabic Crisis Tweets using Data Selection and Pre-trained Language Models. OSACT 2022 @ LREC, pp. 71–78.
Opened: https://aclanthology.org/2022.osact-1.8.pdf.
- **Protocol.** "We experimented with different source and target crisis pairs using the leave-one-out strategy … We report the average score. We used the weighted F1 and macro F1 to evaluate the models' performance on the relevancy detection task … For information classification, we used the accuracy … and macro F1 score."
- **Aggregation.** Table 3 is per target event, and the text reports "the average weighted F1 and macro F1". This is a **mean of per-event macro-F1**.
- **Rare classes.** Table 1 shows Cairo bombing (CB) with 700 relevant and 6 irrelevant tweets.
- **Code.** None found.

### P20 — Zou, Zhou, Zhang & Caragea (2023). DeCrisisMB: Debiased Semi-Supervised Learning for Crisis Tweet Classification via Memory Bank. Findings of EMNLP 2023, pp. 6104–6115.
Opened: https://www.cs.uic.edu/~cornelia/papers/emnlp23c.pdf and https://github.com/HenryPengZou/DeCrisisMB (commit 04afb44).
- **Paper.** Table 4, "Out-of-distribution results. Average over 3 runs." (ThreeCrises↔Hurricane), reporting accuracy and macro-F1. The dataset has 8 classes and "Test: 160" per class (Table 1).
- **Code (`main_ODomain.py`).**
  - L511: `f1 = F1Score(num_classes=n_labels, average='macro')`.
  - L563: `total_eval_f1 += f1(preds, target).item()`, inside the batch loop.
  - L573: `avg_val_f1 = total_eval_f1 / len(loader)`.
  - L422: `target_test_loader = DataLoader(target_test_dataset, batch_size=bs, shuffle=True)`, with `bs = 32` (L71).
  - `requirements.txt`: `torchmetrics==0.10.0`.
- **Effect.** The reported macro-F1 is a mean of per-mini-batch macro-F1 values. The torchmetrics 0.10.0 macro rule drops classes with `tp+fp+fn == 0` and keeps predicted-but-absent classes at 0, the same as scikit-learn's default label set.

### P21 — Meunier, Benamara, Moriceau, Qiao & Ramasamy (2025). CrisisTS. ACL 2025, pp. 16082–16099.
Opened: https://aclanthology.org/2025.acl-long.783.pdf and the file list at https://huggingface.co/api/datasets/Unknees/CrisisTS/tree/main (data and `Linker_*.py` only).
- **Protocol.** "(1) Out-of-event … The models are successively tested on each of these three sets while trained on all events that are not present in the chosen test set … The results are then the mean of the results obtained for each experiments." "(2) Out-of-type … the average of n runs, each run with n − 1 crisis types for training and the remaining crisis type for testing."
- **Metric.** "Table 1: Results on the French dataset in terms of average F-score"; "Table 3: Macro F1-scores".
- **Absent classes (Table 4, French).** Attack: MAT-DMG 0, ADV_WARN 0. Explosion: ADV_WARN 0, NOT USEFUL 0.
- **Absent classes (Table 5, English).** Flood and Fire: NOT USEFUL "N/A".

### P22 — Yu, Huang, Qin, Scheele & Yang (2019). Deep learning for real-time social media text classification for situation awareness – using Hurricanes Sandy, Harvey, and Irma as case studies. Int. J. Digital Earth 12(11), 1230–1247.
Opened: https://par.nsf.gov/servlets/purl/10139179.
- **Protocol.** "For single event experiment, we randomly select from the single-event dataset to generate training (80%) and validation (20%) sets. For cross-event experiment, we train the classifiers using the dataset from one event and test the others."
- **Metrics.** Five classes. "The overall or averaged accuracy scores are demonstrated in Table 3"; the rows are Precision, Recall and F1; averaging is not stated.
- **Different instances.** Irma single-event CNN F1 0.79 (Table 3) vs Sandy→Irma 0.78 on all Irma tweets (Table 4).
- **Code.** None linked.

## 4. Three clearest examples

1. **Pramanick et al. 2022 (code and data): the default-label per-event macro-F1 pattern.**
   - The model is trained and tested inside each event (`Baseline.py` L69–75) with an 11-way head (`CONST.py` L28).
   - The test score is `f1_score(p.label_ids, preds, average='macro')` with no `labels` (`Experiments.py` L234).
   - HumAID's per-event splits keep only 6–10 of the 10 classes (HumAID Tables 6–7), and Table 1 averages per-event F1 differences.
   - Caveat: this is within-event temporal evaluation, and its CONTROL/TEMPORAL contrast is on the same test set. The zero-term mechanism can act, but the pooled-vs-per-event inflation cannot.
2. **DeCrisisMB 2023 (code): macro-F1 averaged over subsets that lack classes.**
   - The reported out-of-distribution macro-F1 is the mean over shuffled 32-tweet mini-batches of torchmetrics-0.10 macro-F1, whose rule keeps predicted-but-absent classes at 0 (`main_ODomain.py` L511, L563, L573; torchmetrics `f_beta.py` L777).
   - This is the same mechanism as per-event averaging, applied at batch level.
3. **TREC-IS official evaluation notebooks: per-event macro over a fixed ontology.**
   - Per-event Information-Type F1 divides a sum of per-type binary F1 by all 25 types, so types absent from the event score 0.0; the rendered 2021 run shows the scikit-learn warning.
   - The 2021-B script prints a skip-absent per-event value but writes a divide-by-all value.
   - Caveat: the headline TREC-IS metric is pooled across events.

Clearest different-instance random-vs-cross-event comparisons:
- Kersten et al. 2019: randomly excluded 10% vs unseen events, per-class F1.
- Yu et al. 2019: single-event random 80/20 vs whole later event.
- Imran et al. 2024: LLM F1 on the HumAID test split vs "RoBERTa (F1=0.78)", which equals HumAID's multi-test-set weighted-F1 "Average".

Important non-example: Medina Maza et al. 2020 averages per-held-out-event macro-F1, but its (probable) code uses a fixed two-label list, and every held-out event contains both classes.

## 5. What the audit supports for the paper's prevalence claim

Supported:
- Macro-F1 is the most common headline metric in this literature: 12 of 21 verified papers report it.
- Cross-event results are aggregated in several incompatible ways (pooled, mean of per-event, mean of groups, mean of batches). Papers rarely state the aggregation and label-set handling together.
- At least 3 cross-event papers average per-held-out-event macro-F1, and 2 more use closely analogous averages (per held-out group; within-event per event).
- At least 6 papers set a random-split or in-distribution score against a cross-event score computed on different test instances.
- Released code computing macro-F1 with the default (data-derived) label set exists and produces reported numbers in 2 of the 10 code-released papers.
- Real per-event splits in HumAID, and held-out type test sets in CrisisTS, lack classes.

Not supported:
- That any published paper combines all ingredients, or that a specific published gap is inflated by the zero-term mechanism.
- Any prevalence estimate beyond these 22 papers. This is a purposive sample, not a systematic review.

## 6. UNVERIFIED items and limitations

- **Livers 2025:** full text not accessed.
- **Alam 2018:** whether transfer rows use the target's 30% test split.
- **Sánchez 2023:** whether the paper's "F1-score" is binary or macro.
- **Imran 2024:** averaging of the Table 3 and Figure 9 F1.
- **Wiegmann 2020 and Yu 2019:** averaging of F1.
- **McDaniel 2024:** identity of the benchmark test instances.
- **Seeberger 2022:** per-event scoring code not found; the Figure 4 metric is not restated in the caption.
- **Seeberger 2025 and HumAID:** code not found or not checked.
- **Medina Maza 2020:** repository not linked from the paper, so it is a probable, unconfirmed release.
- **TREC-IS:** the 2021-B notebook is a per-run rendering in TREC 2021 appendices, taken as representative of the official v3.0 script. The 2018 ISCRAM overview (McCreadie et al. 2019) could not be re-opened because eprints.gla.ac.uk timed out.
- **DeCrisisMB:** per-batch class coverage was not computed.
- **Not audited, known candidates:** Khare, Burel & Alani 2018 (ESWC) and Burel et al. 2017 (cross-crisis semantic classification); Li et al. 2018 (JCCM domain adaptation); Graf et al. 2018; Ghafarian & Yazdi 2020. Full texts were not opened, so they are not counted.
