"""Evaluation companion: score event-held-out predictions under pooled and per-event macro-F1.

Input is one or more prediction files (TSV, optionally gzipped) with the columns tweet_id, event, y_true and
y_pred; other columns are ignored. Rows are validated against the pinned metadata table (tweet_id, event, hazard,
label). Every represented event must be complete; a run that scores only some events must say so with --partial,
and its output is marked partial.

Label-list conventions (--convention, required):
  P  classes with at least one true message in the scored set (event, or pooled set);
  D  classes in the true labels or the predictions (scikit-learn's default label set);
  O  every class of the corpus ontology; a class absent from both labels and predictions scores 0.
Per-class F1 is 2TP / (2TP + FP + FN), and 0 when the denominator is 0; macro-F1 is the mean of per-class F1.
A = pooled - per-event mean. Under P it is split as A^P = T1 + T2 + T3 + T4 (split_terms); under D or O,
A - A^P is a closed-form label-list shift. Every result carries a signature naming the label list, the
zero-denominator rule, the number of events, the metadata hash and the evaluator version.

    python code/companion_eval.py --corpus crisislext26 --system tfidf-lr --seed 42 --convention P --out out
    python code/companion_eval.py --meta experiments/meta/corpus_crisislext26_meta.tsv \
        --preds 'my_run/*.tsv.gz' --convention P --out out
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

VERSION = "1.1.0"
REQUIRED = ("tweet_id", "event", "y_true", "y_pred")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEMS = os.path.join("experiments", "systems.tsv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_predictions(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{p}: missing columns {missing}")
        frames.append(df[list(REQUIRED)])
    return pd.concat(frames, ignore_index=True)


def validate(pred, meta, partial):
    """Return the represented events; raise ValueError on any inconsistency with the metadata."""
    problems = []
    dup = pred["tweet_id"].duplicated()
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate tweet_id rows")
    m = meta.set_index("tweet_id")
    unknown = ~pred["tweet_id"].isin(m.index)
    if unknown.any():
        problems.append(f"{int(unknown.sum())} tweet_id not in the metadata")
    known = pred[~unknown]
    wrong_event = known["event"].to_numpy() != m.loc[known["tweet_id"], "event"].to_numpy()
    if wrong_event.any():
        problems.append(f"{int(wrong_event.sum())} rows whose event disagrees with the metadata")
    wrong_label = known["y_true"].to_numpy() != m.loc[known["tweet_id"], "label"].to_numpy()
    if wrong_label.any():
        problems.append(f"{int(wrong_label.sum())} rows whose y_true disagrees with the metadata")
    classes = set(meta["label"])
    stray = ~pred["y_pred"].isin(classes)
    if stray.any():
        problems.append(f"{int(stray.sum())} predictions outside the corpus ontology")
    events = sorted(set(known["event"]))
    size = meta.groupby("event").size()
    got = known.groupby("event").size()
    incomplete = [e for e in events if got[e] != size[e]]
    if incomplete:
        problems.append(f"incomplete events (whole-event scoring needs every message): {incomplete}")
    absent = sorted(set(size.index) - set(events))
    if absent and not partial:
        problems.append(f"events with no predictions: {absent} (use --partial to score a subset)")
    if problems:
        raise ValueError("; ".join(problems))
    return events, absent


def counts(df, classes):
    y, p = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    out = np.zeros((len(classes), 3), dtype=np.int64)
    for j, c in enumerate(classes):
        t, q = y == c, p == c
        out[j] = [np.sum(t & q), np.sum(~t & q), np.sum(t & ~q)]
    return out


def per_class_f1(cnt):
    tp, fp, fn = cnt[..., 0], cnt[..., 1], cnt[..., 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def label_mask(cnt, convention):
    tp, fp, fn = cnt[..., 0], cnt[..., 1], cnt[..., 2]
    if convention == "P":
        return (tp + fn) > 0
    if convention == "D":
        return ((tp + fn) > 0) | ((tp + fp) > 0)
    return np.ones(tp.shape, dtype=bool)


def macro_f1(cnt, convention):
    f, m = per_class_f1(cnt), label_mask(cnt, convention)
    return float((f * m).sum() / max(m.sum(), 1))


def split_terms(cnt):
    """Split A^P for event-class counts (E, C, 3) into four terms that sum to it exactly.

    With E_c the events containing class c and s_ec its support in e, averaging over the classes of the pool:
    T1 = F1 of counts summed over all events - F1 of counts summed over E_c (false positives where c is absent);
    T2 = F1 of counts summed over E_c - support-weighted mean of F_ec over E_c (non-additivity);
    T3 = support-weighted mean - equal-weighted mean of F_ec over E_c (support weighting);
    T4 = equal-weighted mean over E_c - per-event mean (class composition; 0 when every event has every class).
    T2alt and T3alt reverse the order of T2 and T3, pooling the counts of each event weighted by 1 / s_ec.
    """
    tp, fp, fn = (cnt[..., k].astype(float) for k in range(3))
    sup = tp + fn
    pres = sup > 0
    keep = pres.any(0)

    def f1(a, b, c):
        den = 2 * a + b + c
        return np.where(den > 0, 2 * a / np.where(den > 0, den, 1), 0.0)

    f_ec = f1(tp, fp, fn)
    w = np.where(pres, 1 / np.where(pres, sup, 1), 0.0)
    f_all = f1(tp.sum(0), fp.sum(0), fn.sum(0))
    f_in = f1((tp * pres).sum(0), (fp * pres).sum(0), (fn * pres).sum(0))
    f_eq = f1((tp * w).sum(0), (fp * w).sum(0), (fn * w).sum(0))
    swm = (sup * f_ec).sum(0) / np.maximum(sup.sum(0), 1)
    eqm = (f_ec * pres).sum(0) / np.maximum(pres.sum(0), 1)
    f_e = float(((f_ec * pres).sum(1) / np.maximum(pres.sum(1), 1)).mean())
    m = {k: float(v[keep].mean()) for k, v in
         {"all": f_all, "in": f_in, "eq": f_eq, "swm": swm, "eqm": eqm}.items()}
    return {"T1": m["all"] - m["in"], "T2": m["in"] - m["swm"], "T3": m["swm"] - m["eqm"], "T4": m["eqm"] - f_e,
            "T2alt": m["eq"] - m["eqm"], "T3alt": m["in"] - m["eq"]}


def score(pred, meta, convention, partial=False):
    events, absent = validate(pred, meta, partial)
    classes = sorted(meta["label"].unique())
    cnt = np.stack([counts(pred[pred["event"] == e], classes) for e in events])
    pooled_cnt = cnt.sum(0)
    f_e = np.array([macro_f1(c, convention) for c in cnt])
    n_e = cnt[:, :, [0, 2]].sum((1, 2))
    f_pool = macro_f1(pooled_cnt, convention)
    terms = split_terms(cnt)
    a_p = macro_f1(pooled_cnt, "P") - float(np.mean([macro_f1(c, "P") for c in cnt]))
    rows = []
    for i, e in enumerate(events):
        f = per_class_f1(cnt[i])
        m = label_mask(cnt[i], convention)
        for j, c in enumerate(classes):
            tp, fp, fn = (int(x) for x in cnt[i, j])
            rows.append({"event": e, "class": c, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
                         "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                         "f1": float(f[j]), "in_label_list": bool(m[j])})
    result = {"convention": convention, "classes": classes, "n_events": len(events), "partial": bool(absent),
              "events_absent": absent, "n_messages": int(n_e.sum()), "pooled_macro_f1": f_pool,
              "per_event_mean_macro_f1": float(f_e.mean()), "A": f_pool - float(f_e.mean()),
              "A_P": a_p, "split_of_A_P": terms, "label_list_shift": f_pool - float(f_e.mean()) - a_p,
              "events": [{"event": e, "n": int(n_e[i]), "n_present_classes": int(label_mask(cnt[i], "P").sum()),
                          "macro_f1": float(f_e[i])} for i, e in enumerate(events)]}
    return result, pd.DataFrame(rows)


def signature(result, meta_sha):
    events = f"{result['n_events']}{'+partial' if result['partial'] else ''}"
    return (f"loeo-macro-f1|labels:{result['convention']}|zero-division:0|events:{events}|meta:{meta_sha[:12]}"
            f"|v:{VERSION}")


def resolve_system(corpus, system, seed):
    """Metadata path and prediction pattern of a reference system listed in experiments/systems.tsv."""
    table = pd.read_csv(os.path.join(ROOT, SYSTEMS), sep="\t", dtype=str, keep_default_na=False)
    row = table[(table["corpus"] == corpus) & (table["system"] == system)]
    if row.empty:
        sys.exit(f"no system {system!r} for corpus {corpus!r}; listed: "
                 f"{', '.join(table.loc[table['corpus'] == corpus, 'system'])}")
    row = row.iloc[0]
    seeds = [s for s in row["seeds"].split(",") if s]
    if seeds and seed not in seeds:
        sys.exit(f"{system} on {corpus} needs --seed, one of {', '.join(seeds)}")
    return row["meta"], row["preds"].replace("{seed}", seed or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", help="pinned metadata TSV (tweet_id, event, hazard, label)")
    ap.add_argument("--preds", nargs="+", help="prediction files or quoted glob patterns")
    ap.add_argument("--corpus", help="with --system: corpus of a reference system (experiments/systems.tsv)")
    ap.add_argument("--system", help="score a reference system listed in experiments/systems.tsv")
    ap.add_argument("--seed", help="with --system: training seed")
    ap.add_argument("--convention", required=True, choices=("P", "D", "O"))
    ap.add_argument("--partial", action="store_true", help="allow events without predictions")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    if a.system:
        if a.meta or a.preds or not a.corpus:
            ap.error("--system needs --corpus and replaces --meta and --preds")
        a.meta, pattern = resolve_system(a.corpus, a.system, a.seed)
        a.preds = [pattern]
    elif not (a.meta and a.preds):
        ap.error("give --meta and --preds, or --corpus and --system")
    paths = sorted({p for pat in a.preds for p in (glob.glob(pat) or [pat])})
    meta = pd.read_csv(a.meta, sep="\t", dtype=str, keep_default_na=False)
    try:
        result, cells = score(read_predictions(paths), meta, a.convention, a.partial)
    except ValueError as err:
        sys.exit(f"validation failed: {err}")
    meta_sha = sha256(a.meta)
    config = {"meta": {"path": a.meta, "sha256": meta_sha}, "convention": a.convention,
              "preds": [{"path": p, "sha256": sha256(p)} for p in paths]}
    result["signature"] = signature(result, meta_sha)
    result["evaluator_version"] = VERSION
    result["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    result["config"] = config
    os.makedirs(a.out, exist_ok=True)
    json.dump(result, open(os.path.join(a.out, "scores.json"), "w"), indent=1)
    cells.to_csv(os.path.join(a.out, "event_class_counts.tsv"), sep="\t", index=False)
    print(f"events {result['n_events']}{' (partial)' if result['partial'] else ''}  messages {result['n_messages']}  "
          f"convention {a.convention}")
    print(f"pooled {result['pooled_macro_f1']:.4f}  per-event mean {result['per_event_mean_macro_f1']:.4f}  "
          f"A {result['A']:+.4f}")
    t = result["split_of_A_P"]
    shift = "" if a.convention == "P" else f"  label list {result['label_list_shift']:+.4f}"
    print(f"A_P {result['A_P']:+.4f} = T1 {t['T1']:+.4f}  T2 {t['T2']:+.4f}  T3 {t['T3']:+.4f}  "
          f"T4 {t['T4']:+.4f}{shift}")
    print(f"signature {result['signature']}")
    print(f"wrote {a.out}/scores.json, {a.out}/event_class_counts.tsv")


if __name__ == "__main__":
    main()
