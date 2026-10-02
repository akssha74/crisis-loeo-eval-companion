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
A = pooled - per-event mean = W + N, with W = size-weighted mean - per-event mean (event-size weighting) and
N = pooled - size-weighted mean.

    python code/companion_eval.py --meta experiments/meta/corpus_crisislext26_meta.tsv \
        --preds 'experiments/runs/confirm/crisislext26/roberta-base/seed42/loeo__*/preds.tsv.gz' \
        --convention P --out /tmp/companion_out
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

REQUIRED = ("tweet_id", "event", "y_true", "y_pred")


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


def score(pred, meta, convention, partial=False):
    events, absent = validate(pred, meta, partial)
    classes = sorted(meta["label"].unique())
    cnt = np.stack([counts(pred[pred["event"] == e], classes) for e in events])
    pooled_cnt = cnt.sum(0)
    f_e = np.array([macro_f1(c, convention) for c in cnt])
    n_e = cnt[:, :, [0, 2]].sum((1, 2))
    f_pool = macro_f1(pooled_cnt, convention)
    f_w = float((n_e / n_e.sum() * f_e).sum())
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
              "size_weighted_mean_macro_f1": f_w, "W": f_w - float(f_e.mean()), "N": f_pool - f_w,
              "events": [{"event": e, "n": int(n_e[i]), "n_present_classes": int(label_mask(cnt[i], "P").sum()),
                          "macro_f1": float(f_e[i])} for i, e in enumerate(events)]}
    return result, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", required=True, help="pinned metadata TSV (tweet_id, event, hazard, label)")
    ap.add_argument("--preds", required=True, nargs="+", help="prediction files or quoted glob patterns")
    ap.add_argument("--convention", required=True, choices=("P", "D", "O"))
    ap.add_argument("--partial", action="store_true", help="allow events without predictions")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    paths = sorted({p for pat in a.preds for p in (glob.glob(pat) or [pat])})
    meta = pd.read_csv(a.meta, sep="\t", dtype=str, keep_default_na=False)
    try:
        result, cells = score(read_predictions(paths), meta, a.convention, a.partial)
    except ValueError as err:
        sys.exit(f"validation failed: {err}")
    config = {"meta": {"path": a.meta, "sha256": sha256(a.meta)}, "convention": a.convention,
              "preds": [{"path": p, "sha256": sha256(p)} for p in paths]}
    result["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    result["config"] = config
    os.makedirs(a.out, exist_ok=True)
    json.dump(result, open(os.path.join(a.out, "scores.json"), "w"), indent=1)
    cells.to_csv(os.path.join(a.out, "event_class_counts.tsv"), sep="\t", index=False)
    print(f"events {result['n_events']}{' (partial)' if result['partial'] else ''}  messages {result['n_messages']}  "
          f"convention {a.convention}")
    print(f"pooled {result['pooled_macro_f1']:.4f}  per-event mean {result['per_event_mean_macro_f1']:.4f}  "
          f"A {result['A']:+.4f}  (W {result['W']:+.4f}  N {result['N']:+.4f})")
    print(f"config {result['config_hash'][:12]}  wrote {a.out}/scores.json, {a.out}/event_class_counts.tsv")


if __name__ == "__main__":
    main()
