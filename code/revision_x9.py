"""Exploratory X9 analyses of existing whole-event LOEO predictions (registered in research/preregistration.md at
2026-10-01T21:45Z, before any of them was computed). No model is trained.

(a) LOEO aggregation A under conventions P, D and O for every arm, with the X7 crossed bootstrap;
(b) A = W + N under P: W = sum_e w_e F_e - mean_e F_e with w_e = n_e / sum n (event-size weighting) and
    N = F_pool - sum_e w_e F_e (non-additivity of macro-F1 across events), same bootstrap draws;
(c) A under P after removing test messages in event-class cells with fewer than 15 messages;
(d) Spearman correlation between a class's share of an event and its per-class F1, per seed, two-epoch arms;
(e) per-event n_e, present classes and seed-mean F_e of the two-epoch arms;
(f) per-seed A of every arm;
(g) every per-event and pooled macro-F1 of the two-epoch arms recomputed with scikit-learn and with
    code/companion_eval.py from the prediction files.

    python code/revision_x9.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
import companion_eval as ce  # noqa: E402
from revision_x7 import boot_ci, load_arrays, loeo_aggregation  # noqa: E402

OUT = "experiments/derived/revision_x9.json"
HAVE_ALL_RUNS = bool(glob.glob(os.path.join("experiments", "runs", "confirm", "*", "*", "seed*", "chrono__*")))
MIN_SUPPORT = 15
CONVS = ("P", "D", "O")
DISTIL = "distilbert-base-uncased"
RUNS = os.path.join("experiments", "runs")


def specs():
    for corpus in av.CORPORA:
        for model in (DISTIL, "roberta-base"):
            yield "2 epochs", corpus, model, os.path.join(RUNS, "confirm", corpus, model), "loeo", None
        for model in sorted(os.listdir(os.path.join(RUNS, "confirm_ep4", corpus))):
            yield "4 epochs", corpus, model, os.path.join(RUNS, "confirm_ep4", corpus, model), "loeo", None
        yield "TF-IDF", corpus, "tfidf-lr", os.path.join(RUNS, "x3_tfidf", corpus, "tfidf-lr"), "loeo", None
        yield "uncapped", corpus, DISTIL, os.path.join(RUNS, "confirm", corpus, DISTIL), "loeo_full", ["42"]


def load_meta(corpus):
    return pd.read_csv(av.CORPORA[corpus][0], sep="\t", dtype=str, keep_default_na=False)


def complete_seeds(model_dir, events, prefix, only=None):
    """Seeds with predictions for every held-out event; others are listed with the first missing event."""
    names = sorted((os.path.basename(p)[4:] for p in glob.glob(os.path.join(model_dir, "seed*"))), key=int)
    ok, skipped = [], []
    for s in names if only is None else only:
        sd = os.path.join(model_dir, f"seed{s}")
        miss = [e for e in events if not os.path.exists(os.path.join(sd, f"{prefix}__{e}", "preds.tsv.gz"))]
        if miss:
            skipped.append({"seed": s, "reason": f"missing {prefix}:{miss[0]} in {sd}"})
        else:
            ok.append(s)
    return ok, skipped


def frames(model_dir, seeds, events, prefix):
    out = {}
    for s in seeds:
        for e in events:
            df = av.load(os.path.join(model_dir, f"seed{s}"), f"{prefix}:{e}")
            if df is None:
                raise SystemExit(f"missing {prefix}:{e} for seed {s} in {model_dir}")
            out[s, e] = df
    return out


def count_array(fr, seeds, events, classes, keep=None):
    c = np.zeros((len(events), len(seeds), len(classes), 3))
    for j, s in enumerate(seeds):
        for i, e in enumerate(events):
            df = fr[s, e] if keep is None else fr[s, e][fr[s, e]["tweet_id"].isin(keep)]
            c[i, j] = av.counts(df, classes)
    return c


def split_wn(c, ev, sd):
    x = c[np.ix_(ev, sd)]
    f = av.f1_from_counts(x, "P")
    n = x[..., [0, 2]].sum((-1, -2))
    fw = (n / n.sum(0, keepdims=True) * f).sum(0)
    pool = av.f1_from_counts(x.sum(0), "P")
    return float(np.mean(fw - f.mean(0))), float(np.mean(pool - fw))


def block(fn, n_e, n_s):
    ev, sd = np.arange(n_e), np.arange(n_s)
    return {"mean": fn(ev, sd), "ci95": boot_ci(fn, n_e, n_s)}


def spearman(c, events):
    rows = []
    for j in range(c.shape[1]):
        x = c[:, j]
        sup = x[..., 0] + x[..., 2]
        share = sup / sup.sum(1, keepdims=True)
        f = ce.per_class_f1(x)
        m = sup > 0
        r = spearmanr(share[m], f[m])
        rows.append((float(r.statistic), float(r.pvalue), int(m.sum())))
    rho = [r[0] for r in rows]
    return {"rho_mean": float(np.mean(rho)), "rho_min": float(min(rho)), "rho_max": float(max(rho)),
            "max_p": float(max(r[1] for r in rows)), "n_cells": rows[0][2]}


def sklearn_check(fr, seeds, events, classes, c, meta):
    d_sk, d_ce, n = 0.0, 0.0, 0
    for j, s in enumerate(seeds):
        pooled = pd.concat([fr[s, e] for e in events], ignore_index=True)
        for conv in CONVS:
            res, _ = ce.score(pooled[list(ce.REQUIRED)], meta, conv)
            ours_e = av.f1_from_counts(c[:, j], conv)
            ours_pool = float(av.f1_from_counts(c[:, j].sum(0), conv))
            d_ce = max(d_ce, abs(res["pooled_macro_f1"] - ours_pool),
                       max(abs(r["macro_f1"] - ours_e[i]) for i, r in enumerate(res["events"])))
            for i, e in enumerate(events + [None]):
                df = pooled if e is None else fr[s, e]
                y, p = df["y_true"], df["y_pred"]
                labels = {"P": sorted(set(y)), "D": None, "O": classes}[conv]
                sk = f1_score(y, p, labels=labels, average="macro", zero_division=0)
                d_sk = max(d_sk, abs(sk - (ours_pool if e is None else ours_e[i])))
                n += 1
    return d_sk, d_ce, n


def main():
    out = {"arms": [], "spearman": [], "per_event": {}, "cells_below_min_support": {}}
    check = {"max_abs_diff_sklearn": 0.0, "max_abs_diff_companion": 0.0, "n_scores": 0, "n_runs": 0}
    for corpus in av.CORPORA:
        meta = load_meta(corpus)
        cell = meta.groupby(["event", "label"]).size()
        small = cell[cell < MIN_SUPPORT]
        keep = set(meta.loc[~meta.set_index(["event", "label"]).index.isin(small.index), "tweet_id"])
        out["cells_below_min_support"][corpus] = {"cells": int(len(small)), "messages": int(small.sum()),
                                                  "events": int(small.index.get_level_values(0).nunique()),
                                                  "cells_total": int(len(cell)), "messages_total": int(len(meta))}
        out["_keep_" + corpus] = keep
    for arm, corpus, model, model_dir, prefix, seeds in specs():
        classes = sorted(load_meta(corpus)["label"].unique())
        events = sorted(json.load(open(av.CORPORA[corpus][1]))["events"])
        s_reg, skipped = complete_seeds(model_dir, events, prefix, seeds)
        fr = frames(model_dir, s_reg, events, prefix)
        c = count_array(fr, s_reg, events, classes)
        if seeds is None and HAVE_ALL_RUNS:
            A, _, s_full, _, _, _ = load_arrays(corpus, model_dir)
            assert s_full == s_reg and np.allclose(c, A["c"]), f"count mismatch {arm} {corpus} {model}"
        cf = count_array(fr, s_reg, events, classes, keep=out["_keep_" + corpus])
        E, S = len(events), len(s_reg)
        rec = {"arm": arm, "corpus": corpus, "model": model, "seeds": s_reg, "incomplete_seeds": skipped,
               "A": {conv: block(lambda e, s, cv=conv: loeo_aggregation({"c": c}, e, s, cv), E, S) for conv in CONVS},
               "W": block(lambda e, s: split_wn(c, e, s)[0], E, S),
               "N": block(lambda e, s: split_wn(c, e, s)[1], E, S),
               "A_min_support": block(lambda e, s: loeo_aggregation({"c": cf}, e, s, "P"), E, S),
               "per_seed_A": {s: loeo_aggregation({"c": c}, np.arange(E), np.array([j]), "P")
                              for j, s in enumerate(s_reg)}}
        out["arms"].append(rec)
        if arm == "2 epochs":
            out["spearman"].append({"corpus": corpus, "model": model, **spearman(c, events)})
            f = av.f1_from_counts(c, "P").mean(1)
            n = c[:, 0, :, [0, 2]].sum((0, 2))
            npres = ((c[:, 0, :, 0] + c[:, 0, :, 2]) > 0).sum(1)
            tab = out["per_event"].setdefault(corpus, {e: {"n": int(n[i]), "n_present": int(npres[i])}
                                                       for i, e in enumerate(events)})
            for i, e in enumerate(events):
                tab[e][model] = float(f[i])
            sup, fc = c[:, 0, :, 0] + c[:, 0, :, 2], ce.per_class_f1(c).mean(1)
            out.setdefault("cells", {}).setdefault(corpus, {})[model] = [
                {"event": e, "class": k, "support": int(sup[i, j]), "f1": float(fc[i, j])}
                for i, e in enumerate(events) for j, k in enumerate(classes) if sup[i, j] > 0]
            d_sk, d_ce, k = sklearn_check(fr, s_reg, events, classes, c, load_meta(corpus))
            check["max_abs_diff_sklearn"] = max(check["max_abs_diff_sklearn"], d_sk)
            check["max_abs_diff_companion"] = max(check["max_abs_diff_companion"], d_ce)
            check["n_scores"] += k
            check["n_runs"] += S * E
        print(arm, corpus, model, {cv: round(rec["A"][cv]["mean"], 4) for cv in CONVS},
              "W", round(rec["W"]["mean"], 4), "N", round(rec["N"]["mean"], 4),
              "A15", round(rec["A_min_support"]["mean"], 4))
    for corpus in av.CORPORA:
        del out["_keep_" + corpus]
    out["implementation_check"] = check
    json.dump(out, open(OUT, "w"), indent=1)
    print("check", check)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
