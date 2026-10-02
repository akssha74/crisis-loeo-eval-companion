"""Exploratory X11 analyses of existing whole-event LOEO predictions (registered in research/preregistration.md and
pushed to the public companion repository, commit 6eb95e3, before any of them was computed). No model is trained.

(a) closed forms of A^D and A^O in terms of A^P, checked per arm and seed; k_e; A^O of a perfect classifier;
    A under P, D and O for every system; scikit-learn labels=O with zero_division=nan against convention D;
(b) A^P = T1 + T2 + T3 + T4 (false positives in events lacking the class, non-additivity, support weighting,
    class composition), and the second ordering T2' + T3';
(c) agreement of pooled and per-event rankings of the systems, per corpus and convention;
(d) A^P after removing event-class cells below t messages, t in THRESHOLDS;
(e) the event-class cells below 15 messages;
(f) within-class Spearman correlation of cell support and per-class F1;
(g) event jackknife and bootstrap quantiles of A^P;
(h) cross-event duplicates by normalised text, and A^P without them.

    python code/revision_x11.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
import companion_eval as ce  # noqa: E402
from revision_x7 import B, RNG_SEED  # noqa: E402
from revision_x9 import complete_seeds, count_array, frames, load_meta, specs  # noqa: E402

OUT = "experiments/derived/revision_x11.json"
ZS = os.path.join("experiments", "runs", "x10_zeroshot")
THRESHOLDS = (5, 10, 15, 20, 30)
CONVS = ("P", "D", "O")
TERMS = ("T1", "T2", "T3", "T4", "T2alt", "T3alt")


def dup_path(corpus):
    return os.path.join("experiments", "meta", f"cross_event_duplicates_{corpus}.tsv")


def systems():
    for arm, corpus, model, model_dir, prefix, seeds in specs():
        yield arm, corpus, model, model_dir, prefix, seeds
    for corpus in av.CORPORA:
        yield "zero-shot", corpus, "bart-large-mnli", os.path.join(ZS, corpus, "bart-large-mnli"), None, None


def load_system(corpus, model_dir, prefix, seeds, events):
    if prefix is None:
        df = pd.read_csv(os.path.join(model_dir, "preds.tsv.gz"), sep="\t", dtype=str, keep_default_na=False)
        return ["0"], {("0", e): g for e, g in df.groupby("event")}
    s_reg, _ = complete_seeds(model_dir, events, prefix, seeds)
    return s_reg, frames(model_dir, s_reg, events, prefix)


def a_conv(c, conv):
    """A per seed for counts (E, S, C, 3), averaged over seeds."""
    return float(np.mean(av.f1_from_counts(c.sum(0), conv) - av.f1_from_counts(c, conv).mean(0)))


def terms(c):
    """Seed-mean T1..T4, T2alt, T3alt (code/companion_eval.split_terms) for counts (E, S, C, 3)."""
    return np.mean([[ce.split_terms(c[:, s])[k] for k in TERMS] for s in range(c.shape[1])], axis=0)


def boot(fn, n_e, n_s):
    """Crossed bootstrap with the X7 draws; fn returns a vector; returns the draws (B, k)."""
    rng = np.random.default_rng(RNG_SEED)
    return np.array(av.crossed_boot(lambda e, s: np.atleast_1d(fn(e, s)), n_e, n_s, rng, B), float)


def ci(draws):
    return [[float(np.percentile(draws[:, k], 2.5)), float(np.percentile(draws[:, k], 97.5))]
            for k in range(draws.shape[1])]


def closed_forms(c):
    """Deviation of the closed forms, the shifts, and k_e summaries."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    pres, n_c = (tp + fn) > 0, c.shape[2]
    k = ((~pres) & (fp > 0)).sum(-1)
    n_pres = pres.sum(-1)
    fp_e, fd_e, fo_e = (av.f1_from_counts(c, cv) for cv in CONVS)
    dev = max(np.abs(fo_e - n_pres / n_c * fp_e).max(), np.abs(fd_e - n_pres / (n_pres + k) * fp_e).max())
    pool = [av.f1_from_counts(c.sum(0), cv) for cv in CONVS]
    pooled_all = bool(((tp + fn).sum(0) > 0).all())
    shift_o = float(np.mean(((1 - n_pres / n_c) * fp_e).mean(0)))
    shift_d = float(np.mean((k / (n_pres + k) * fp_e).mean(0)))
    a = {cv: a_conv(c, cv) for cv in CONVS}
    dev_a = max(abs(a["O"] - a["P"] - shift_o), abs(a["D"] - a["P"] - shift_d))
    return {"max_abs_dev_event": float(dev), "max_abs_dev_A": float(dev_a), "pooled_has_every_class": pooled_all,
            "pooled_equal_across_conventions": bool(np.allclose(pool[0], pool[1]) and np.allclose(pool[0], pool[2])),
            "shift_O": shift_o, "shift_D": shift_d, "k_mean": float(k.mean()), "k_max": int(k.max()),
            "share_events_k_pos": float((k > 0).mean())}


def sklearn_nan_check(fr, seeds, events, classes):
    dev, n = 0.0, 0
    for s in seeds:
        for e in events:
            y, p = fr[s, e]["y_true"], fr[s, e]["y_pred"]
            a = f1_score(y, p, labels=classes, average="macro", zero_division=np.nan)
            d = f1_score(y, p, average="macro", zero_division=0)
            dev, n = max(dev, abs(a - d)), n + 1
    return {"max_abs_diff": float(dev), "n_events_scored": n}


def keep_ids(meta, t):
    cell = meta.groupby(["event", "label"]).size()
    small = cell[cell < t]
    drop = meta.set_index(["event", "label"]).index.isin(small.index)
    return set(meta.loc[~drop, "tweet_id"]), int(len(small)), int(small.sum())


def duplicates(corpus, meta):
    src = os.path.join("data", f"corpus_{corpus}.tsv")
    if os.path.exists(src):
        df = pd.read_csv(src, sep="\t", dtype={"tweet_id": str}, keep_default_na=False,
                         usecols=["tweet_id", "event", "text_norm"])
        n_ev = df.groupby("text_norm")["event"].nunique()
        flagged = df.loc[df["text_norm"].map(n_ev) > 1, ["tweet_id", "event"]].sort_values(["event", "tweet_id"])
        flagged.to_csv(dup_path(corpus), sep="\t", index=False)
    flagged = pd.read_csv(dup_path(corpus), sep="\t", dtype=str, keep_default_na=False)
    assert flagged["tweet_id"].isin(meta["tweet_id"]).all()
    return set(flagged["tweet_id"]), flagged.groupby("event").size().to_dict()


def jackknife(c, events, a_full):
    vals = np.array([a_conv(c[np.delete(np.arange(len(events)), i)], "P") for i in range(len(events))])
    i = int(np.argmax(np.abs(vals - a_full)))
    return {"min": float(vals.min()), "max": float(vals.max()), "most_influential_event": events[i],
            "A_without_it": float(vals[i]), "values": {e: float(v) for e, v in zip(events, vals)}}


def main():
    out = {"systems": [], "ranking": {}, "small_cells": {}, "within_class": [], "duplicates": {},
           "thresholds": {}, "perfect_classifier_A_O": {}}
    metas = {k: load_meta(k) for k in av.CORPORA}
    keeps = {k: {t: keep_ids(metas[k], t) for t in THRESHOLDS} for k in av.CORPORA}
    dups = {k: duplicates(k, metas[k]) for k in av.CORPORA}
    for corpus, meta in metas.items():
        out["thresholds"][corpus] = {str(t): {"cells": v[1], "messages": v[2]} for t, v in keeps[corpus].items()}
        n_pres = meta.groupby("event")["label"].nunique()
        out["perfect_classifier_A_O"][corpus] = float((1 - n_pres / meta["label"].nunique()).mean())
        out["duplicates"][corpus] = {"messages": len(dups[corpus][0]), "events_with_any": len(dups[corpus][1]),
                                     "messages_total": int(len(meta))}
    cells_kept = {}
    for arm, corpus, model, model_dir, prefix, seeds in systems():
        meta = metas[corpus]
        classes = sorted(meta["label"].unique())
        events = sorted(json.load(open(av.CORPORA[corpus][1]))["events"])
        s_reg, fr = load_system(corpus, model_dir, prefix, seeds, events)
        c = count_array(fr, s_reg, events, classes)
        E, S = len(events), len(s_reg)
        rec = {"arm": arm, "corpus": corpus, "model": model, "seeds": s_reg,
               "pooled": {cv: float(av.f1_from_counts(c.sum(0), cv).mean()) for cv in CONVS},
               "per_event": {cv: float(av.f1_from_counts(c, cv).mean()) for cv in CONVS}}
        d = boot(lambda e, s: [a_conv(c[np.ix_(e, s)], cv) for cv in CONVS], E, S)
        rec["A"] = {cv: {"mean": a_conv(c, cv), "ci95": ci(d)[j]} for j, cv in enumerate(CONVS)}
        q = np.percentile(d[:, 0], [2.5, 25, 50, 75, 97.5])
        rec["A_P_boot_quantiles"] = [float(x) for x in q]
        rng = np.random.default_rng(RNG_SEED)
        pres_e = ((c[:, 0, :, 0] + c[:, 0, :, 2]) > 0)
        lost = 0
        for _ in range(B):
            ev = rng.integers(0, E, E)
            rng.integers(0, S, S)
            lost += int(not pres_e[ev].any(0).all())
        rec["share_draws_missing_a_class"] = lost / B
        rec["closed_forms"] = closed_forms(c)
        t_point = terms(c)
        t_ci = ci(boot(lambda e, s: terms(c[np.ix_(e, s)]), E, S))
        rec["terms"] = {k: {"mean": float(t_point[j]), "ci95": t_ci[j]} for j, k in enumerate(TERMS)}
        assert abs(t_point[:4].sum() - rec["A"]["P"]["mean"]) < 1e-12
        assert abs(t_point[[0, 4, 5, 3]].sum() - rec["A"]["P"]["mean"]) < 1e-12
        rec["thresholds"] = {}
        for t in THRESHOLDS:
            cf = count_array(fr, s_reg, events, classes, keep=keeps[corpus][t][0])
            ev_ok = np.where(cf[:, 0, :, [0, 2]].sum((0, 2)) > 0)[0]
            cf = cf[ev_ok]
            dt = boot(lambda e, s: a_conv(cf[np.ix_(e, s)], "P"), len(ev_ok), S)
            rec["thresholds"][str(t)] = {"mean": a_conv(cf, "P"), "ci95": ci(dt)[0], "events": int(len(ev_ok))}
        cd = count_array(fr, s_reg, events, classes, keep=set(meta["tweet_id"]) - dups[corpus][0])
        dd = boot(lambda e, s: a_conv(cd[np.ix_(e, s)], "P"), E, S)
        rec["A_P_without_cross_event_duplicates"] = {"mean": a_conv(cd, "P"), "ci95": ci(dd)[0]}
        rec["jackknife"] = jackknife(c, events, rec["A"]["P"]["mean"])
        if arm == "2 epochs":
            rec["sklearn_zero_division_nan"] = sklearn_nan_check(fr, s_reg, events, classes)
            sup = c[:, 0, :, 0] + c[:, 0, :, 2]
            f = ce.per_class_f1(c).mean(1)
            per_class = {}
            for j, k in enumerate(classes):
                m = sup[:, j] > 0
                rho = float(spearmanr(sup[m, j], f[m, j]).statistic) if np.ptp(f[m, j]) > 0 else None
                per_class[k] = {"rho": rho, "n_cells": int(m.sum())}
            defined = [v["rho"] for v in per_class.values() if v["rho"] is not None]
            out["within_class"].append({"corpus": corpus, "model": model, "per_class": per_class,
                                        "median_rho": float(np.median(defined)), "n_classes_defined": len(defined)})
            cells_kept[corpus, model] = (sup, f, events, classes)
        out["systems"].append(rec)
        print(arm, corpus, model, {cv: round(rec["A"][cv]["mean"], 4) for cv in CONVS},
              {k: round(v["mean"], 4) for k, v in rec["terms"].items()}, flush=True)
    for corpus in av.CORPORA:
        rows = [r for r in out["systems"] if r["corpus"] == corpus]
        out["ranking"][corpus] = {}
        for cv in CONVS:
            pool = np.array([r["pooled"][cv] for r in rows])
            per = np.array([r["per_event"][cv] for r in rows])
            tau = kendalltau(pool, per).statistic
            flips = [[f"{rows[i]['arm']}|{rows[i]['model']}", f"{rows[j]['arm']}|{rows[j]['model']}"]
                     for i in range(len(rows)) for j in range(i + 1, len(rows))
                     if np.sign(pool[i] - pool[j]) != np.sign(per[i] - per[j])]
            out["ranking"][corpus][cv] = {"kendall_tau": float(tau), "pairs": len(rows) * (len(rows) - 1) // 2,
                                          "discordant": flips}
        small = []
        for (cp, model), (sup, f, events, classes) in cells_kept.items():
            if cp != corpus:
                continue
            for i, e in enumerate(events):
                for j, k in enumerate(classes):
                    if 0 < sup[i, j] < 15:
                        small.append({"event": e, "class": k, "support": int(sup[i, j]), "model": model,
                                      "f1": float(f[i, j])})
        out["small_cells"][corpus] = small
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
