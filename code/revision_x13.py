"""Exploratory X13 analyses of the whole-event LOEO predictions of X11 (registered in research/preregistration.md
and pushed to the public companion repository, commit 05e8cd4, before any of them was computed). No model is
trained.

(a) composition-first ordering of T3 + T4 (T4c, then T3c; code/companion_eval.split_terms);
(b) fixed-rate null: A^P and its terms when every class keeps its pooled recall and false-positive rate in every
    event, with all observed supports; the same after removing cells below 15 messages; observed minus null;
(c) mixed summaries: pairs whose order reverses when the pooled score of one system is set beside the per-event
    mean of the other;
(d) the evaluator option --min-cell 15 against the X11(d) A^P at t = 15.

    python code/revision_x13.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
import companion_eval as ce  # noqa: E402
from revision_x7 import B, RNG_SEED  # noqa: E402
from revision_x9 import count_array, load_meta  # noqa: E402
from revision_x11 import a_conv, boot, ci, keep_ids, load_system, systems  # noqa: E402

OUT = "experiments/derived/revision_x13.json"
X11 = "experiments/derived/revision_x11.json"
SIM_SEED, REPS, REPS_BOOT = 20261002, 200, 20
NULL_TERMS = ("T1", "T2", "T3", "T4")


def name(arm, model):
    return f"{arm}|{model}"


def comp_terms(c):
    """Seed-mean T3c and T4c for counts (E, S, C, 3)."""
    return np.mean([[ce.split_terms(c[:, s])[k] for k in ("T3c", "T4c")] for s in range(c.shape[1])], axis=0)


def a_p(tp, fp, fn):
    """A^P for counts (..., E, C)."""
    den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    pres = (tp + fn) > 0
    per = ((f * pres).sum(-1) / np.maximum(pres.sum(-1), 1)).mean(-1)
    tq, pq, nq = tp.sum(-2), fp.sum(-2), fn.sum(-2)
    dq = 2 * tq + pq + nq
    fq = np.where(dq > 0, 2 * tq / np.where(dq > 0, dq, 1), 0.0)
    pq_mask = (tq + nq) > 0
    return (fq * pq_mask).sum(-1) / np.maximum(pq_mask.sum(-1), 1) - per


def simulate(cs, rng, reps):
    """Replicates (reps, E, C) of TP, FP, FN for one seed's counts (E, C, 3) under the fixed-rate null."""
    tp, fp, fn = cs[..., 0], cs[..., 1], cs[..., 2]
    sup = tp + fn
    other = sup.sum(1, keepdims=True) - sup
    s_c, n = sup.sum(0), sup.sum()
    r = np.where(s_c > 0, tp.sum(0) / np.maximum(s_c, 1), 0.0)
    f = fp.sum(0) / np.maximum(n - s_c, 1)
    t = rng.binomial(np.broadcast_to(sup.astype(np.int64), (reps,) + sup.shape), r)
    p = rng.binomial(np.broadcast_to(other.astype(np.int64), (reps,) + sup.shape), f)
    return t.astype(float), p.astype(float), sup - t


def null(c, rng, reps, with_terms=False):
    """Replicate values of seed-mean A^P (and T1-T4) under the fixed-rate null for counts (E, S, C, 3)."""
    a, terms = np.zeros(reps), np.zeros((reps, len(NULL_TERMS)))
    for s in range(c.shape[1]):
        t, p, n = simulate(c[:, s], rng, reps)
        a += a_p(t, p, n) / c.shape[1]
        if with_terms:
            for r in range(reps):
                x = ce.split_terms(np.stack([t[r], p[r], n[r]], -1))
                terms[r] += np.array([x[k] for k in NULL_TERMS]) / c.shape[1]
    return a, terms


def summary(v):
    return {"mean": float(v.mean()), "q025": float(np.percentile(v, 2.5)), "q975": float(np.percentile(v, 97.5))}


def evaluator_check(fr, s_reg, events, meta, x11_value):
    vals = []
    for s in s_reg:
        pred = pd.concat([fr[s, e][["tweet_id", "event", "y_true", "y_pred"]] for e in events], ignore_index=True)
        res, _ = ce.score(pred, meta, "P", min_cell=15)
        vals.append(res["A_P"])
    return {"evaluator_A_P": float(np.mean(vals)), "x11_A_P": x11_value,
            "abs_diff": float(abs(np.mean(vals) - x11_value)), "removed_messages": res["removed_messages"]}


def mixed(rows):
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            same = np.sign(a["pooled"] - b["pooled"])
            if same != np.sign(a["per_event"] - b["per_event"]):
                continue
            for x, y in ((a, b), (b, a)):
                gap = x["pooled"] - y["per_event"]
                out.append({"pooled_of": x["system"], "per_event_of": y["system"], "gap": float(gap),
                            "gap_pooled": float(x["pooled"] - y["pooled"]),
                            "gap_per_event": float(x["per_event"] - y["per_event"]),
                            "reversed": bool(np.sign(gap) != np.sign(x["pooled"] - y["pooled"]))})
    return out


def main():
    x11 = {(r["corpus"], name(r["arm"], r["model"])): r for r in json.load(open(X11))["systems"]}
    out = {"systems": [], "mixed": {}}
    metas = {k: load_meta(k) for k in av.CORPORA}
    keep15 = {k: keep_ids(metas[k], 15)[0] for k in av.CORPORA}
    rng = np.random.default_rng(SIM_SEED)
    for arm, corpus, model, model_dir, prefix, seeds in systems():
        meta = metas[corpus]
        classes = sorted(meta["label"].unique())
        events = sorted(json.load(open(av.CORPORA[corpus][1]))["events"])
        s_reg, fr = load_system(corpus, model_dir, prefix, seeds, events)
        c = count_array(fr, s_reg, events, classes)
        E, S = len(events), len(s_reg)
        sid = name(arm, model)
        rec = {"system": sid, "corpus": corpus, "seeds": s_reg, "A_P": a_conv(c, "P"),
               "pooled": float(av.f1_from_counts(c.sum(0), "P").mean()),
               "per_event": float(av.f1_from_counts(c, "P").mean())}
        point = comp_terms(c)
        draws = ci(boot(lambda e, s: comp_terms(c[np.ix_(e, s)]), E, S))
        rec["composition_first"] = {k: {"mean": float(point[j]), "ci95": draws[j]} for j, k in
                                    enumerate(("T3c", "T4c"))}
        x11_terms = x11[corpus, sid]["terms"]
        assert abs(point.sum() - x11_terms["T3"]["mean"] - x11_terms["T4"]["mean"]) < 1e-12
        a_null, t_null = null(c, rng, REPS, with_terms=True)
        rec["null"] = {"A_P": summary(a_null),
                       **{k: summary(t_null[:, j]) for j, k in enumerate(NULL_TERMS)}}
        c15 = count_array(fr, s_reg, events, classes, keep=keep15[corpus])
        c15 = c15[np.where(c15[:, 0, :, [0, 2]].sum((0, 2)) > 0)[0]]
        rec["null_t15"] = {"A_P_observed": a_conv(c15, "P"), "A_P": summary(null(c15, rng, REPS)[0])}
        brng = np.random.default_rng(SIM_SEED + 1)
        d = boot(lambda e, s: a_conv(c[np.ix_(e, s)], "P") - null(c[np.ix_(e, s)], brng, REPS_BOOT)[0].mean(), E, S)
        rec["observed_minus_null"] = {"mean": rec["A_P"] - rec["null"]["A_P"]["mean"], "ci95": ci(d)[0]}
        rec["evaluator_min_cell_15"] = evaluator_check(fr, s_reg, events, meta,
                                                       x11[corpus, sid]["thresholds"]["15"]["mean"])
        out["systems"].append(rec)
        print(corpus, sid, "A_P", round(rec["A_P"], 4), "null", round(rec["null"]["A_P"]["mean"], 4),
              "obs-null", [round(x, 4) for x in rec["observed_minus_null"]["ci95"]],
              "T3c/T4c", [round(x, 4) for x in point], "eval-diff", rec["evaluator_min_cell_15"]["abs_diff"],
              flush=True)
    for corpus in av.CORPORA:
        rows = [r for r in out["systems"] if r["corpus"] == corpus]
        m = mixed(rows)
        out["mixed"][corpus] = {"comparisons": m, "n": len(m), "n_reversed": sum(x["reversed"] for x in m)}
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, {k: (v["n_reversed"], v["n"]) for k, v in out["mixed"].items()})


if __name__ == "__main__":
    main()
