"""Exploratory X8 analyses (registered in research/preregistration.md, commit 34ae3932d, before any of them was
computed or run).

(a) identical-seed repeats of the two-epoch DistilBERT HumAID runs behind the protected E2 (seed 42);
(b) CrisisLexT26 E2 against the deduplicated in-distribution model; H3 per corpus; whole-event LOEO aggregation
    and E2 for every arm; per-seed composition of the four-epoch HumAID arms.

    python code/revision_x8.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
from revision_tmlr import re_interval  # noqa: E402
from revision_x7 import boot_ci, cell, load_arrays, loeo_aggregation, tost_p  # noqa: E402

OUT = "experiments/derived/revision_x8.json"
SUMMARY = "experiments/derived/summary_v2_confirm.json"
MARGIN = 0.02
DISTIL = "distilbert-base-uncased"
REPEATS = ("x8_repeat_a", "x8_repeat_b")


def e2_block(A, evs):
    y = cell(A, evs, "a", "b")
    return {"mean": float(y.mean()), "t95": av.t_interval(y.mean(1), 0.95),
            "crossed95": re_interval(y, 0.95) if y.shape[1] > 1 else None}


def agg_block(A, evs, n_seeds):
    return {"mean": loeo_aggregation(A, evs, np.arange(n_seeds), "P"),
            "ci95": boot_ci(lambda e, s: loeo_aggregation(A, evs[e], s, "P"), len(evs), n_seeds)}


def repeats():
    """(a): protected per-event F1 and E2 of the original seed-42 runs and of each identical-seed repeat."""
    cf, mf = av.CORPORA["humaid19"]
    man = json.load(open(mf))
    classes = sorted(pd.read_csv(cf, sep="\t", keep_default_na=False, usecols=["label"])["label"].unique())
    prot = av.PROTECTED["humaid19"]
    R = {e: set(map(str, man["events"][e]["R"])) for e in prot}
    dirs = {"original": os.path.join("experiments", "runs", "confirm", "humaid19", DISTIL, "seed42")}
    dirs.update({r: os.path.join("experiments", "runs", r, "humaid19", DISTIL, "seed42") for r in REPEATS})
    reps = {}
    for lab, d in dirs.items():
        idf = av.load(d, "id_random")
        if idf is None or any(av.load(d, f"loeo:{e}") is None for e in prot):
            continue
        a, b, c = [], [], []
        for e in prot:
            l = av.load(d, f"loeo:{e}")
            ae = idf[idf["event"] == e]
            lr = l[l["tweet_id"].isin(R[e])].set_index("tweet_id").loc[ae["tweet_id"]].reset_index()
            a.append(av.counts(ae, classes)); b.append(av.counts(lr, classes)); c.append(av.counts(l, classes))
        fa, fb, fc = (av.f1_from_counts(np.array(x), "P") for x in (a, b, c))
        reps[lab] = {"train_ids_sha256": av.run_meta(d, "id_random").get("train_ids_sha256"),
                     "F_id_R": dict(zip(prot, map(float, fa))), "F_loeo_R": dict(zip(prot, map(float, fb))),
                     "F_loeo_whole": dict(zip(prot, map(float, fc))), "E2_events": dict(zip(prot, map(float, fa - fb))),
                     "E2": float((fa - fb).mean())}
    out = {"replicates": reps, "complete": len(reps) == 1 + len(REPEATS)}
    if len(reps) >= 2:
        labs = list(reps)
        same = len({reps[k]["train_ids_sha256"] for k in labs}) == 1
        E2s = np.array([reps[k]["E2"] for k in labs])
        ev = np.array([[reps[k]["E2_events"][e] for k in labs] for e in prot])
        fid = np.array([[reps[k]["F_id_R"][e] for k in labs] for e in prot])
        floeo = np.array([[reps[k]["F_loeo_whole"][e] for k in labs] for e in prot])
        A2, subsets, seeds, _, _, _ = load_arrays("humaid19", os.path.join("experiments", "runs", "confirm", "humaid19", DISTIL))
        pe = np.array(subsets["protected"])
        seedE2 = cell(A2, pe, "a", "b").mean(0)
        seed_ev = cell(A2, pe, "a", "b")
        seed_fc = av.f1_from_counts(A2["c"][pe], "P")
        out.update({"same_training_sample": bool(same), "labels": labs,
                    "sd_E2_repeats": float(E2s.std(ddof=1)), "range_E2_repeats": [float(E2s.min()), float(E2s.max())],
                    "sd_E2_seeds": float(seedE2.std(ddof=1)), "E2_seeds": dict(zip(seeds, map(float, seedE2))),
                    "mean_sd_event_E2_repeats": float(ev.std(1, ddof=1).mean()),
                    "mean_sd_event_E2_seeds": float(seed_ev.std(1, ddof=1).mean()),
                    "mean_sd_event_F_id_repeats": float(fid.std(1, ddof=1).mean()),
                    "mean_sd_event_F_loeo_whole_repeats": float(floeo.std(1, ddof=1).mean()),
                    "mean_sd_event_F_loeo_whole_seeds": float(seed_fc.std(1, ddof=1).mean()),
                    "max_abs_event_F_diff_repeats": float(np.max(np.abs(np.concatenate([fid, floeo], 0)
                                                                        - np.concatenate([fid, floeo], 0)[:, :1])))})
    return out


def dedup_e2():
    """(b1): CrisisLexT26 E2 against the deduplicated in-distribution model."""
    out = {}
    for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm", "crisislext26", "*"))):
        A, subsets, seeds, skipped, _, _ = load_arrays("crisislext26", model_dir)
        assert not skipped
        evs = np.array(subsets["all"])
        yd, y = cell(A, evs, "ad", "b"), cell(A, evs, "a", "b")
        out[os.path.basename(model_dir)] = {
            "seeds": seeds, "E2_dedup": float(yd.mean()), "t95": av.t_interval(yd.mean(1), 0.95),
            "crossed95": re_interval(yd, 0.95), "events_positive": int((yd.mean(1) > 0).sum()), "n_events": int(len(evs)),
            "E2": float(y.mean()), "E2_t95": av.t_interval(y.mean(1), 0.95)}
    return out


def h3_per_corpus():
    """(b2): the H3 equivalence rule per corpus and decision set."""
    summary = json.load(open(SUMMARY))
    out = {}
    for model in (DISTIL, "roberta-base"):
        rH = next(r for r in summary if r["corpus"] == "humaid19" and r["model"] == model)
        rC = next(r for r in summary if r["corpus"] == "crisislext26" and r["model"] == model)
        blk = {}
        for lab, r, sub in (("CrisisLexT26", rC, "all"), ("HumAID protected", rH, "protected"), ("HumAID all", rH, "all")):
            ec = r["subsets"][sub]["P"]["event_contrib"]
            x = np.array([v for e, v in ec["E5"].items() if ec["future_share"][e] >= 0.25], float)
            t90 = av.t_interval(x, 0.90)
            blk[lab] = {"n_events": int(len(x)), "mean": float(x.mean()), "t90": t90,
                        "tost_p": tost_p(x, MARGIN) if len(x) > 1 else None,
                        "equivalent": bool(len(x) > 1 and t90[0] > -MARGIN and t90[1] < MARGIN)}
        out[model] = blk
    return out


def full_arrays(corpus, model_dir):
    """Count arrays (E, 1, C, 3) for the uncapped arm of seed 42."""
    cf, mf = av.CORPORA[corpus]
    man = json.load(open(mf))
    classes = sorted(pd.read_csv(cf, sep="\t", keep_default_na=False, usecols=["label"])["label"].unique())
    events = sorted(man["events"])
    R = {e: set(map(str, man["events"][e]["R"])) for e in events}
    sd = os.path.join(model_dir, "seed42")
    idf = av.load(sd, "id_random_full")
    if idf is None or any(av.load(sd, f"loeo_full:{e}") is None for e in events):
        return None
    a, b, c = [], [], []
    for e in events:
        l = av.load(sd, f"loeo_full:{e}")
        ae = idf[idf["event"] == e]
        lr = l[l["tweet_id"].isin(R[e])].set_index("tweet_id").loc[ae["tweet_id"]].reset_index()
        a.append(av.counts(ae, classes)); b.append(av.counts(lr, classes)); c.append(av.counts(l, classes))
    return {"a": np.array(a)[:, None], "b": np.array(b)[:, None], "c": np.array(c)[:, None]}


def arms():
    """(b3): whole-event LOEO aggregation (P) and E2 for every arm."""
    out = []
    specs = []
    for corpus in av.CORPORA:
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm", corpus, "*"))):
            specs.append(("2 epochs", corpus, os.path.basename(model_dir), model_dir))
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm_ep4", corpus, "*"))):
            specs.append(("4 epochs", corpus, os.path.basename(model_dir), model_dir))
        specs.append(("TF-IDF", corpus, "tfidf-lr", os.path.join("experiments", "runs", "x3_tfidf", corpus, "tfidf-lr")))
    for arm, corpus, model, model_dir in specs:
        A, subsets, seeds, skipped, _, _ = load_arrays(corpus, model_dir)
        rec = {"arm": arm, "corpus": corpus, "model": model, "seeds": seeds, "incomplete_seeds": skipped, "subsets": {}}
        for sub in ("all", "protected"):
            if sub in subsets:
                evs = np.array(subsets[sub])
                rec["subsets"][sub] = {"loeo_aggregation": agg_block(A, evs, len(seeds)), "E2": e2_block(A, evs)}
        out.append(rec)
    for corpus in av.CORPORA:
        A = full_arrays(corpus, os.path.join("experiments", "runs", "confirm", corpus, DISTIL))
        if A is None:
            continue
        evs = np.arange(A["a"].shape[0])
        out.append({"arm": "uncapped", "corpus": corpus, "model": DISTIL, "seeds": ["42"], "incomplete_seeds": [],
                    "subsets": {"all": {"loeo_aggregation": agg_block(A, evs, 1), "E2": e2_block(A, evs)}}})
    return out


RARE = "missing_or_found_people"


def run_row(A, evs, j, classes):
    p, d = av.estimands(A, evs, np.array([j]), "P"), av.estimands(A, evs, np.array([j]), "D")
    tp, fp, fn = (A["a"][evs, j, :, q].sum(0) for q in range(3))
    k = classes.index(RARE)
    return {"E0D": float(d["E0"]), "E1": float(p["E1"]), "E2": float(p["E2"]), "E3": float(p["E3"]),
            "MC": float(p["MC"]), "predicts_every_class": bool(((tp + fp) > 0).all()),
            "classes_never_predicted": [classes[c] for c in np.where((tp + fp) == 0)[0]],
            "rare_pooled_id_f1": float(2 * tp[k] / (2 * tp[k] + fp[k] + fn[k]))}


def four_epoch_seeds():
    """(b4): per-seed composition of the four-epoch HumAID arms beside the two-epoch recipe on the same seed, and
    of the TF-IDF arm (per-seed values of estimands reported elsewhere; descriptive)."""
    out = {}
    for model in (DISTIL, "roberta-base"):
        A4, subsets, seeds4, skipped, classes, _ = load_arrays("humaid19", os.path.join("experiments", "runs", "confirm_ep4", "humaid19", model))
        A2, _, seeds2, _, _, _ = load_arrays("humaid19", os.path.join("experiments", "runs", "confirm", "humaid19", model))
        evs = np.array(subsets["all"])
        rows = {}
        for lab, A, ss in (("ep4", A4, seeds4), ("ep2", A2, seeds2)):
            for j, s in enumerate(ss):
                rows.setdefault(s, {})[lab] = run_row(A, evs, j, classes)
        ep4 = [rows[s]["ep4"] for s in seeds4]
        out[model] = {"seeds": seeds4, "incomplete_seeds": skipped, "per_seed": rows,
                      "n_seeds_every_class_ep4": int(sum(r["predicts_every_class"] for r in ep4)),
                      "n_seeds_every_class_ep2": int(sum(rows[s]["ep2"]["predicts_every_class"] for s in seeds2)),
                      "range_ep4": {k: [float(min(r[k] for r in ep4)), float(max(r[k] for r in ep4))]
                                    for k in ("E0D", "E1", "E2", "E3", "MC")}}
    A, subsets, seeds, _, classes, _ = load_arrays("humaid19", os.path.join("experiments", "runs", "x3_tfidf", "humaid19", "tfidf-lr"))
    evs = np.array(subsets["all"])
    out["tfidf-lr"] = {"seeds": seeds, "per_seed": {s: {"tfidf": run_row(A, evs, j, classes)} for j, s in enumerate(seeds)}}
    return out


def main():
    res = {"repeats": repeats(), "dedup_E2_crisislext26": dedup_e2(), "h3_per_corpus": h3_per_corpus(),
           "arms": arms(), "four_epoch_seeds": four_epoch_seeds()}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
