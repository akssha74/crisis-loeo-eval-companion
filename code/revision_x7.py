"""Exploratory X7(b) analyses of existing predictions (registered in research/preregistration.md at
2026-10-01T12:56Z, before any of them was computed). No model is trained.

(a) crossed events x seeds intervals for DUP, DUP_F and E3, per-seed DUP, and two one-sided tests of DUP at
    +-0.01 and +-0.02;
(b) LOEO aggregation: pooled minus per-event macro-F1 of the same whole-event LOEO predictions, crossed bootstrap;
(c) per-seed protected E2 of the four-epoch arms, and the four-epoch estimands for seeds 42, 1, 2 and for every
    completed seed;
(d) Holm adjustment over the ten registered decisions, and H1/H2 materiality judged on the 95% lower bound;
(e) the loss-based epoch rule beside the macro-F1 rule;
(f) per-class recall of whole-event LOEO predictions for three HumAID classes, pooled and per-event mean.

    python code/revision_x7.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
from revision_tmlr import re_interval  # noqa: E402

OUT = "experiments/derived/revision_x7.json"
SUMMARY = "experiments/derived/summary_v2_confirm.json"
CONV_DIR = "experiments/derived/convergence"
B = 2000
RNG_SEED = 20261001
MARGIN = 0.02
TOST_MARGINS = (0.01, 0.02)
OLD_SEEDS = ["42", "1", "2"]
RECALL_CLASSES = ("missing_or_found_people", "injured_or_dead_people", "requests_or_urgent_needs")
KEYS = ("a", "b", "c", "at", "bt", "ad", "atd", "af", "k", "n", "kT", "nT", "kk")


def load_arrays(corpus, model_dir):
    """Count arrays (E, S, C, 3) over the seeds whose registered runs are complete; incomplete seeds are listed."""
    cf, mf = av.CORPORA[corpus]
    man = json.load(open(mf))
    meta = pd.read_csv(cf, sep="\t", keep_default_na=False, usecols=["event", "hazard", "label"])
    classes = sorted(meta["label"].unique())
    hazard_of = meta.groupby("event")["hazard"].first().to_dict()
    seed_dirs = sorted(glob.glob(os.path.join(model_dir, "seed*")), key=lambda p: int(p.rsplit("seed", 1)[1]))
    built, seeds, skipped = [], [], []
    for sd in seed_dirs:
        try:
            built.append(av.build(sd, man, classes, hazard_of)[0])
            seeds.append(os.path.basename(sd)[4:])
        except SystemExit as err:
            skipped.append({"seed": os.path.basename(sd)[4:], "reason": str(err)})
    A = {k: np.stack([bd[k] for bd in built], axis=1) for k in KEYS}
    events = sorted(man["events"])
    subsets = {"all": list(range(len(events)))}
    if corpus in av.PROTECTED:
        subsets["protected"] = [events.index(e) for e in av.PROTECTED[corpus]]
        subsets["development"] = [i for i, e in enumerate(events) if e not in av.PROTECTED[corpus]]
    return A, subsets, seeds, skipped, classes, events


def cell(A, ev, k1, k2, conv="P"):
    return av.f1_from_counts(A[k1][ev], conv) - av.f1_from_counts(A[k2][ev], conv)


def crossed(Y, margins=()):
    out = {"t95": re_interval(Y, 0.95), "t90": re_interval(Y, 0.90), "per_seed_mean": [float(x) for x in Y.mean(0)]}
    for m in margins:
        t90, u90 = out["t90"], out["t90"]["unbiased"]
        out[f"tost_{m}"] = {"conservative": bool(t90["lo"] > -m and t90["hi"] < m),
                            "unbiased": bool(u90["lo"] > -m and u90["hi"] < m)}
    return out


def loeo_aggregation(A, ev, sd, conv):
    c = A["c"][np.ix_(ev, sd)]
    return float(np.mean(av.f1_from_counts(np.nansum(c, axis=0), conv) - av.f1_from_counts(c, conv).mean(0)))


def boot_ci(fn, n_e, n_s):
    rng = np.random.default_rng(RNG_SEED)
    draws = av.crossed_boot(fn, n_e, n_s, rng, B)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def evaluation_terms(A, subsets, seeds):
    """(a) and (b) for one corpus x encoder (registered two-epoch recipe)."""
    out = {}
    S = len(seeds)
    fuzzy = np.array([j for j in range(S) if not np.isnan(A["af"][0, j, 0, 0])], int)
    for sub, evs in subsets.items():
        evs = np.array(evs)
        blk = {"DUP": crossed(cell(A, evs, "a", "ad"), TOST_MARGINS),
               "E3": crossed(cell(A, evs, "b", "c"))}
        blk["DUP"]["seeds"] = seeds
        if len(fuzzy) >= 2:
            blk["DUP_F"] = crossed(cell(A, evs, "a", "af")[:, fuzzy], TOST_MARGINS)
            blk["DUP_F"]["seeds"] = [seeds[j] for j in fuzzy]
        blk["loeo_aggregation"] = {}
        for conv in ("P", "D"):
            blk["loeo_aggregation"][conv] = {
                "mean": loeo_aggregation(A, evs, np.arange(S), conv),
                "ci95": boot_ci(lambda e, s: loeo_aggregation(A, evs[e], s, conv), len(evs), S),
                "per_seed": [loeo_aggregation(A, evs, np.array([j]), conv) for j in range(S)]}
        out[sub] = blk
    return out


def four_epoch(model):
    """(c): four-epoch arms, per seed and for the old and the full seed sets, beside the two-epoch recipe."""
    A4, subsets, seeds4, skipped, _, _ = load_arrays("humaid19", os.path.join("experiments", "runs", "confirm_ep4", "humaid19", model))
    A2, _, seeds2, _, _, _ = load_arrays("humaid19", os.path.join("experiments", "runs", "confirm", "humaid19", model))
    out = {"seeds": seeds4, "incomplete_seeds": skipped, "subsets": {}}
    sets = {"old": [s for s in OLD_SEEDS if s in seeds4], "complete": seeds4}
    for sub, evs in subsets.items():
        evs = np.array(evs)
        y4, y2 = cell(A4, evs, "a", "b"), cell(A2, evs, "a", "b")
        blk = {"per_seed_E2_ep4": dict(zip(seeds4, map(float, y4.mean(0)))),
               "per_seed_E2_ep2": dict(zip(seeds2, map(float, y2.mean(0))))}
        for lab, ss in sets.items():
            j4 = np.array([seeds4.index(s) for s in ss])
            j2 = np.array([seeds2.index(s) for s in ss])
            p4, d4 = av.estimands(A4, evs, j4, "P"), av.estimands(A4, evs, j4, "D")
            p2, d2 = av.estimands(A2, evs, j2, "P"), av.estimands(A2, evs, j2, "D")
            ev4 = y4[:, j4].mean(1)
            blk[lab] = {"seeds": ss,
                        "ep4": {**{k: float(p4[k]) for k in ("E0", "E1", "E2", "E2p", "E3", "MC", "mean_c", "pooled_id")},
                                "E0D": float(d4["E0"]), "E2_t95": av.t_interval(ev4, 0.95),
                                "E2_events_positive": int((ev4 > 0).sum()), "n_events": int(len(ev4)),
                                "E2_crossed": re_interval(y4[:, j4], 0.95) if len(j4) > 1 else None},
                        "ep2": {**{k: float(p2[k]) for k in ("E0", "E1", "E2", "E2p", "E3", "MC", "mean_c", "pooled_id")},
                                "E0D": float(d2["E0"]), "E2_t95": av.t_interval(y2[:, j2].mean(1), 0.95)}}
        out["subsets"][sub] = blk
    return out


def t_two_sided_p(x):
    x = np.asarray(x, float)
    n, se = len(x), x.std(ddof=1) / np.sqrt(len(x))
    return float(2 * tdist.sf(abs(x.mean() / se), n - 1))


def tost_p(x, m):
    x = np.asarray(x, float)
    n, se = len(x), x.std(ddof=1) / np.sqrt(len(x))
    return float(max(tdist.sf((x.mean() + m) / se, n - 1), tdist.cdf((x.mean() - m) / se, n - 1)))


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, run = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def registered_decisions(summary):
    """(d): the ten registered decisions from the seed-averaged event contributions of the registered t rules."""
    MODEL = {"distilbert-base-uncased": "DistilBERT", "roberta-base": "RoBERTa"}
    rows = []
    for model, name in MODEL.items():
        rH = next(r for r in summary if r["corpus"] == "humaid19" and r["model"] == model)
        rC = next(r for r in summary if r["corpus"] == "crisislext26" and r["model"] == model)
        tests = [("H1", rH["subsets"]["protected"]["P"], "E2"), ("H1b", rC["subsets"]["all"]["P"], "E2"),
                 ("H2 protected", rH["subsets"]["protected"]["P"], "MC"), ("H2 CrisisLexT26", rC["subsets"]["all"]["P"], "MC")]
        for lab, blk, est in tests:
            x = np.array(list(blk["event_contrib"][est].values()), float)
            ci = av.t_interval(x, 0.95)
            assert np.allclose(ci, blk["t_rules"][est]["t95"]), f"t interval mismatch for {lab} {name}"
            rows.append({"decision": lab, "encoder": name, "kind": "material_positive", "mean": float(x.mean()),
                         "t95": ci, "p": t_two_sided_p(x), "n_events": int(len(x)),
                         "material_on_lower_bound": bool(ci[0] >= MARGIN)})
        x = []
        for r, sub in ((rH, "protected"), (rC, "all")):
            ec = r["subsets"][sub]["P"]["event_contrib"]
            x += [v for e, v in ec["E5"].items() if ec["future_share"][e] >= 0.25]
        x = np.array(x, float)
        rows.append({"decision": "H3", "encoder": name, "kind": "equivalence", "mean": float(x.mean()),
                     "t90": av.t_interval(x, 0.90), "p": tost_p(x, MARGIN), "n_events": int(len(x))})
    adj = holm(np.array([r["p"] for r in rows]))
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        if r["kind"] == "material_positive":
            r["supported_raw"] = bool(r["p"] < 0.05 and r["mean"] >= MARGIN)
            r["supported_holm"] = bool(a < 0.05 and r["mean"] >= MARGIN)
            r["positive_holm"] = bool(a < 0.05 and r["mean"] > 0)
        else:
            r["supported_raw"] = bool(r["p"] < 0.05)
            r["supported_holm"] = bool(a < 0.05)
    return {"family": "ten registered decisions (H1, H1b, H2 protected, H2 CrisisLexT26, H3; two encoders)",
            "alpha": 0.05, "rows": rows}


def epoch_rules():
    """(e): per encoder, the epoch of highest mean source-validation macro-F1 (X6c) and of lowest mean loss."""
    files = sorted(glob.glob(os.path.join(CONV_DIR, "*", "*", "*.json")))
    out = {}
    for model in sorted({os.path.basename(os.path.dirname(f)) for f in files}):
        curves = [json.load(open(f))["curve"] for f in files if os.path.basename(os.path.dirname(f)) == model]
        f1 = [float(np.mean([c[e]["val_macro_f1_present"] for c in curves])) for e in range(len(curves[0]))]
        loss = [float(np.mean([c[e]["val_loss"] for c in curves])) for e in range(len(curves[0]))]
        out[model] = {"n_folds": len(curves), "mean_val_macro_f1": f1, "mean_val_loss": loss,
                      "macro_f1_rule": 1 + int(np.argmax(np.round(f1, 3))), "loss_rule": 1 + int(np.argmin(loss)),
                      "fold_loss_min_epochs": [1 + int(np.argmin([x["val_loss"] for x in c])) for c in curves]}
    return out


def class_recall(A, classes, events):
    """(f): recall and F1 of one class from (E, S, C, 3) counts, pooled over events and averaged over the events
    where the class occurs, plus predictions of the class in events where it does not occur."""
    out = {}
    for name in RECALL_CLASSES:
        k = classes.index(name)
        res = {}
        for key, lab in (("c", "loeo_whole"), ("a", "id_R")):
            tp, fp, fn = A[key][:, :, k, 0], A[key][:, :, k, 1], A[key][:, :, k, 2]
            present = (tp + fn)[:, 0] > 0
            rec_pool = tp[present].sum(0) / (tp + fn)[present].sum(0)
            rec_ev = (tp[present] / (tp + fn)[present]).mean(0)
            f1_pool = 2 * tp.sum(0) / (2 * tp.sum(0) + fp.sum(0) + fn.sum(0))
            den = 2 * tp[present] + fp[present] + fn[present]
            f1_ev = np.where(den > 0, 2 * tp[present] / np.where(den > 0, den, 1), 0).mean(0)
            res[lab] = {"recall_pooled": float(rec_pool.mean()), "recall_event_mean": float(rec_ev.mean()),
                        "recall_pooled_per_seed": [float(x) for x in rec_pool],
                        "recall_event_mean_per_seed": [float(x) for x in rec_ev],
                        "f1_pooled": float(f1_pool.mean()), "f1_event_mean": float(f1_ev.mean()),
                        "false_alarms_absent_events": float(fp[~present].sum(0).mean()),
                        "events_absent": int((~present).sum())}
        res["events_present"] = int(((A["c"][:, 0, k, 0] + A["c"][:, 0, k, 2]) > 0).sum())
        res["support_whole"] = int((A["c"][:, 0, k, 0] + A["c"][:, 0, k, 2]).sum())
        res["support_R"] = int((A["a"][:, 0, k, 0] + A["a"][:, 0, k, 2]).sum())
        out[name] = res
    return out


def main():
    summary = json.load(open(SUMMARY))
    results = {"evaluation_terms": {}, "four_epoch": {}, "class_recall": {}}
    for corpus in av.CORPORA:
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm", corpus, "*"))):
            model = os.path.basename(model_dir)
            print("evaluation terms:", corpus, model, flush=True)
            A, subsets, seeds, skipped, classes, events = load_arrays(corpus, model_dir)
            assert not skipped, f"incomplete registered runs in {model_dir}: {skipped}"
            results["evaluation_terms"][f"{corpus}/{model}"] = evaluation_terms(A, subsets, seeds)
            if corpus == "humaid19":
                results["class_recall"][model] = class_recall(A, classes, events)
    for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm_ep4", "humaid19", "*"))):
        print("four-epoch arms:", model_dir, flush=True)
        results["four_epoch"][os.path.basename(model_dir)] = four_epoch(os.path.basename(model_dir))
    results["holm"] = registered_decisions(summary)
    results["epoch_rules"] = epoch_rules()
    json.dump(results, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
