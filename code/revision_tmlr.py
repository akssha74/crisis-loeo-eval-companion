"""Exploratory X5 and X6 analyses of existing predictions (registered in research/preregistration.md before they
were computed). No model is trained.

(a) practice gaps: the gap each audited practice produces on these data, beside the matched contrast, including
    the different-events comparison (X6a);
(b) HumAID with missing_or_found_people merged into injured_or_dead_people in labels and predictions;
(c) crossed random-effects (events x seeds) intervals for the per-event estimands E2 and MC, and a percentile
    bootstrap over events for MC (X6b).

    python code/revision_tmlr.py
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402

OUT = "experiments/derived/revision_tmlr.json"
MERGE = {"missing_or_found_people": "injured_or_dead_people"}
B = 2000
RNG_SEED = 20261001
_orig_load = av.load
LABEL_MAP = {}


def _mapped_load(d, key):
    df = _orig_load(d, key)
    if df is not None and LABEL_MAP:
        df["y_true"] = df["y_true"].replace(LABEL_MAP)
        df["y_pred"] = df["y_pred"].replace(LABEL_MAP)
    return df


av.load = _mapped_load


def arrays(corpus, model_dir):
    cf, mf = av.CORPORA[corpus]
    man = json.load(open(mf))
    meta = pd.read_csv(cf, sep="\t", keep_default_na=False, usecols=["event", "hazard", "label"])
    meta["label"] = meta["label"].replace(LABEL_MAP)
    classes = sorted(meta["label"].unique())
    hazard_of = meta.groupby("event")["hazard"].first().to_dict()
    seed_dirs = sorted(glob.glob(os.path.join(model_dir, "seed*")), key=lambda p: int(p.rsplit("seed", 1)[1]))
    built = [av.build(sd, man, classes, hazard_of) for sd in seed_dirs]
    A = {k: np.stack([bd[0][k] for bd in built], axis=1) for k in ("a", "b", "c")}
    events = sorted(man["events"])
    subsets = {"all": list(range(len(events)))}
    if corpus in av.PROTECTED:
        subsets["protected"] = [events.index(e) for e in av.PROTECTED[corpus]]
        subsets["development"] = [i for i, e in enumerate(events) if e not in av.PROTECTED[corpus]]
    return A, subsets, [os.path.basename(s)[4:] for s in seed_dirs]


def different_events_table(A, conv):
    """X6a: per (target event, seed), pooled in-distribution score on the R tweets of all other events minus the
    whole-event LOEO score. The pool is over every other event of the corpus (the LOEO model's training events),
    so it stays fixed when target events are resampled."""
    a = np.nan_to_num(A["a"])
    total = a.sum(axis=0)
    others = np.stack([av.f1_from_counts(total - a[e], conv) for e in range(a.shape[0])])
    return others - av.f1_from_counts(A["c"], conv)


def practices(A, ev, sd, conv, dtab):
    f = lambda k: av.f1_from_counts(A[k][np.ix_(ev, sd)], conv)
    pool = lambda k: av.f1_from_counts(np.nansum(A[k][np.ix_(ev, sd)], axis=0), conv)
    a, b, c = f("a"), f("b"), f("c")
    pa, pb, pc = pool("a"), pool("b"), pool("c")
    return {"matched_event": float(np.mean(a - b)),           # E2: per-event, identical tweets
            "matched_pooled": float(np.mean(pa - pb)),         # E2^pool: pooled, identical tweets
            "pooled_vs_pooled_whole": float(np.mean(pa - pc)),  # one pooled test set over held-out events
            "event_vs_event_whole": float(np.mean(a.mean(0) - c.mean(0))),  # per-event means, different tweets
            "different_events_pooled": float(np.mean(dtab[np.ix_(ev, sd)])),  # X6a: ID pooled on other events
            "composite": float(np.mean(pa - c.mean(0)))}        # E0: pooled ID minus per-event LOEO mean


def mc_event_bootstrap(Y):
    """X6b: percentile interval over events of seed-averaged MC contributions (MC >= 0 by construction)."""
    x = Y.mean(1)
    rng = np.random.default_rng(RNG_SEED)
    draws = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return {"mean": float(x.mean()), "pct95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_positive": int((x > 1e-12).sum()), "n_events": int(len(x))}


def boot(fn, n_e, n_s):
    rng = np.random.default_rng(RNG_SEED)
    return av.crossed_boot(fn, n_e, n_s, rng, B)


def re_interval(Y, level):
    """Crossed (events x seeds) interval for the grand mean of an events x seeds table.

    Var(mean) = s_e^2/E + s_s^2/S + s_r^2/(ES). The reported (conservative) interval uses (MS_e + MS_s)/(ES), whose
    expectation exceeds that variance by s_r^2/(ES), with Satterthwaite df of the two positive terms. The unbiased
    estimate (MS_e + MS_s - MS_r)/(ES) is kept as well; its Satterthwaite df collapses when MS_s and MS_r nearly
    cancel."""
    Y = np.asarray(Y, float)
    E, S = Y.shape
    m, me, ms = Y.mean(), Y.mean(1), Y.mean(0)
    mse = S * ((me - m) ** 2).sum() / (E - 1)
    out = {"mean": float(m), "se_events_only": float(np.sqrt(mse / (E * S)))}
    if S < 2:
        v, df = mse / (E * S), E - 1
        vu, dfu = v, df
    else:
        mss = E * ((ms - m) ** 2).sum() / (S - 1)
        msr = ((Y - me[:, None] - ms[None, :] + m) ** 2).sum() / ((E - 1) * (S - 1))
        v = (mse + mss) / (E * S)
        df = (mse + mss) ** 2 / (mse ** 2 / (E - 1) + mss ** 2 / (S - 1))
        num = mse + mss - msr
        if num <= mse:
            vu, dfu = mse / (E * S), E - 1
        else:
            vu = num / (E * S)
            dfu = num ** 2 / (mse ** 2 / (E - 1) + mss ** 2 / (S - 1) + msr ** 2 / ((E - 1) * (S - 1)))
    h, hu = tdist.ppf(0.5 + level / 2, df) * np.sqrt(v), tdist.ppf(0.5 + level / 2, dfu) * np.sqrt(vu)
    out.update({"lo": float(m - h), "hi": float(m + h), "df": float(df), "se": float(np.sqrt(v)),
                "unbiased": {"lo": float(m - hu), "hi": float(m + hu), "df": float(dfu), "se": float(np.sqrt(vu))}})
    return out


def per_cell(A, ev, kind):
    if kind == "E2":
        return av.f1_from_counts(A["a"][ev], "P") - av.f1_from_counts(A["b"][ev], "P")
    return av.f1_from_counts(A["c"][ev], "P") - av.f1_from_counts(A["c"][ev], "D")


def merged_analysis(model_dir):
    """Full amended analysis of HumAID with the merged label scheme (convention P and D, all subsets)."""
    cf, mf = av.CORPORA["humaid19"]
    meta = pd.read_csv(cf, sep="\t", keep_default_na=False, dtype={"tweet_id": str})
    meta["label"] = meta["label"].replace(MERGE)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    meta.to_csv(tmp.name, sep="\t", index=False)
    saved = av.CORPORA["humaid19"]
    av.CORPORA["humaid19"] = (tmp.name, mf)
    try:
        res = av.analyse("humaid19", model_dir)
    finally:
        av.CORPORA["humaid19"] = saved
        os.unlink(tmp.name)
    out = {}
    for sub, blk in res["subsets"].items():
        out[sub] = {"P": {k: blk["P"][k] for k in ("E0", "E1", "E2", "E3", "E2p", "MC")},
                    "D": {k: blk["D"][k] for k in ("E0",)},
                    "headline": blk["headline"], "MC_events_affected": blk["MC_events_affected"],
                    "MC_share_of_E0D": blk["MC_share_of_E0D"], "t_rules_E2": blk["P"]["t_rules"]["E2"],
                    "t_rules_MC": blk["P"]["t_rules"]["MC"]}
    return out


def main():
    global LABEL_MAP
    results = {"practices": {}, "random_effects": {}, "merged_humaid": {}, "encoder_difference": {},
               "mc_bootstrap": {}}
    cells = {}
    for corpus in av.CORPORA:
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm", corpus, "*"))):
            model = os.path.basename(model_dir)
            print("practices / random effects:", corpus, model, flush=True)
            LABEL_MAP = {}
            A, subsets, seeds = arrays(corpus, model_dir)
            key = f"{corpus}/{model}"
            results["practices"][key], results["random_effects"][key] = {}, {}
            results["mc_bootstrap"][key] = {}
            dtabs = {conv: different_events_table(A, conv) for conv in ("P", "D")}
            for sub, evs in subsets.items():
                evs = np.array(evs)
                S = len(seeds)
                blk = {}
                for conv in ("P", "D"):
                    point = practices(A, evs, np.arange(S), conv, dtabs[conv])
                    draws = boot(lambda e, s: practices(A, evs[e], s, conv, dtabs[conv]), len(evs), S)
                    blk[conv] = {k: {"mean": v, "ci95": [float(np.percentile([d[k] for d in draws], 2.5)),
                                                         float(np.percentile([d[k] for d in draws], 97.5))]}
                                 for k, v in point.items()}
                results["practices"][key][sub] = blk
                re = {}
                for kind in ("E2", "MC"):
                    Y = per_cell(A, evs, kind)
                    re[kind] = {"t95": re_interval(Y, 0.95), "t90": re_interval(Y, 0.90),
                                "seeds": seeds, "per_seed_mean": [float(x) for x in Y.mean(0)]}
                    cells[(corpus, model, sub, kind)] = Y
                results["random_effects"][key][sub] = re
                results["mc_bootstrap"][key][sub] = mc_event_bootstrap(cells[(corpus, model, sub, "MC")])
    for corpus, subs in (("humaid19", ("all", "protected", "development")), ("crisislext26", ("all",))):
        for sub in subs:
            yd = cells[(corpus, "distilbert-base-uncased", sub, "E2")]
            yr = cells[(corpus, "roberta-base", sub, "E2")]
            d, r = re_interval(yd, 0.90), re_interval(yr, 0.90)
            diff = yr.mean() - yd.mean()
            se = np.sqrt(d["se"] ** 2 + r["se"] ** 2)
            df = se ** 4 / (d["se"] ** 4 / d["df"] + r["se"] ** 4 / r["df"])
            h90 = tdist.ppf(0.95, df) * se
            results["encoder_difference"][f"{corpus}/{sub}"] = {
                "R_minus_D": float(diff), "re_t90": [float(diff - h90), float(diff + h90)], "df": float(df)}
    LABEL_MAP = dict(MERGE)
    for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", "confirm", "humaid19", "*"))):
        print("merged HumAID:", model_dir, flush=True)
        results["merged_humaid"][os.path.basename(model_dir)] = merged_analysis(model_dir)
    LABEL_MAP = {}
    json.dump(results, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
