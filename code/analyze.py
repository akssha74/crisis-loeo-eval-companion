"""Confirmatory analysis: registered estimands from saved predictions, across seeds, per corpus/model/subset.

Per seed s and event e (F1 = macro-F1; convention P = classes present in the evaluated y_true,
convention D = scikit-learn default, classes in y_true or y_pred):
  a_e  F1(id_random, R_e)        b_e  F1(loeo_e, R_e)        c_e  F1(loeo_e, all of e)
  at_e F1(id_temporal, T_e)      bt_e F1(loeo_e, T_e)
  ad_e F1(id_random_dedup, R_e)  atd_e F1(id_temporal_dedup, T_e)
  k_e  F1(chrono_e, all of e)    n_e F1(size-matched loeo, all of e)
Estimands (event means with equal event weight; pooled F1 over the union of R):
  E0  conventional gap        F1pool(id_random, R) - mean c_e
  E1  aggregation             F1pool(id_random, R) - mean a_e
  E2  matched exposure        mean (a_e - b_e)
  E3  test subsample          mean (b_e - c_e)                  E0 = E1 + E2 + E3
  E4  matched temporal        mean (at_e - bt_e)
  E5  future-event leakage    mean (n_e - k_e) over chrono-eligible events
  MC  metric convention       mean (c_e[P] - c_e[D])            how much convention D lowers LOEO scores
  DUP duplicate leakage       mean (a_e - ad_e);  DUPT = mean (at_e - atd_e)
Uncertainty: hierarchical bootstrap (events, then seeds within event), 5,000 draws; event-level Wilcoxon
on seed-averaged differences; across-seed SD of each estimand.

    python code/analyze.py --plan confirm
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

B = 5000
RNG_SEED = 20260929
MATERIAL = 0.02
CORPORA = {"humaid19": "experiments/splits/confirm_humaid19.json",
           "crisislext26": "experiments/splits/confirm_crisislext26.json"}
PROTECTED = {"humaid19": ["california_wildfires_2018", "hurricane_dorian_2019", "hurricane_florence_2018",
                          "kerala_floods_2018", "midwestern_us_floods_2019", "pakistan_earthquake_2019"]}


def f1(y, p, conv):
    if conv == "P":
        return f1_score(y, p, average="macro", labels=sorted(set(y)), zero_division=0)
    return f1_score(y, p, average="macro", zero_division=0)


def load(d, key):
    p = os.path.join(d, key.replace(":", "__"), "preds.tsv.gz")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t", dtype={"tweet_id": str}, keep_default_na=False)


def per_seed(seed_dir, man):
    events = sorted(man["events"])
    R = {e: set(map(str, man["events"][e]["R"])) for e in events}
    T = {e: set(map(str, man["events"][e]["T"])) for e in events}
    idr, idt = load(seed_dir, "id_random"), load(seed_dir, "id_temporal")
    idrd, idtd = load(seed_dir, "id_random_dedup"), load(seed_dir, "id_temporal_dedup")
    rows, pool = [], {}
    for conv in ["P", "D"]:
        pool[conv] = f1(idr["y_true"], idr["y_pred"], conv)
    loeo_R_all = []
    for e in events:
        l, k, n = load(seed_dir, f"loeo:{e}"), load(seed_dir, f"chrono:{e}"), load(seed_dir, f"loeon:{e}")
        if l is None:
            raise SystemExit(f"missing loeo:{e} in {seed_dir}")
        a, at = idr[idr["event"] == e], idt[idt["event"] == e]
        ad, atd = idrd[idrd["event"] == e], idtd[idtd["event"] == e]
        lr = l[l["tweet_id"].isin(R[e])]
        lt = l[l["tweet_id"].isin(T[e])]
        loeo_R_all.append(lr)
        stray = l["y_pred"][~l["y_pred"].isin(set(l["y_true"]))]
        row = {"event": e, "n": int(len(l)), "n_true_classes": int(l["y_true"].nunique()),
               "stray_pred_classes": int(stray.nunique()), "stray_preds": int(len(stray))}
        for conv in ["P", "D"]:
            row[f"a_{conv}"] = f1(a["y_true"], a["y_pred"], conv)
            row[f"b_{conv}"] = f1(lr["y_true"], lr["y_pred"], conv)
            row[f"c_{conv}"] = f1(l["y_true"], l["y_pred"], conv)
            row[f"at_{conv}"] = f1(at["y_true"], at["y_pred"], conv)
            row[f"bt_{conv}"] = f1(lt["y_true"], lt["y_pred"], conv)
            row[f"ad_{conv}"] = f1(ad["y_true"], ad["y_pred"], conv)
            row[f"atd_{conv}"] = f1(atd["y_true"], atd["y_pred"], conv)
            if k is not None:
                ref = n if n is not None else l
                row[f"k_{conv}"] = f1(k["y_true"], k["y_pred"], conv)
                row[f"n_{conv}"] = f1(ref["y_true"], ref["y_pred"], conv)
        rows.append(row)
    lR = pd.concat(loeo_R_all).set_index("tweet_id").loc[idr["tweet_id"]].reset_index()
    for conv in ["P", "D"]:
        pool[f"loeoR_{conv}"] = f1(lR["y_true"], lR["y_pred"], conv)
    return pd.DataFrame(rows), pool


def diffs(tab, conv):
    """Per-event contributions whose event mean is each estimand (E0/E1 also need the pooled score)."""
    d = pd.DataFrame({"event": tab["event"]})
    d["E2"] = tab[f"a_{conv}"] - tab[f"b_{conv}"]
    d["E3"] = tab[f"b_{conv}"] - tab[f"c_{conv}"]
    d["E4"] = tab[f"at_{conv}"] - tab[f"bt_{conv}"]
    d["DUP"] = tab[f"a_{conv}"] - tab[f"ad_{conv}"]
    d["DUPT"] = tab[f"at_{conv}"] - tab[f"atd_{conv}"]
    if f"k_{conv}" in tab:
        d["E5"] = tab[f"n_{conv}"] - tab[f"k_{conv}"]
    d["MC"] = tab["c_P"] - tab["c_D"]
    d["a"], d["c"] = tab[f"a_{conv}"], tab[f"c_{conv}"]
    return d.set_index("event")


def hier_boot(mat, rng, b=B):
    """mat: events x seeds array (NaN allowed for ineligible events). Returns bootstrap means."""
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    n_e, n_s = mat.shape
    out = np.empty(b)
    for i in range(b):
        ev = rng.integers(0, n_e, n_e)
        sd = rng.integers(0, n_s, (n_e, n_s))
        out[i] = np.nanmean(mat[ev[:, None], sd])
    return out


def summarize(mat, rng):
    mat = mat[~np.all(np.isnan(mat), axis=1)]
    ev_mean = np.nanmean(mat, axis=1)
    boot = hier_boot(mat, rng)
    seed_means = np.nanmean(mat, axis=0)
    res = {"mean": float(np.nanmean(mat)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
           "n_events": int(len(ev_mean)), "n_seeds": int(mat.shape[1]),
           "events_positive": int((ev_mean > 0).sum()), "seed_sd": float(np.std(seed_means, ddof=1)) if mat.shape[1] > 1 else None}
    if len(ev_mean) >= 5 and np.any(ev_mean != 0):
        res["wilcoxon_p"] = float(wilcoxon(ev_mean).pvalue)
    res["material_positive"] = bool(res["ci95"][0] > 0 and res["mean"] >= MATERIAL)
    res["equivalent_within_material"] = bool(res["ci90"][0] > -MATERIAL and res["ci90"][1] < MATERIAL)
    return res


def analyse(plan, corpus, model_dir):
    man = json.load(open(CORPORA[corpus]))
    seed_dirs = sorted(glob.glob(os.path.join(model_dir, "seed*")))
    tabs, pools = {}, {}
    for sd in seed_dirs:
        s = os.path.basename(sd)
        tabs[s], pools[s] = per_seed(sd, man)
    seeds = sorted(tabs)
    events = sorted(man["events"])
    subsets = {"all": events}
    if corpus in PROTECTED:
        subsets["protected"] = PROTECTED[corpus]
        subsets["development"] = [e for e in events if e not in PROTECTED[corpus]]
    out = {"corpus": corpus, "model": os.path.basename(model_dir), "seeds": seeds, "conventions": {}}
    for conv in ["P", "D"]:
        rng = np.random.default_rng(RNG_SEED)
        D = {s: diffs(tabs[s], conv) for s in seeds}
        res = {}
        for name, evs in subsets.items():
            r = {}
            for est in ["E2", "E3", "E4", "E5", "MC", "DUP", "DUPT"]:
                if est not in D[seeds[0]]:
                    continue
                mat = np.column_stack([D[s].loc[evs, est].to_numpy(dtype=float) for s in seeds])
                if np.all(np.isnan(mat)):
                    continue
                r[est] = summarize(mat, rng)
            if name == "all":
                e0 = [pools[s][conv] - D[s]["c"].mean() for s in seeds]
                e1 = [pools[s][conv] - D[s]["a"].mean() for s in seeds]
                e2p = [pools[s][conv] - pools[s][f"loeoR_{conv}"] for s in seeds]
                r["E0"] = {"mean": float(np.mean(e0)), "seed_sd": float(np.std(e0, ddof=1)) if len(e0) > 1 else None}
                r["E1"] = {"mean": float(np.mean(e1)), "seed_sd": float(np.std(e1, ddof=1)) if len(e1) > 1 else None}
                r["E2p"] = {"mean": float(np.mean(e2p)), "seed_sd": float(np.std(e2p, ddof=1)) if len(e2p) > 1 else None}
                r["pooled_id_random"] = float(np.mean([pools[s][conv] for s in seeds]))
                r["pooled_loeo_on_R"] = float(np.mean([pools[s][f"loeoR_{conv}"] for s in seeds]))
                r["decomposition_residual"] = float(r["E0"]["mean"] - (r["E1"]["mean"] + r["E2"]["mean"] + r["E3"]["mean"]))
            r["mean_a"] = float(np.mean([D[s].loc[evs, "a"].mean() for s in seeds]))
            r["mean_c"] = float(np.mean([D[s].loc[evs, "c"].mean() for s in seeds]))
            res[name] = r
        out["conventions"][conv] = res
    ev_tab = pd.concat([tabs[s].assign(seed=s) for s in seeds])
    out["per_event_seed_mean"] = ev_tab.drop(columns="seed").groupby("event").mean().round(4).reset_index().to_dict(orient="records")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="confirm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = []
    for corpus in CORPORA:
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", args.plan, corpus, "*"))):
            print("analysing", model_dir, flush=True)
            results.append(analyse(args.plan, corpus, model_dir))
    out = args.out or os.path.join("experiments", "derived", f"summary_{args.plan}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    for r in results:
        for conv in ["P", "D"]:
            for sub, v in r["conventions"][conv].items():
                line = {k: (round(x["mean"], 4) if isinstance(x, dict) else x) for k, x in v.items()
                        if k in ("E0", "E1", "E2", "E3", "E4", "E5", "MC", "DUP")}
                print(r["corpus"], r["model"], conv, sub, line)
    print("wrote", out)


if __name__ == "__main__":
    main()
