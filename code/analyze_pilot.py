"""Compute the registered pilot estimands from saved per-instance predictions only.

Estimands (macro-F1; per-event values averaged with equal event weight unless stated):
  E0  prior-study gap      F1pool(ID_random, R) - mean_e F1(LOEO_e, e)
  E1  aggregation          F1pool(ID_random, R) - mean_e F1(ID_random, R_e)
  E2  matched exposure     mean_e [F1(ID_random, R_e) - F1(LOEO_e, R_e)]
  E3  test subsample       mean_e F1(LOEO_e, R_e) - mean_e F1(LOEO_e, e)          (E0 = E1 + E2 + E3)
  E2p pooled matched       F1pool(ID_random, R) - F1pool(LOEO -> R)
  E4  matched temporal     mean_e [F1(ID_temporal, T_e) - F1(LOEO_e, T_e)]
  E5  future-event leakage mean_{e in chrono} [F1(LOEO size-matched, e) - F1(CHRONO_e, e)]

    python code/analyze_pilot.py --runs experiments/runs/pilot/distilbert-base-uncased/seed42
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

B = 10000
RNG_SEED = 7


def f1(y_true, y_pred, convention):
    if convention == "present":
        return f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true)), zero_division=0)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def load(runs_dir, key):
    p = os.path.join(runs_dir, key.replace(":", "__"), "preds.tsv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t", dtype={"tweet_id": str}, keep_default_na=False)


def event_bootstrap(diffs, rng):
    d = np.asarray(diffs, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pooled_paired_bootstrap(a, b, events, convention, rng):
    """Instance bootstrap stratified by event for F1pool(a) - F1pool(b) on identical instances."""
    y = a["y_true"].to_numpy()
    pa, pb = a["y_pred"].to_numpy(), b["y_pred"].to_numpy()
    groups = [np.where(events == e)[0] for e in np.unique(events)]
    out = np.empty(2000)
    for i in range(len(out)):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[i] = f1(y[idx], pa[idx], convention) - f1(y[idx], pb[idx], convention)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def summarize(diffs, rng):
    d = np.asarray(diffs, float)
    res = {"mean": float(d.mean()), "event_bootstrap_95ci": event_bootstrap(d, rng),
           "n_events": int(len(d)), "n_positive": int((d > 0).sum())}
    if len(d) >= 5 and np.any(d != 0):
        res["wilcoxon_p"] = float(wilcoxon(d).pvalue)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", default="experiments/splits/pilot_splits.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    events = sorted(man["events"])
    R = {e: set(map(str, man["events"][e]["R"])) for e in events}
    T = {e: set(map(str, man["events"][e]["T"])) for e in events}

    idr = load(args.runs, "id_random")
    idt = load(args.runs, "id_temporal")
    loeo = {e: load(args.runs, f"loeo:{e}") for e in events}
    chrono = {e: load(args.runs, f"chrono:{e}") for e in events}
    loeon = {e: load(args.runs, f"loeon:{e}") for e in events}
    missing = [k for k, v in [("id_random", idr), ("id_temporal", idt)] + [(f"loeo:{e}", loeo[e]) for e in events]
               if v is None]
    if missing:
        raise SystemExit(f"missing runs: {missing}")

    summary = {"runs_dir": args.runs, "conventions": {}}
    for conv in ["all_predicted", "present"]:
        rng = np.random.default_rng(RNG_SEED)
        rows = []
        for e in events:
            a = idr[idr["event"] == e]
            l = loeo[e]
            lr = l[l["tweet_id"].isin(R[e])].set_index("tweet_id").loc[a["tweet_id"]].reset_index()
            t = idt[idt["event"] == e]
            lt = l[l["tweet_id"].isin(T[e])].set_index("tweet_id").loc[t["tweet_id"]].reset_index()
            row = {
                "event": e, "n": int(len(l)),
                "id_random_R": f1(a["y_true"], a["y_pred"], conv),
                "loeo_R": f1(lr["y_true"], lr["y_pred"], conv),
                "loeo_all": f1(l["y_true"], l["y_pred"], conv),
                "id_temporal_T": f1(t["y_true"], t["y_pred"], conv),
                "loeo_T": f1(lt["y_true"], lt["y_pred"], conv),
            }
            if chrono[e] is not None:
                ref = loeon[e] if loeon[e] is not None else l
                row["chrono_all"] = f1(chrono[e]["y_true"], chrono[e]["y_pred"], conv)
                row["loeo_sizematched_all"] = f1(ref["y_true"], ref["y_pred"], conv)
                cT = chrono[e][chrono[e]["tweet_id"].isin(T[e])]
                row["chrono_T"] = f1(cT["y_true"], cT["y_pred"], conv)
                refT = ref[ref["tweet_id"].isin(T[e])]
                row["loeo_sizematched_T"] = f1(refT["y_true"], refT["y_pred"], conv)
            rows.append(row)
        tab = pd.DataFrame(rows)

        pool_id = f1(idr["y_true"], idr["y_pred"], conv)
        loeo_on_R = pd.concat([loeo[e][loeo[e]["tweet_id"].isin(R[e])] for e in events])
        loeo_on_R = loeo_on_R.set_index("tweet_id").loc[idr["tweet_id"]].reset_index()
        pool_loeo_R = f1(loeo_on_R["y_true"], loeo_on_R["y_pred"], conv)

        E1 = pool_id - tab["id_random_R"].mean()
        E2d = tab["id_random_R"] - tab["loeo_R"]
        E3 = tab["loeo_R"].mean() - tab["loeo_all"].mean()
        E0 = pool_id - tab["loeo_all"].mean()
        E4d = tab["id_temporal_T"] - tab["loeo_T"]
        res = {
            "F1pool_id_random_R": pool_id,
            "F1pool_loeo_on_R": pool_loeo_R,
            "mean_event_id_random_R": float(tab["id_random_R"].mean()),
            "mean_event_loeo_R": float(tab["loeo_R"].mean()),
            "mean_event_loeo_all": float(tab["loeo_all"].mean()),
            "mean_event_id_temporal_T": float(tab["id_temporal_T"].mean()),
            "mean_event_loeo_T": float(tab["loeo_T"].mean()),
            "E0_prior_study_gap": float(E0),
            "E1_aggregation": float(E1),
            "E2_matched_exposure": summarize(E2d, rng),
            "E3_test_subsample": float(E3),
            "E2p_pooled_matched": {"value": float(pool_id - pool_loeo_R),
                                   "instance_bootstrap_95ci": pooled_paired_bootstrap(
                                       idr, loeo_on_R, idr["event"].to_numpy(), conv, rng)},
            "E4_matched_temporal": summarize(E4d, rng),
            "decomposition_check": float(E0 - (E1 + E2d.mean() + E3)),
            "size_weighted_E2": float(np.average(E2d, weights=tab["n"])),
            "size_weighted_E4": float(np.average(E4d, weights=tab["n"])),
        }
        if "chrono_all" in tab:
            c = tab.dropna(subset=["chrono_all"])
            res["E5_future_event_leakage"] = summarize(c["loeo_sizematched_all"] - c["chrono_all"], rng)
            res["E5_future_event_leakage_on_T"] = summarize(c["loeo_sizematched_T"] - c["chrono_T"], rng)
            res["E5_events"] = c["event"].tolist()
        summary["conventions"][conv] = res
        summary.setdefault("per_event", {})[conv] = tab.round(4).to_dict(orient="records")

    out = args.out or os.path.join("experiments", "derived", "pilot_summary_" +
                                   args.runs.strip("/").replace("/", "_").replace("experiments_runs_", "") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary["conventions"], indent=1))
    print(pd.DataFrame(summary["per_event"]["all_predicted"]).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
