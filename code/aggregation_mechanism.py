"""Where the aggregation term comes from (OPUS-R1-F1, F2), from saved predictions only.

For each plan, corpus, model and seed (present-class convention P):
- E1 = F(ID predictions pooled over R) - mean_e F(ID predictions on R_e), split exactly into
    scoring     = (1/K) sum_k (F_k^pool - Fbar_k)          classes score lower inside events than in the pool
    reweighting = sum_k (1/K - w_k) Fbar_k                   the per-event mean weights class k by w_k, the pool by 1/K
  with w_k = (1/|E|) sum_{e contains k} 1/K_e and Fbar_k the w-weighted mean of F_{k,e};
- aggregation of the same LOEO predictions on R_e and on whole events (does it shrink with larger test sets?);
- Spearman correlation of per-cell LOEO F1 with the class's within-event prevalence (whole events), and of
  per-cell ID F1 on R_e with the class's prevalence in R_e (raw and within class);
- pooled ID F1 and prediction count of the class present in the fewest events, and across all runs of a corpus
  the Spearman correlation of that F1 with the reweighting term;
- per class: pooled ID F1 and prediction count on R, and pooled LOEO F1 and prediction count on the same R_e.

    python code/aggregation_mechanism.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PLANS = ("confirm", "confirm_ep4", "x3_tfidf")
CORPORA = ("humaid19", "crisislext26")


def per_class_f1(y, p, classes):
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for c in classes:
        tp, fp, fn = np.sum((y == c) & (p == c)), np.sum((y != c) & (p == c)), np.sum((y == c) & (p != c))
        out[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return out


def macro_p(frame):
    f = per_class_f1(frame.y_true, frame.y_pred, sorted(set(frame.y_true)))
    return float(np.mean(list(f.values())))


def split_e1(frame, events):
    classes = sorted(set(frame.y_true))
    K, E = len(classes), len(events)
    Fp = per_class_f1(frame.y_true, frame.y_pred, classes)
    w, wf = dict.fromkeys(classes, 0.0), dict.fromkeys(classes, 0.0)
    for e in events:
        x = frame[frame.event == e]
        ce = sorted(set(x.y_true))
        for k, f in per_class_f1(x.y_true, x.y_pred, ce).items():
            w[k] += 1 / (E * len(ce))
            wf[k] += f / (E * len(ce))
    Fbar = {k: wf[k] / w[k] for k in classes}
    scoring = sum(Fp[k] - Fbar[k] for k in classes) / K
    reweighting = sum((1 / K - w[k]) * Fbar[k] for k in classes)
    e1 = macro_p(frame) - np.mean([macro_p(frame[frame.event == e]) for e in events])
    assert abs(e1 - scoring - reweighting) < 1e-9
    return float(e1), float(scoring), float(reweighting)


def prevalence_cells(frame, events):
    """Spearman correlation of per-cell F1 with the class's share of the event, raw and within class
    (prevalence and F1 centred on each class's mean over events)."""
    rows = []
    for e in events:
        x = frame[frame.event == e]
        prev = x.y_true.value_counts(normalize=True)
        for k, f in per_class_f1(x.y_true, x.y_pred, sorted(set(x.y_true))).items():
            rows.append((k, prev[k], f))
    c = pd.DataFrame(rows, columns=["k", "prev", "f1"])
    raw = spearmanr(c.prev, c.f1)
    within = spearmanr(c.prev - c.groupby("k").prev.transform("mean"), c.f1 - c.groupby("k").f1.transform("mean"))
    return {"n": int(len(c)), "spearman": float(raw.statistic), "p": float(raw.pvalue),
            "spearman_within_class": float(within.statistic), "p_within_class": float(within.pvalue)}


def seed_record(seed_dir, man):
    events = sorted(man["events"])
    idr = pd.read_csv(os.path.join(seed_dir, "id_random", "preds.tsv.gz"), sep="\t", keep_default_na=False,
                      dtype={"tweet_id": str})
    idr = idr[idr.event.isin(events)]
    e1, sc, rw = split_e1(idr, events)
    rec = {"E1": e1, "scoring": sc, "reweighting": rw, "id_cells_R": prevalence_cells(idr, events)}
    n_ev = {k: int(idr[idr.y_true == k].event.nunique()) for k in sorted(set(idr.y_true))}
    rare = min(n_ev, key=lambda k: (n_ev[k], k))
    f_pool = per_class_f1(idr.y_true, idr.y_pred, sorted(set(idr.y_true)))
    rec["rarest_class"] = {"class": rare, "n_events": n_ev[rare], "n_events_total": len(events),
                           "pooled_id_f1": float(f_pool[rare]), "n_pred": int((idr.y_pred == rare).sum()),
                           "n_true": int((idr.y_true == rare).sum())}
    loeo = {e: pd.read_csv(os.path.join(seed_dir, f"loeo__{e}", "preds.tsv.gz"), sep="\t", keep_default_na=False,
                           dtype={"tweet_id": str}) for e in events}
    R = {e: set(map(str, man["events"][e]["R"])) for e in events}
    sub = {e: x[x.tweet_id.isin(R[e])] for e, x in loeo.items()}
    sub_all = pd.concat(sub.values())
    f_loeo = per_class_f1(sub_all.y_true, sub_all.y_pred, sorted(set(idr.y_true)))
    rec["per_class"] = {k: {"n_events": n_ev[k], "n_true": int((idr.y_true == k).sum()),
                            "id_f1": float(f_pool[k]), "id_n_pred": int((idr.y_pred == k).sum()),
                            "loeo_f1_R": float(f_loeo[k]), "loeo_n_pred_R": int((sub_all.y_pred == k).sum())}
                        for k in sorted(set(idr.y_true))}
    agg = lambda d: macro_p(pd.concat(d.values())) - np.mean([macro_p(x) for x in d.values()])
    rec["loeo_aggregation_R"], rec["loeo_aggregation_whole"] = float(agg(sub)), float(agg(loeo))
    cells = []
    for e, x in loeo.items():
        prev = x.y_true.value_counts(normalize=True)
        for k, f in per_class_f1(x.y_true, x.y_pred, sorted(set(x.y_true))).items():
            cells.append((prev[k], f, int((x.y_true == k).sum())))
    c = np.array(cells)
    rho = spearmanr(c[:, 0], c[:, 1])
    lo = c[:, 0] < 0.05
    rec["cells"] = {"n": int(len(c)), "spearman_prev_f1": float(rho.statistic), "p": float(rho.pvalue),
                    "share_prev_lt_5pct": float(lo.mean()), "median_f1_prev_lt_5pct": float(np.median(c[lo, 1])),
                    "median_f1_prev_ge_5pct": float(np.median(c[~lo, 1])),
                    "median_n_prev_lt_5pct": float(np.median(c[lo, 2]))}
    return rec


def main():
    out = []
    for plan in PLANS:
        for corpus in CORPORA:
            man = json.load(open(f"experiments/splits/confirm_{corpus}.json"))
            for model_dir in sorted(glob.glob(f"experiments/runs/{plan}/{corpus}/*")):
                seeds = []
                for sd in sorted(glob.glob(f"{model_dir}/seed*"), key=lambda p: int(p.rsplit("seed", 1)[1])):
                    have = os.path.exists(f"{sd}/id_random/run.json") and all(
                        os.path.exists(f"{sd}/loeo__{e}/run.json") for e in man["events"])
                    if have:
                        seeds.append({"seed": int(sd.rsplit("seed", 1)[1]), **seed_record(sd, man)})
                if not seeds:
                    continue
                keys = ("E1", "scoring", "reweighting", "loeo_aggregation_R", "loeo_aggregation_whole")
                summ = {k: {"mean": float(np.mean([s[k] for s in seeds])), "min": float(min(s[k] for s in seeds)),
                            "max": float(max(s[k] for s in seeds))} for k in keys}
                summ["spearman_prev_f1"] = {"mean": float(np.mean([s["cells"]["spearman_prev_f1"] for s in seeds])),
                                            "max_p": float(max(s["cells"]["p"] for s in seeds))}
                summ["id_spearman_prev_f1_R"] = {
                    "mean": float(np.mean([s["id_cells_R"]["spearman"] for s in seeds])),
                    "max_p": float(max(s["id_cells_R"]["p"] for s in seeds)),
                    "within_class_mean": float(np.mean([s["id_cells_R"]["spearman_within_class"] for s in seeds])),
                    "within_class_max_p": float(max(s["id_cells_R"]["p_within_class"] for s in seeds))}
                learned = [s for s in seeds if s["rarest_class"]["n_pred"] > 0]
                summ["rarest_class_learned_seeds"] = [s["seed"] for s in learned]
                summ["reweighting_by_rare_class"] = {
                    "learned": [float(np.mean([s["reweighting"] for s in learned]))] if learned else [],
                    "not_learned": [float(np.mean([s["reweighting"] for s in seeds if s not in learned]))]
                    if len(learned) < len(seeds) else []}
                out.append({"plan": plan, "corpus": corpus, "model": os.path.basename(model_dir),
                            "summary": summ, "seeds": seeds})
                print(plan, corpus, os.path.basename(model_dir), len(seeds), "seeds",
                      {k: round(v["mean"], 4) for k, v in summ.items() if isinstance(v, dict) and "min" in v},
                      "rare learned", summ["rarest_class_learned_seeds"], flush=True)
    rare = {}
    for corpus in CORPORA:
        runs = [(g["plan"], g["model"], s) for g in out if g["corpus"] == corpus for s in g["seeds"]]
        f1 = np.array([s["rarest_class"]["pooled_id_f1"] for _, _, s in runs])
        rw = np.array([s["reweighting"] for _, _, s in runs])
        rho = spearmanr(f1, rw)
        bands = {}
        for name, lo, hi in (("zero", 0.0, 0.0), ("low", 1e-9, 0.3), ("mid", 0.3, 0.6), ("high", 0.6, 1.0)):
            sel = (f1 >= lo) & (f1 <= hi)
            if sel.any():
                bands[name] = {"n": int(sel.sum()), "f1_min": float(f1[sel].min()), "f1_max": float(f1[sel].max()),
                               "rw_min": float(rw[sel].min()), "rw_max": float(rw[sel].max())}
        rare[corpus] = {"n_runs": len(runs), "spearman_f1_reweighting": float(rho.statistic), "p": float(rho.pvalue),
                        "bands": bands}
        print(corpus, "rare-class F1 vs reweighting over", len(runs), "runs: rho", round(rho.statistic, 3), bands)
    json.dump({"groups": out, "rare_class_vs_reweighting": rare},
              open("experiments/derived/aggregation_mechanism.json", "w"), indent=1)


if __name__ == "__main__":
    main()
