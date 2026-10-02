"""Exploratory X12 analyses of the whole-event LOEO predictions of X11 (registered in research/preregistration.md
before any of them was computed). No model is trained.

(a) paired bootstrap of between-system differences in A, with events resampled once for all systems;
(b) bias of the percentile intervals of A^P and BCa intervals from the X11 draws;
(c) within-class Spearman correlation of cell support with per-cell precision and with per-cell recall;
(d) A^P with within-event duplicate texts collapsed to one message;
(e) event-class cells with at least 50 messages and seed-mean F1 below 0.05, with their training-sample counts;
(f) per-event macro-F1 of the runs of seeds left out because they lack some held-out events (X12 addendum).

    python code/revision_x12.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
from revision_x7 import B, RNG_SEED  # noqa: E402
import companion_eval as ce  # noqa: E402
from revision_x9 import complete_seeds, count_array, frames, load_meta  # noqa: E402
from revision_x11 import CONVS, a_conv, boot, ci, jackknife, load_system, systems  # noqa: E402
from verify_training_ids import CORPORA as ID_CORPORA, run_rng  # noqa: E402

OUT = "experiments/derived/revision_x12.json"
ZERO_MIN_SUPPORT, ZERO_MAX_F1 = 50, 0.05


def name(arm, model):
    return f"{arm}|{model}"


def within_dup_path(corpus):
    return os.path.join("experiments", "meta", f"within_event_duplicates_{corpus}.tsv")


def within_duplicates(corpus, meta):
    """Messages dropped when one message (smallest tweet identifier) of each normalised text is kept per event."""
    src = os.path.join("data", f"corpus_{corpus}.tsv")
    if os.path.exists(src):
        df = pd.read_csv(src, sep="\t", dtype={"tweet_id": str}, keep_default_na=False,
                         usecols=["tweet_id", "event", "text_norm"])
        df = df.assign(n=df["tweet_id"].astype("int64")).sort_values(["event", "text_norm", "n"])
        dropped = df[df.duplicated(["event", "text_norm"], keep="first")][["tweet_id", "event"]]
        dropped.sort_values(["event", "tweet_id"]).to_csv(within_dup_path(corpus), sep="\t", index=False)
    dropped = pd.read_csv(within_dup_path(corpus), sep="\t", dtype=str, keep_default_na=False)
    assert dropped["tweet_id"].isin(meta["tweet_id"]).all()
    return set(dropped["tweet_id"])


def pool_per(c, cv):
    """Seed-mean pooled and per-event macro-F1 for counts (E, S, C, 3)."""
    return float(av.f1_from_counts(c.sum(0), cv).mean()), float(av.f1_from_counts(c, cv).mean())


def paired(arrays):
    """Point values and bootstrap draws of pooled and per-event scores; events drawn once per draw for all."""
    names = list(arrays)
    point = {k: {cv: pool_per(arrays[k], cv) for cv in CONVS} for k in names}
    rng = np.random.default_rng(RNG_SEED)
    E = next(iter(arrays.values())).shape[0]
    draws = {k: {cv: [] for cv in CONVS} for k in names}
    for _ in range(B):
        ev = rng.integers(0, E, E)
        for k in names:
            S = arrays[k].shape[1]
            x = arrays[k][np.ix_(ev, rng.integers(0, S, S))]
            for cv in CONVS:
                draws[k][cv].append(pool_per(x, cv))
    return point, {k: {cv: np.array(v) for cv, v in d.items()} for k, d in draws.items()}


def pair_table(point, draws):
    names, out = list(point), {}
    for cv in CONVS:
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                gp = point[a][cv][0] - point[b][cv][0]
                ge = point[a][cv][1] - point[b][cv][1]
                dp = draws[a][cv][:, 0] - draws[b][cv][:, 0]
                de = draws[a][cv][:, 1] - draws[b][cv][:, 1]
                da = dp - de
                rows.append({"i": a, "j": b, "gap_pooled": gp, "gap_per_event": ge, "dA": gp - ge,
                             "dA_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                             "share_draws_orders_disagree": float(np.mean(np.sign(dp) != np.sign(de))),
                             "per_event_gap_below_abs_dA": bool(abs(ge) < abs(gp - ge)),
                             "orders_disagree": bool(np.sign(gp) != np.sign(ge))})
        out[cv] = {"pairs": rows, "n_pairs": len(rows),
                   "n_per_event_gap_below_abs_dA": sum(r["per_event_gap_below_abs_dA"] for r in rows),
                   "n_orders_disagree": sum(r["orders_disagree"] for r in rows)}
    return out


def bca(draws, theta, jack):
    share = np.mean(draws < theta) + 0.5 * np.mean(draws == theta)
    z0 = float(norm.ppf(np.clip(share, 1 / (len(draws) + 1), len(draws) / (len(draws) + 1))))
    d = jack.mean() - jack
    acc = float((d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)) if (d ** 2).sum() > 0 else 0.0
    q = []
    for alpha in (0.025, 0.975):
        z = norm.ppf(alpha)
        q.append(float(np.percentile(draws, 100 * norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))))
    return {"z0": z0, "acceleration": acc, "ci95": q}


def precision_recall_rho(c, classes):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nanmean(np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), np.nan), axis=1)
        rec = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), np.nan).mean(1)
    sup = tp[:, 0] + fn[:, 0]
    per_class = {}
    for j, k in enumerate(classes):
        row = {}
        for lab, v in (("precision", prec[:, j]), ("recall", rec[:, j])):
            m = (sup[:, j] > 0) & ~np.isnan(v)
            ok = m.sum() >= 3 and np.ptp(v[m]) > 0 and np.ptp(sup[m, j]) > 0
            row[lab] = float(spearmanr(sup[m, j], v[m]).statistic) if ok else None
            row[f"n_cells_{lab}"] = int(m.sum())
        per_class[k] = row
    med = {lab: [v[lab] for v in per_class.values() if v[lab] is not None] for lab in ("precision", "recall")}
    return {"per_class": per_class, **{f"median_rho_{lab}": float(np.median(v)) for lab, v in med.items()},
            **{f"n_classes_defined_{lab}": len(v) for lab, v in med.items()}}


def zero_cells(corpus, model_dir, seeds, events, classes, c, fr):
    meta_path, _ = ID_CORPORA[corpus]
    meta = pd.read_csv(meta_path, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    ids, ev_arr = meta["tweet_id"].to_numpy(), meta["event"].to_numpy()
    lab = dict(zip(meta["tweet_id"], meta["label"]))
    sup = c[:, 0, :, 0] + c[:, 0, :, 2]
    f = np.where(2 * c[..., 0] + c[..., 1] + c[..., 2] > 0,
                 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1), 0.0).mean(1)
    out = []
    for i, e in enumerate(events):
        for j, k in enumerate(classes):
            if sup[i, j] < ZERO_MIN_SUPPORT or f[i, j] >= ZERO_MAX_F1:
                continue
            preds = pd.concat([fr[s, e].loc[fr[s, e]["y_true"] == k, "y_pred"] for s in seeds])
            top = preds.value_counts()
            n_train = []
            for s in seeds:
                rec = json.load(open(os.path.join(model_dir, f"seed{s}", f"loeo__{e}", "run.json")))
                pool = ids[ev_arr != e]
                train = run_rng(rec["run_key"], rec["seed"]).choice(pool, size=min(rec["cap"], len(pool)),
                                                                    replace=False)
                n_train.append(sum(lab[t] == k for t in train))
            out.append({"event": e, "class": k, "support": int(sup[i, j]), "f1": float(f[i, j]),
                        "top_prediction": str(top.index[0]), "top_prediction_share": float(top.iloc[0] / top.sum()),
                        "train_sample_n_mean": float(np.mean(n_train)),
                        "train_events_n": int(((meta["label"] == k) & (meta["event"] != e)).sum())})
    return out


def incomplete_seed_runs(metas):
    out = []
    for arm, corpus, model, model_dir, prefix, seeds in systems():
        if prefix is None:
            continue
        events = sorted(json.load(open(av.CORPORA[corpus][1]))["events"])
        classes = sorted(metas[corpus]["label"].unique())
        s_reg, skipped = complete_seeds(model_dir, events, prefix, seeds)
        for s in (k["seed"] for k in skipped):
            have = [e for e in events
                    if os.path.exists(os.path.join(model_dir, f"seed{s}", f"{prefix}__{e}", "preds.tsv.gz"))]
            fr, ref = frames(model_dir, [s], have, prefix), frames(model_dir, s_reg, have, prefix)
            for e in have:
                f_ref = [ce.macro_f1(ce.counts(ref[r, e], classes), "P") for r in s_reg]
                out.append({"arm": arm, "corpus": corpus, "model": model, "seed": s, "event": e,
                            "f1_P": ce.macro_f1(ce.counts(fr[s, e], classes), "P"), "complete_seeds": s_reg,
                            "complete_mean": float(np.mean(f_ref)), "complete_min": float(min(f_ref)),
                            "complete_max": float(max(f_ref))})
    return out


def main():
    out = {"pairs": {}, "bias": [], "precision_recall": [], "within_event_duplicates": {}, "zero_cells": []}
    metas = {k: load_meta(k) for k in av.CORPORA}
    wdup = {k: within_duplicates(k, metas[k]) for k in av.CORPORA}
    arrays = {k: {} for k in av.CORPORA}
    for arm, corpus, model, model_dir, prefix, seeds in systems():
        meta = metas[corpus]
        classes = sorted(meta["label"].unique())
        events = sorted(json.load(open(av.CORPORA[corpus][1]))["events"])
        s_reg, fr = load_system(corpus, model_dir, prefix, seeds, events)
        c = count_array(fr, s_reg, events, classes)
        E, S = len(events), len(s_reg)
        arrays[corpus][name(arm, model)] = c
        theta = a_conv(c, "P")
        d = boot(lambda e, s: a_conv(c[np.ix_(e, s)], "P"), E, S)[:, 0]
        jack = np.array(list(jackknife(c, events, theta)["values"].values()))
        out["bias"].append({"system": name(arm, model), "corpus": corpus, "A_P": theta,
                            "boot_mean": float(d.mean()), "boot_median": float(np.median(d)),
                            "bias": float(d.mean() - theta), "percentile_ci95": ci(d[:, None])[0],
                            "bca": bca(d, theta, jack)})
        cw = count_array(fr, s_reg, events, classes, keep=set(meta["tweet_id"]) - wdup[corpus])
        dw = boot(lambda e, s: a_conv(cw[np.ix_(e, s)], "P"), E, S)
        out["within_event_duplicates"].setdefault(corpus, {"dropped": len(wdup[corpus]), "systems": []})
        out["within_event_duplicates"][corpus]["systems"].append(
            {"system": name(arm, model), "A_P": a_conv(cw, "P"), "ci95": ci(dw)[0], "A_P_all": theta})
        if arm == "2 epochs":
            out["precision_recall"].append({"corpus": corpus, "model": model,
                                            **precision_recall_rho(c, classes)})
            for z in zero_cells(corpus, model_dir, s_reg, events, classes, c, fr):
                out["zero_cells"].append({"corpus": corpus, "model": model, **z})
        print(arm, corpus, model, round(theta, 4), "bias", round(out["bias"][-1]["bias"], 4),
              "within-dup", round(a_conv(cw, "P"), 4), flush=True)
    for corpus in av.CORPORA:
        point, draws = paired(arrays[corpus])
        out["pairs"][corpus] = pair_table(point, draws)
    out["incomplete_seed_runs"] = incomplete_seed_runs(metas)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
