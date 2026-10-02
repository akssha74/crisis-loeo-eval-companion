"""Amended confirmatory analysis (amendment A2). code/analyze.py is kept unchanged as the as-registered
analysis and is reported alongside.

Changes relative to analyze.py:
- crossed bootstrap: each draw resamples events and one global vector of seeds (the in-distribution model of
  a seed is shared by all events), recomputing every term -- including pooled scores from per-class counts --
  so E0 = E1 + E2 + E3 holds in every draw; intervals for E0, E1, E3 and component shares;
- the full decomposition on every subset (all, protected, development);
- three macro-F1 label-set conventions: P (classes present in the evaluated labels), D (scikit-learn default,
  classes in labels or predictions) and O (fixed corpus ontology, absent classes scored 0);
- H2 as a magnitude test (MC >= margin) plus the share of events affected and of E0^D carried by MC;
- a margin grid (0.01, 0.02, 0.03) and leave-one-protected-event-out sensitivity for E2;
- exposure dose (target-event tweets in the in-distribution training sample) and E5 source-composition
  diagnostics (hazard total-variation distance) from run metadata;
- paired DUP, fuzzy DUP_F, and the event-token-masking consequence estimands on the seeds that have them;
- hard failures on any missing registered run, missing size-matched comparator or misaligned instances.

    python code/analyze_v2.py --plan confirm
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, t as tdist, wilcoxon

B = 5000
SKIP_INCOMPLETE_EXTRAS = False  # --skip_incomplete_extras: leave out optional arms (mask, full) not yet complete
RNG_SEED = 20260929
MARGINS = (0.01, 0.02, 0.03)
MATERIAL = 0.02
# metadata-only tables (tweet_id, event, hazard, label); tweet texts are never needed by the analysis
CORPORA = {"humaid19": ("experiments/meta/corpus_humaid19_meta.tsv", "experiments/splits/confirm_humaid19.json"),
           "crisislext26": ("experiments/meta/corpus_crisislext26_meta.tsv", "experiments/splits/confirm_crisislext26.json")}
PROTECTED = {"humaid19": ["california_wildfires_2018", "hurricane_dorian_2019", "hurricane_florence_2018",
                          "kerala_floods_2018", "midwestern_us_floods_2019", "pakistan_earthquake_2019"]}
CAP = 6000


def load(d, key):
    p = os.path.join(d, key.replace(":", "__"), "preds.tsv.gz")
    return pd.read_csv(p, sep="\t", dtype={"tweet_id": str}, keep_default_na=False) if os.path.exists(p) else None


def run_meta(d, key):
    p = os.path.join(d, key.replace(":", "__"), "run.json")
    return json.load(open(p)) if os.path.exists(p) else None


def counts(df, classes):
    """Per-class TP, FP, FN for one prediction frame, shape (C, 3)."""
    y, p = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    out = np.zeros((len(classes), 3))
    for j, c in enumerate(classes):
        t, q = y == c, p == c
        out[j] = [np.sum(t & q), np.sum(~t & q), np.sum(t & ~q)]
    return out


def f1_from_counts(cnt, conv):
    """Macro-F1 from per-class counts (..., C, 3) under convention P, D or O."""
    tp, fp, fn = cnt[..., 0], cnt[..., 1], cnt[..., 2]
    den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    if conv == "P":
        m = (tp + fn) > 0
    elif conv == "D":
        m = ((tp + fn) > 0) | ((tp + fp) > 0)
    else:
        m = np.ones_like(tp, dtype=bool)
    return (f * m).sum(-1) / np.maximum(m.sum(-1), 1)


def build(seed_dir, man, classes, hazard_of):
    """Per-event count arrays for one seed. Returns dict name -> (E, C, 3) arrays and metadata."""
    events = sorted(man["events"])
    R = {e: set(map(str, man["events"][e]["R"])) for e in events}
    T = {e: set(map(str, man["events"][e]["T"])) for e in events}
    need = ["id_random", "id_temporal", "id_random_dedup", "id_temporal_dedup"]
    frames = {k: load(seed_dir, k) for k in need}
    missing = [k for k, v in frames.items() if v is None]
    if missing:
        raise SystemExit(f"missing registered runs {missing} in {seed_dir}")
    opt = {k: load(seed_dir, k) for k in ("id_random_fuzzydedup", "id_random_mask")}
    if SKIP_INCOMPLETE_EXTRAS and opt["id_random_mask"] is not None and \
            any(load(seed_dir, f"loeo_mask:{e}") is None for e in man["events"]):
        opt["id_random_mask"] = None
    C = len(classes)
    E = len(events)
    arr = {k: np.full((E, C, 3), np.nan) for k in
           ["a", "b", "c", "at", "bt", "ad", "atd", "k", "n", "kT", "nT", "kk", "af", "am", "bm", "cm"]}
    stray, dose, tvd, has_chrono = np.zeros(E), np.full(E, np.nan), np.full(E, np.nan), np.zeros(E, bool)
    idmeta = run_meta(seed_dir, "id_random")
    has_mask = opt["id_random_mask"] is not None
    for i, e in enumerate(events):
        l = load(seed_dir, f"loeo:{e}")
        if l is None:
            raise SystemExit(f"missing loeo:{e} in {seed_dir}")
        a = frames["id_random"][frames["id_random"]["event"] == e]
        lr = l[l["tweet_id"].isin(R[e])].set_index("tweet_id").loc[a["tweet_id"]].reset_index()
        assert len(lr) == len(a) == len(R[e]), f"R alignment failed for {e}"
        at = frames["id_temporal"][frames["id_temporal"]["event"] == e]
        lt = l[l["tweet_id"].isin(T[e])].set_index("tweet_id").loc[at["tweet_id"]].reset_index()
        assert len(lt) == len(at) == len(T[e]), f"T alignment failed for {e}"
        arr["a"][i], arr["b"][i], arr["c"][i] = counts(a, classes), counts(lr, classes), counts(l, classes)
        arr["at"][i], arr["bt"][i] = counts(at, classes), counts(lt, classes)
        arr["ad"][i] = counts(frames["id_random_dedup"][frames["id_random_dedup"]["event"] == e], classes)
        arr["atd"][i] = counts(frames["id_temporal_dedup"][frames["id_temporal_dedup"]["event"] == e], classes)
        if opt["id_random_fuzzydedup"] is not None:
            arr["af"][i] = counts(opt["id_random_fuzzydedup"][opt["id_random_fuzzydedup"]["event"] == e], classes)
        if has_mask:
            lm = load(seed_dir, f"loeo_mask:{e}")
            if lm is None:
                raise SystemExit(f"missing loeo_mask:{e} in {seed_dir}")
            am = opt["id_random_mask"][opt["id_random_mask"]["event"] == e]
            lmr = lm[lm["tweet_id"].isin(R[e])].set_index("tweet_id").loc[am["tweet_id"]].reset_index()
            arr["am"][i], arr["bm"][i], arr["cm"][i] = counts(am, classes), counts(lmr, classes), counts(lm, classes)
        stray[i] = (~l["y_pred"].isin(set(l["y_true"]))).sum()
        dose[i] = idmeta["train_event_counts"].get(e, 0)
        k = load(seed_dir, f"chrono:{e}")
        if k is not None:
            km = run_meta(seed_dir, f"chrono:{e}")
            if km["n_train"] < CAP:
                ref, refm = load(seed_dir, f"loeon:{e}"), run_meta(seed_dir, f"loeon:{e}")
                if ref is None:
                    raise SystemExit(f"chrono:{e} trained on {km['n_train']} < {CAP} but loeon:{e} is missing")
            else:
                ref, refm = l, run_meta(seed_dir, f"loeo:{e}")
            assert refm["n_train"] == km["n_train"], f"size mismatch chrono/loeo for {e}"
            has_chrono[i] = True
            arr["k"][i], arr["n"][i] = counts(k, classes), counts(ref, classes)
            arr["kT"][i] = counts(k[k["tweet_id"].isin(T[e])], classes)
            lk = load(seed_dir, f"loeok:{e}")
            if lk is not None:
                assert run_meta(seed_dir, f"loeok:{e}")["n_train"] == km["n_train"], f"size mismatch loeok for {e}"
                arr["kk"][i] = counts(lk, classes)
            arr["nT"][i] = counts(ref[ref["tweet_id"].isin(T[e])], classes)
            hz = lambda m: pd.Series({hazard_of[ev]: n for ev, n in m["train_event_counts"].items()}).groupby(level=0).sum()
            h1, h2 = hz(km), hz(refm)
            allh = sorted(set(h1.index) | set(h2.index))
            p1 = h1.reindex(allh, fill_value=0) / h1.sum()
            p2 = h2.reindex(allh, fill_value=0) / h2.sum()
            tvd[i] = 0.5 * float(np.abs(p1 - p2).sum())
    pooled_classes_present = bool((arr["a"][..., 0] + arr["a"][..., 2]).sum(0).min() > 0)
    return arr, {"stray": stray, "dose": dose, "tvd": tvd, "has_chrono": has_chrono,
                 "has_fuzzy": opt["id_random_fuzzydedup"] is not None, "has_mask": has_mask,
                 "pooled_classes_present": pooled_classes_present}


def estimands(A, ev, sd, conv):
    """Estimands for event index multiset ev and seed index multiset sd. A[name]: (E, S, C, 3)."""
    f = lambda k: f1_from_counts(A[k][np.ix_(ev, sd)], conv)          # (e, s)
    pool = lambda k: f1_from_counts(np.nansum(A[k][np.ix_(ev, sd)], axis=0), conv)  # (s,)
    a, b, c = f("a"), f("b"), f("c")
    out = {"E1_loeo": np.mean(pool("b") - b.mean(0)),
           "E0": np.mean(pool("a") - c.mean(0)), "E1": np.mean(pool("a") - a.mean(0)),
           "E2": np.mean(a - b), "E3": np.mean(b - c), "E2p": np.mean(pool("a") - pool("b")),
           "E4": np.mean(f("at") - f("bt")), "DUP": np.mean(a - f("ad")), "DUPT": np.mean(f("at") - f("atd")),
           "MC": np.mean(f1_from_counts(A["c"][np.ix_(ev, sd)], "P") - f1_from_counts(A["c"][np.ix_(ev, sd)], "D")),
           "mean_a": np.mean(a), "mean_c": np.mean(c), "pooled_id": np.mean(pool("a"))}
    out["I_order"] = out["E2p"] - out["E2"]
    out["E1_sym"], out["E2_sym"] = (out["E1"] + out["E1_loeo"]) / 2, (out["E2"] + out["E2p"]) / 2
    kk = A["k"][np.ix_(ev, sd)]
    elig = ~np.isnan(kk[..., 0, 0])
    kd = A["kk"][np.ix_(ev, sd)]
    eligk = elig & ~np.isnan(kd[..., 0, 0])
    if eligk.any():
        fk, fkk, fn = f1_from_counts(kk, conv), f1_from_counts(kd, conv), f1_from_counts(A["n"][np.ix_(ev, sd)], conv)
        out["E5_div"] = np.mean((fn - fkk)[eligk])
        out["E5_time"] = np.mean((fkk - fk)[eligk])
    if elig.any():
        out["E5"] = np.mean((f1_from_counts(A["n"][np.ix_(ev, sd)], conv) - f1_from_counts(kk, conv))[elig])
        out["E5_T"] = np.mean((f1_from_counts(A["nT"][np.ix_(ev, sd)], conv)
                               - f1_from_counts(A["kT"][np.ix_(ev, sd)], conv))[elig])
    return out


def seed_subset_estimands(A, ev, sd, conv, keys):
    """Estimands that exist only for some seeds (fuzzy, mask)."""
    out = {}
    f = lambda k: f1_from_counts(A[k][np.ix_(ev, sd)], conv)
    pool = lambda k: f1_from_counts(np.nansum(A[k][np.ix_(ev, sd)], axis=0), conv)
    if "fuzzy" in keys:
        out["DUP_F"] = np.mean(f("a") - f("af"))
    if "mask" in keys:
        a, b, c, am, bm, cm = f("a"), f("b"), f("c"), f("am"), f("bm"), f("cm")
        out["E0_base"] = np.mean(pool("a") - c.mean(0))
        out["E0_mask"] = np.mean(pool("am") - cm.mean(0))
        out["E2_base"], out["E2_mask"] = np.mean(a - b), np.mean(am - bm)
        out["LOEO_benefit_whole"] = np.mean(cm - c)
        out["LOEO_benefit_R"] = np.mean(bm - b)
        out["ID_benefit_R"] = np.mean(am - a)
        out["dE0"] = out["E0_mask"] - out["E0_base"]
        out["dE2"] = out["E2_mask"] - out["E2_base"]
    return out


def t_interval(x, level):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return [float("nan"), float("nan")]
    h = tdist.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean() - h), float(x.mean() + h)]


def event_contrib(A, evs, conv, k1, k2, mask=None):
    """Seed-averaged per-event contribution F(k1) - F(k2) (NaN rows skipped)."""
    x = f1_from_counts(A[k1][evs], conv) - f1_from_counts(A[k2][evs], conv)
    if mask is not None:
        x = np.where(mask, x, np.nan)
    return np.nanmean(x, 1)


def crossed_boot(fn, n_e, n_s, rng, b=B):
    draws = []
    for _ in range(b):
        ev = rng.integers(0, n_e, n_e)
        sd = rng.integers(0, n_s, n_s)
        draws.append(fn(ev, sd))
    return draws


def summarise(point, draws, ev_contrib=None):
    res = {}
    for k, v in point.items():
        xs = np.array([d[k] for d in draws if k in d], float)
        xs = xs[~np.isnan(xs)]
        r = {"mean": float(v)}
        if len(xs):
            r.update({"ci95": [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))],
                      "ci90": [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]})
            r["material_positive"] = {str(m): bool(r["ci95"][0] > 0 and v >= m) for m in MARGINS}
            r["equivalent_within"] = {str(m): bool(r["ci90"][0] > -m and r["ci90"][1] < m) for m in MARGINS}
        res[k] = r
    return res


def analyse(corpus, model_dir):
    cf, mf = CORPORA[corpus]
    man = json.load(open(mf))
    meta_df = pd.read_csv(cf, sep="\t", keep_default_na=False, usecols=["event", "hazard", "label"])
    classes = sorted(meta_df["label"].unique())
    hazard_of = meta_df.groupby("event")["hazard"].first().to_dict()
    events = sorted(man["events"])
    n_total = sum(v["n"] for v in man["events"].values())
    man_share = {e: man["events"][e]["chrono_pool_n"] / (n_total - man["events"][e]["n"]) for e in events}
    seed_dirs = sorted(glob.glob(os.path.join(model_dir, "seed*")), key=lambda p: int(p.rsplit("seed", 1)[1]))
    built = [build(sd, man, classes, hazard_of) for sd in seed_dirs]
    names = built[0][0].keys()
    A = {k: np.stack([bd[0][k] for bd in built], axis=1) for k in names}      # (E, S, C, 3)
    M = {k: np.stack([bd[1][k] for bd in built], axis=1) for k in ("stray", "dose", "tvd", "has_chrono")}
    assert all(bd[1]["pooled_classes_present"] for bd in built), "a class is absent from pooled R; MC != E0^D - E0^P"
    fuzzy_s = [j for j, bd in enumerate(built) if bd[1]["has_fuzzy"]]
    mask_s = [j for j, bd in enumerate(built) if bd[1]["has_mask"]]
    subsets = {"all": list(range(len(events)))}
    if corpus in PROTECTED:
        subsets["protected"] = [events.index(e) for e in PROTECTED[corpus]]
        subsets["development"] = [i for i, e in enumerate(events) if e not in PROTECTED[corpus]]
    res = {"corpus": corpus, "model": os.path.basename(model_dir), "seeds": [os.path.basename(s)[4:] for s in seed_dirs],
           "fuzzy_seeds": [os.path.basename(seed_dirs[j])[4:] for j in fuzzy_s],
           "mask_seeds": [os.path.basename(seed_dirs[j])[4:] for j in mask_s], "subsets": {}}
    S = len(seed_dirs)
    for name, evs in subsets.items():
        evs = np.array(evs)
        block = {}
        paired = {}
        for conv in ("P", "D", "O"):
            rng = np.random.default_rng(RNG_SEED)
            point = estimands(A, evs, np.arange(S), conv)
            draws = crossed_boot(lambda e, s: estimands(A, evs[e], s, conv), len(evs), S, rng)
            paired[conv] = (point, draws)
            for d in draws + [point]:
                d["share_E1"], d["share_E2"], d["share_E3"] = (d["E1"] / d["E0"], d["E2"] / d["E0"], d["E3"] / d["E0"]) \
                    if abs(d["E0"]) > 1e-9 else (np.nan, np.nan, np.nan)
            summ = summarise(point, draws)
            summ["decomposition_residual_max"] = float(max(abs(d["E0"] - d["E1"] - d["E2"] - d["E3"]) for d in draws + [point]))
            # events-only bootstrap of seed-averaged contributions (sensitivity)
            e2_ev = (f1_from_counts(A["a"][evs], conv) - f1_from_counts(A["b"][evs], conv)).mean(1)
            rng2 = np.random.default_rng(RNG_SEED + 1)
            bs = e2_ev[rng2.integers(0, len(evs), (B, len(evs)))].mean(1)
            summ["E2_events_only_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            summ["E2_events_positive"] = int((e2_ev > 0).sum())
            summ["E2_n_events"] = int(len(evs))
            if len(evs) >= 5 and np.any(e2_ev != 0):
                summ["E2_wilcoxon_p"] = float(wilcoxon(e2_ev).pvalue)
            # registered primary rules (A2): t-intervals on seed-averaged event contributions
            tt = {}
            for est, (k1, k2) in {"E2": ("a", "b"), "E3": ("b", "c"), "E4": ("at", "bt"), "DUP": ("a", "ad")}.items():
                x = event_contrib(A, evs, conv, k1, k2)
                tt[est] = {"mean": float(np.nanmean(x)), "t95": t_interval(x, 0.95), "t90": t_interval(x, 0.90), "n_events": int(np.sum(~np.isnan(x)))}
            mcx = np.nanmean(f1_from_counts(A["c"][evs], "P") - f1_from_counts(A["c"][evs], "D"), 1)
            tt["MC"] = {"mean": float(mcx.mean()), "t95": t_interval(mcx, 0.95), "t90": t_interval(mcx, 0.90), "n_events": int(len(mcx))}
            share = 1 - np.array([man_share[events[i]] for i in evs])
            elig5 = ~np.isnan(A["k"][evs][:, 0, 0, 0])
            for lab, sel in {"E5": elig5, "E5_futureshare_ge_0.25": elig5 & (share >= 0.25), "E5_placebo_lt_0.05": elig5 & (share < 0.05)}.items():
                x = event_contrib(A, evs[sel], conv, "n", "k") if sel.any() else np.array([])
                tt[lab] = {"mean": float(np.nanmean(x)) if len(x) else None, "t95": t_interval(x, 0.95), "t90": t_interval(x, 0.90),
                           "n_events": int(len(x)), "tost_within": {str(m): bool(len(x) > 1 and t_interval(x, 0.90)[0] > -m and t_interval(x, 0.90)[1] < m) for m in MARGINS}}
            eligk = elig5 & ~np.isnan(A["kk"][evs][:, 0, 0, 0])
            if eligk.any():
                tt["E5_div"] = {"mean": float(np.nanmean(event_contrib(A, evs[eligk], conv, "n", "kk"))), "t95": t_interval(event_contrib(A, evs[eligk], conv, "n", "kk"), 0.95)}
                tt["E5_time"] = {"mean": float(np.nanmean(event_contrib(A, evs[eligk], conv, "kk", "k"))), "t95": t_interval(event_contrib(A, evs[eligk], conv, "kk", "k"), 0.95)}
            for est, d in tt.items():
                if d.get("t95") and d["mean"] is not None and not np.isnan(d["t95"][0]):
                    d["material_positive"] = {str(m): bool(d["t95"][0] > 0 and d["mean"] >= m) for m in MARGINS}
            summ["t_rules"] = tt
            ev_names = [events[i] for i in evs]
            summ["event_contrib"] = {
                "E2": dict(zip(ev_names, map(float, event_contrib(A, evs, conv, "a", "b")))),
                "MC": dict(zip(ev_names, map(float, mcx))),
                "E5": {events[i]: float(v) for i, v in zip(evs[elig5], event_contrib(A, evs[elig5], conv, "n", "k"))} if elig5.any() else {},
                "future_share": dict(zip(ev_names, map(float, share)))}
            if conv == "P":
                w = np.array([A["c"][i, 0, :, [0, 2]].sum() for i in evs], float)
                summ["E2_tweet_weighted"] = float(np.average(event_contrib(A, evs, conv, "a", "b"), weights=w))
                summ["E4_tweet_weighted"] = float(np.average(event_contrib(A, evs, conv, "at", "bt"), weights=w))
            seed_vals = [estimands(A, evs, np.array([j]), conv) for j in range(S)]
            summ["seed_sd"] = {k: float(np.std([v[k] for v in seed_vals], ddof=1)) for k in ("E0", "E1", "E2", "E3", "MC")} if S > 1 else {}
            summ["seed_values"] = [{"seed": res["seeds"][j], **{k: float(v[k]) for k in ("E0", "E1", "E2", "E3", "E4", "DUP", "MC", "E2p",
                                                                                   "mean_a", "mean_c", "pooled_id")},
                                    **({"E5": float(v["E5"])} if "E5" in v else {})} for j, v in enumerate(seed_vals)]
            if name == "protected":
                summ["E2_leave_one_event_out"] = {events[evs[i]]: float(np.delete(e2_ev, i).mean()) for i in range(len(evs))}
            if fuzzy_s:
                sub = np.array(fuzzy_s)
                p2 = seed_subset_estimands(A, evs, sub, conv, {"fuzzy"})
                d2 = crossed_boot(lambda e, s: seed_subset_estimands(A, evs[e], sub[s], conv, {"fuzzy"}), len(evs), len(sub), rng, 2000)
                summ.update(summarise(p2, d2))
            if mask_s:
                sub = np.array(mask_s)
                p3 = seed_subset_estimands(A, evs, sub, conv, {"mask"})
                d3 = crossed_boot(lambda e, s: seed_subset_estimands(A, evs[e], sub[s], conv, {"mask"}), len(evs), len(sub), rng, 2000)
                summ.update({"mask_" + k: v for k, v in summarise(p3, d3).items()})
                summ["mask_seed_values"] = [
                    {"seed": res["seeds"][j], **{k: float(v) for k, v in
                                                 seed_subset_estimands(A, evs, np.array([j]), conv, {"mask"}).items()}}
                    for j in mask_s]
            block[conv] = summ
        # every convention restarts the generator at RNG_SEED, so draw i resamples the same events and seeds in P and D
        (pP, dP), (pD, dD) = paired["P"], paired["D"]
        cross = {"share_E2": lambda p, d: p["E2"] / d["E0"], "share_E2p": lambda p, d: p["E2p"] / d["E0"],
                 "share_E2_sym": lambda p, d: p["E2_sym"] / d["E0"], "share_E1": lambda p, d: p["E1"] / d["E0"],
                 "share_MC": lambda p, d: p["MC"] / d["E0"], "remainder": lambda p, d: d["E0"] - p["E2"],
                 "remainder_p": lambda p, d: d["E0"] - p["E2p"]}
        block["headline"] = {}
        for lab, fn in cross.items():
            xs = np.array([fn(x, y) for x, y in zip(dP, dD)], float)
            block["headline"][lab] = {"mean": float(fn(pP, pD)), "ci95": [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]}
        four = block["P"]["E1"]["mean"] + block["P"]["E2"]["mean"] + block["P"]["E3"]["mean"] + block["P"]["MC"]["mean"]
        block["four_term_residual"] = float(block["D"]["E0"]["mean"] - four)
        # consequence: per-event difficulty rankings of LOEO under D vs P (seed means)
        cD = f1_from_counts(A["c"][evs], "D").mean(1)
        cP = f1_from_counts(A["c"][evs], "P").mean(1)
        tau = kendalltau(cD, cP)
        rD, rP = np.argsort(np.argsort(cD)), np.argsort(np.argsort(cP))
        block["ranking_D_vs_P"] = {"kendall_tau": float(tau.correlation), "p": float(tau.pvalue),
                                   "events_moving_ge3_ranks": int((np.abs(rD - rP) >= 3).sum()),
                                   "hardest_D": events[evs[int(np.argmin(cD))]], "hardest_P": events[evs[int(np.argmin(cP))]]}
        hz = pd.Series([hazard_of[events[i]] for i in evs])
        hD, hP = pd.Series(cD).groupby(hz).mean(), pd.Series(cP).groupby(hz).mean()
        block["hazard_ranking_D_vs_P"] = {"hardest_D": str(hD.idxmin()), "hardest_P": str(hP.idxmin()),
                                          "mean_D": hD.round(4).to_dict(), "mean_P": hP.round(4).to_dict()}
        block["future_share"] = {events[i]: round(1 - man_share[events[i]], 3) for i in evs}
        # H2 magnitude and prevalence (per-event MC, seed mean)
        mc_ev = (f1_from_counts(A["c"][evs], "P") - f1_from_counts(A["c"][evs], "D")).mean(1)
        block["MC_events_affected"] = int((mc_ev > 1e-12).sum())
        block["MC_share_of_E0D"] = float(block["D"]["MC"]["mean"] / block["D"]["E0"]["mean"]) if abs(block["D"]["E0"]["mean"]) > 1e-9 else None
        # exposure dose and E5 composition diagnostics (convention P)
        e2_ev_p = (f1_from_counts(A["a"][evs], "P") - f1_from_counts(A["b"][evs], "P")).mean(1)
        dose = np.nanmean(M["dose"][evs], 1)
        block["dose_target_tweets_mean"] = {events[i]: float(dose[j]) for j, i in enumerate(evs)}
        rho = spearmanr(dose, e2_ev_p)
        block["dose_vs_E2_spearman"] = {"rho": float(rho.correlation), "p": float(rho.pvalue)}
        ch = M["has_chrono"][evs].all(1)
        if ch.sum() >= 5:
            e5_ev = (f1_from_counts(A["n"][evs][ch], "P") - f1_from_counts(A["k"][evs][ch], "P")).mean(1)
            tv = np.nanmean(M["tvd"][evs][ch], 1)
            r5 = spearmanr(tv, e5_ev)
            block["E5_hazard_tvd_mean"] = float(np.mean(tv))
            block["E5_vs_tvd_spearman"] = {"rho": float(r5.correlation), "p": float(r5.pvalue)}
        block["stray_preds_per_event_mean"] = float(np.mean(M["stray"][evs]))
        res["subsets"][name] = block
    per_event = []
    for i, e in enumerate(events):
        row = {"event": e, "protected": e in PROTECTED.get(corpus, []), "stray_preds": float(np.mean(M["stray"][i])),
               "dose": float(np.nanmean(M["dose"][i]))}
        for conv in ("P", "D"):
            for k in ("a", "b", "c"):
                row[f"{k}_{conv}"] = float(np.mean(f1_from_counts(A[k][i], conv)))
        row["n_true_classes"] = int(((A["c"][i, 0, :, 0] + A["c"][i, 0, :, 2]) > 0).sum())
        row["n"] = int((A["c"][i, 0, :, 0] + A["c"][i, 0, :, 2]).sum())
        per_event.append(row)
    res["per_event_seed_mean"] = per_event
    return res


def analyse_full(corpus, model_dir):
    """Full-pool arm (A2): id_random_full vs loeo_full:e at equal size, seeds that have it."""
    cf, mf = CORPORA[corpus]
    man = json.load(open(mf))
    classes = sorted(pd.read_csv(cf, sep="\t", keep_default_na=False, usecols=["label"])["label"].unique())
    events = sorted(man["events"])
    R = {e: set(map(str, man["events"][e]["R"])) for e in events}
    out = []
    for sd in sorted(glob.glob(os.path.join(model_dir, "seed*"))):
        idf = load(sd, "id_random_full")
        if idf is None:
            continue
        a, b, c = [], [], []
        for e in events:
            l = load(sd, f"loeo_full:{e}")
            if l is None:
                if SKIP_INCOMPLETE_EXTRAS:
                    a = None
                    break
                raise SystemExit(f"missing loeo_full:{e} in {sd}")
            ae = idf[idf["event"] == e]
            lr = l[l["tweet_id"].isin(R[e])].set_index("tweet_id").loc[ae["tweet_id"]].reset_index()
            assert len(lr) == len(ae) == len(R[e])
            a.append(counts(ae, classes)); b.append(counts(lr, classes)); c.append(counts(l, classes))
        if a is None:
            continue
        A = {"a": np.array(a)[:, None], "b": np.array(b)[:, None], "c": np.array(c)[:, None]}
        rec = {"seed": os.path.basename(sd)[4:], "n_train": run_meta(sd, "id_random_full")["n_train"]}
        for conv in ("P", "D"):
            f = lambda k: f1_from_counts(A[k], conv)[:, 0]
            pool = f1_from_counts(A["a"].sum(0), conv)[0]
            e2 = f("a") - f("b")
            rec[conv] = {"E0": float(pool - f("c").mean()), "E1": float(pool - f("a").mean()), "E2": float(e2.mean()),
                         "E3": float((f("b") - f("c")).mean()), "E2_t95": t_interval(e2, 0.95),
                         "pooled_id": float(pool), "mean_loeo_whole": float(f("c").mean())}
        mc = f1_from_counts(A["c"], "P")[:, 0] - f1_from_counts(A["c"], "D")[:, 0]
        rec["MC"] = float(mc.mean())
        rec["MC_t95"] = t_interval(mc, 0.95)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="confirm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip_incomplete_extras", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated model directory names to analyse (default: all)")
    ap.add_argument("--corpora", default="", help="comma-separated corpora to analyse (default: all)")
    args = ap.parse_args()
    global SKIP_INCOMPLETE_EXTRAS
    SKIP_INCOMPLETE_EXTRAS = args.skip_incomplete_extras
    results = []
    for corpus in CORPORA:
        if args.corpora and corpus not in args.corpora.split(","):
            continue
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", args.plan, corpus, "*"))):
            if args.models and os.path.basename(model_dir) not in args.models.split(","):
                continue
            print("analysing", model_dir, flush=True)
            res = analyse(corpus, model_dir)
            res["full_pool"] = analyse_full(corpus, model_dir)
            results.append(res)
    out = args.out or os.path.join("experiments", "derived", f"summary_v2_{args.plan}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    for r in results:
        for sub, blk in r["subsets"].items():
            p = blk["P"]
            print(r["corpus"], r["model"], sub, {k: round(p[k]["mean"], 4) for k in ("E0", "E1", "E2", "E3", "MC") if k in p},
                  "E0D", round(blk["D"]["E0"]["mean"], 4))
    print("wrote", out)


if __name__ == "__main__":
    main()
