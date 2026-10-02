"""Registered secondary diagnostics (amendment A2), computed from saved predictions and reconstructed
training samples; no additional training.

1. Oracle label-prior matching (upper bound, uses target labels): each model's saved probabilities are
   reweighted by q_true / q_model, the true class distribution of the scored set over the model's mean
   predicted distribution on it, in one step; E2 is recomputed on R_e (convention P). The share of E2 that
   disappears bounds how much of it a label-prior correction could explain. (EM re-estimation, Saerens et al.
   2002, was tried first and diverged: after two epochs the models assign every class >= 0.3% probability,
   so EM inflates rare classes without source-side calibration.)
2. Similarity strata: each R tweet's maximum char 3-5-gram TF-IDF cosine to its seed's id_random training
   sample; pooled present-class macro-F1 of ID and LOEO predictions within strata [0, .5), [.5, .8), [.8, 1].
3. Rare-class sensitivity: per-event macro-F1 over classes with >= 5 instances in R_e (same set for both).
4. Stray classes on matched instances: ID vs LOEO on R_e, and LOEO on R_e vs the whole event.
5. MC by the number of classes absent from the event.

    python code/diagnostics.py --plan confirm
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from run_protocol import CORPORA, build_plan, run_rng

STRATA = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]


def prior_match(frame, classes, eps=1e-4):
    """One-step oracle reweighting of a prediction frame to its true class distribution."""
    P = frame[[f"p_{c}" for c in classes]].to_numpy(float)
    q_true = frame["y_true"].value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy(float)
    w = P * (q_true + eps) / (P.mean(0) + eps)
    return pd.Series(np.array(classes)[w.argmax(1)], index=frame.index)


def f1p(y, p):
    return f1_score(y, p, average="macro", labels=sorted(set(y)), zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="confirm")
    ap.add_argument("--out", default="experiments/derived/diagnostics_confirm.json")
    args = ap.parse_args()
    out = []
    for corpus, (cf, mf) in CORPORA.items():
        df = pd.read_csv(cf, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
        man = json.load(open(mf))
        plan = {k: pool for k, pool, _, _ in build_plan(df, man, 6000, 3000)}
        label_of = df.set_index("tweet_id")["label"]
        text_of = df.set_index("tweet_id")["text_norm"]
        classes = sorted(df["label"].unique())
        events = sorted(man["events"])
        R = {e: set(man["events"][e]["R"]) for e in events}
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(df["text_norm"])
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", args.plan, corpus, "*"))):
            rec = {"corpus": corpus, "model": os.path.basename(model_dir), "em": [], "strata": [], "rare": [],
                   "stray_matched": [], "mc_by_absent": {}}
            strata_rows = []
            for sd in sorted(glob.glob(os.path.join(model_dir, "seed*"))):
                seed = int(sd.rsplit("seed", 1)[1])
                idr = pd.read_csv(os.path.join(sd, "id_random", "preds.tsv.gz"), sep="\t", dtype={"tweet_id": "int64"},
                                  keep_default_na=False)
                id_train = run_rng("id_random", seed).choice(plan["id_random"], size=6000, replace=False)
                Tm = vec.transform(text_of.loc[id_train].to_numpy()).T.tocsc()
                for e in events:
                    l = pd.read_csv(os.path.join(sd, f"loeo__{e}", "preds.tsv.gz"), sep="\t",
                                    dtype={"tweet_id": "int64"}, keep_default_na=False)
                    a = idr[idr["event"] == e].set_index("tweet_id")
                    lr = l.set_index("tweet_id").loc[a.index]
                    # 1. oracle label-prior matching on R_e for both models
                    a_adj = prior_match(a, classes)
                    l_adj = prior_match(lr, classes)
                    e2 = f1p(a["y_true"], a["y_pred"]) - f1p(lr["y_true"], lr["y_pred"])
                    e2_pm = f1p(a["y_true"], a_adj.to_numpy()) - f1p(lr["y_true"], l_adj.to_numpy())
                    rec["em"].append({"seed": seed, "event": e, "E2": e2, "E2_after_prior_match": e2_pm})
                    # 3. rare-class sensitivity
                    vc = a["y_true"].value_counts()
                    keep = sorted(vc[vc >= 5].index)
                    if keep:
                        fa = f1_score(a["y_true"], a["y_pred"], labels=keep, average="macro", zero_division=0)
                        fl = f1_score(lr["y_true"], lr["y_pred"], labels=keep, average="macro", zero_division=0)
                        rec["rare"].append({"seed": seed, "event": e, "E2_classes_ge5": fa - fl})
                    # 4. stray classes on matched instances
                    present = set(l["y_true"])
                    presentR = set(a["y_true"])
                    rec["stray_matched"].append({
                        "seed": seed, "event": e,
                        "id_on_R": int(len(set(a["y_pred"]) - presentR)),
                        "loeo_on_R": int(len(set(lr["y_pred"]) - presentR)),
                        "loeo_on_event": int(len(set(l["y_pred"]) - present)),
                        "n_absent_classes": int(len(classes) - len(present)),
                        "MC_e": f1p(l["y_true"], l["y_pred"]) - f1_score(l["y_true"], l["y_pred"], average="macro", zero_division=0)})
                    # 2. similarity strata
                    S = (vec.transform(text_of.loc[a.index].to_numpy()) @ Tm).max(1).toarray().ravel()
                    strata_rows.append(pd.DataFrame({"sim": S, "y": a["y_true"].to_numpy(), "id": a["y_pred"].to_numpy(),
                                                     "loeo": lr["y_pred"].to_numpy(), "seed": seed}))
            st = pd.concat(strata_rows)
            for lo, hi in STRATA:
                g = st[(st["sim"] >= lo) & (st["sim"] < hi)]
                if len(g) == 0:
                    continue
                per_seed = [f1p(x["y"], x["id"]) - f1p(x["y"], x["loeo"]) for _, x in g.groupby("seed")]
                rec["strata"].append({"stratum": [lo, min(hi, 1.0)], "n_tweets_per_seed": int(len(g) / g["seed"].nunique()),
                                      "pooled_E2_P_mean_over_seeds": float(np.mean(per_seed))})
            sm = pd.DataFrame(rec["stray_matched"])
            rec["mc_by_absent"] = sm.groupby("n_absent_classes")["MC_e"].agg(["mean", "count"]).round(4).reset_index().to_dict(orient="records")
            em = pd.DataFrame(rec["em"]).groupby("event")[["E2", "E2_after_prior_match"]].mean()
            rec["prior_match_summary"] = {"E2": float(em["E2"].mean()), "E2_after_prior_match": float(em["E2_after_prior_match"].mean()),
                                          "share_closed": float(1 - em["E2_after_prior_match"].mean() / em["E2"].mean()) if em["E2"].mean() != 0 else None}
            rr = pd.DataFrame(rec["rare"]).groupby("event")["E2_classes_ge5"].mean()
            rec["rare_summary"] = {"E2_classes_ge5": float(rr.mean()), "n_events": int(len(rr))}
            rec["stray_summary"] = sm[["id_on_R", "loeo_on_R", "loeo_on_event"]].mean().round(3).to_dict()
            out.append(rec)
            print(corpus, rec["model"], rec["prior_match_summary"], rec["rare_summary"], rec["stray_summary"], rec["strata"])
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
