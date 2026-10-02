"""Exploratory (not preregistered) class-level breakdown, from saved predictions only.

1. Stray predictions: LOEO predictions of classes absent from the held-out event, by class.
2. Matched exposure by class: per event and class present in R_e, F1_class(id_random, R_e) minus
   F1_class(loeo_e, R_e), averaged over events containing the class and over seeds.

    python code/analyze_classes.py --plan confirm
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analyze import CORPORA, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="confirm")
    ap.add_argument("--out", default="experiments/derived/classes_confirm.json")
    args = ap.parse_args()
    out = []
    for corpus, man_path in CORPORA.items():
        man = json.load(open(man_path))
        events = sorted(man["events"])
        R = {e: set(map(str, man["events"][e]["R"])) for e in events}
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", args.plan, corpus, "*"))):
            stray = defaultdict(list)
            gain = defaultdict(list)
            seeds = sorted(glob.glob(os.path.join(model_dir, "seed*")))
            for sd in seeds:
                idr = load(sd, "id_random")
                s_counts = defaultdict(int)
                for e in events:
                    l = load(sd, f"loeo:{e}")
                    present = set(l["y_true"])
                    for c, n in l.loc[~l["y_pred"].isin(present), "y_pred"].value_counts().items():
                        s_counts[c] += int(n)
                    a = idr[idr["event"] == e]
                    lr = l[l["tweet_id"].isin(R[e])].set_index("tweet_id").loc[a["tweet_id"]].reset_index()
                    labs = sorted(set(a["y_true"]))
                    fa = f1_score(a["y_true"], a["y_pred"], labels=labs, average=None, zero_division=0)
                    fl = f1_score(lr["y_true"], lr["y_pred"], labels=labs, average=None, zero_division=0)
                    for c, x, y in zip(labs, fa, fl):
                        gain[c].append((e, os.path.basename(sd), float(x - y)))
                for c in set(s_counts) | set(stray):
                    stray[c].append(s_counts.get(c, 0))
            classes = sorted(set(gain) | set(stray))
            rows = []
            for c in classes:
                g = pd.DataFrame(gain.get(c, []), columns=["event", "seed", "d"])
                ev_mean = g.groupby("event")["d"].mean() if len(g) else pd.Series(dtype=float)
                rows.append({"class": c,
                             "stray_predictions_per_seed": float(np.mean(stray[c])) if c in stray else 0.0,
                             "matched_gain_mean": float(ev_mean.mean()) if len(ev_mean) else None,
                             "events_with_class_in_R": int(len(ev_mean)),
                             "events_positive": int((ev_mean > 0).sum()) if len(ev_mean) else 0})
            out.append({"corpus": corpus, "model": os.path.basename(model_dir), "n_seeds": len(seeds),
                        "exploratory": True, "classes": rows})
            print(corpus, os.path.basename(model_dir))
            print(pd.DataFrame(rows).round(3).to_string(index=False))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
