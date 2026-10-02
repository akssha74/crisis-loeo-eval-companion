"""Outcome-independent split manifest for the pilot.

Uses only tweet_id, event and posting time (decoded from the tweet ID); labels are never read.
Per event e:
  R_e  random 20% of e (split seed fixed, independent of training seeds)
  T_e  latest 20% of e by posting time (ties broken by tweet_id)
Chronological pool for e: tweets of other events posted strictly before e's first post.
"""
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd

SPLIT_SEED = 20260928
TEST_FRAC = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="data/humaid_canonical.tsv")
    ap.add_argument("--out", default="experiments/splits/pilot_splits.json")
    args = ap.parse_args()

    df = pd.read_csv(args.canonical, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False,
                     usecols=["tweet_id", "event", "posted_at_utc"])
    df["t"] = pd.to_datetime(df["posted_at_utc"], format="ISO8601")
    rng = np.random.default_rng(SPLIT_SEED)

    events = {}
    for ev in sorted(df["event"].unique()):
        g = df[df["event"] == ev].sort_values(["t", "tweet_id"])
        n_test = math.ceil(TEST_FRAC * len(g))
        ids = g["tweet_id"].to_numpy()
        r_ids = np.sort(rng.choice(ids, size=n_test, replace=False))
        t_ids = ids[-n_test:]
        first = g["t"].min()
        prior = df[(df["event"] != ev) & (df["t"] < first)]
        events[ev] = {
            "n": int(len(g)),
            "first_post_utc": first.isoformat(),
            "R": [int(x) for x in r_ids],
            "T": [int(x) for x in t_ids],
            "T_cutoff_utc": g["t"].iloc[-n_test].isoformat(),
            "chrono_pool_n": int(len(prior)),
            "chrono_pool_events": sorted(prior["event"].unique().tolist()),
        }
    manifest = {
        "split_seed": SPLIT_SEED,
        "test_frac": TEST_FRAC,
        "labels_read": False,
        "canonical_sha256": hashlib.sha256(open(args.canonical, "rb").read()).hexdigest(),
        "events": events,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f)
    print("manifest sha256", hashlib.sha256(open(args.out, "rb").read()).hexdigest())
    for ev, e in sorted(events.items(), key=lambda kv: kv[1]["first_post_utc"]):
        print(f"{ev:32s} n={e['n']:5d} |R|={len(e['R']):4d} |T|={len(e['T']):4d} "
              f"T>= {e['T_cutoff_utc'][:16]} chrono_pool={e['chrono_pool_n']:6d} ({len(e['chrono_pool_events'])} ev)")


if __name__ == "__main__":
    main()
