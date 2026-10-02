"""Rebuild the training identifiers of every whole-event LOEO reference run from the text-free metadata and check
them against the SHA-256 recorded in its run.json (the sampling of code/run_protocol.py and code/run_tfidf.py).

    python code/verify_training_ids.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = "experiments/derived/training_ids_check.json"
PLANS = ("confirm", "confirm_ep4", "x3_tfidf")
CORPORA = {"humaid19": ("experiments/meta/corpus_humaid19_meta.tsv", "experiments/splits/confirm_humaid19.json"),
           "crisislext26": ("experiments/meta/corpus_crisislext26_meta.tsv",
                            "experiments/splits/confirm_crisislext26.json")}


def run_rng(run_key, seed):
    digest = hashlib.sha256(f"{run_key}|{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def main():
    checked, bad, n_fulls = 0, [], {}
    for corpus, (meta_path, manifest_path) in CORPORA.items():
        meta = pd.read_csv(meta_path, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
        ids, ev_arr = meta["tweet_id"].to_numpy(), meta["event"].to_numpy()
        info = json.load(open(manifest_path))["events"]
        r_all = np.concatenate([np.array(e["R"], dtype=np.int64) for e in info.values()])
        n_full = n_fulls[corpus] = int((~np.isin(ids, r_all)).sum())
        for plan in PLANS:
            for path in sorted(glob.glob(os.path.join("experiments", "runs", plan, corpus, "*", "seed*",
                                                      "loeo*__*", "run.json"))):
                rec = json.load(open(path))
                kind, event = rec["run_key"].split(":", 1)
                if kind not in ("loeo", "loeo_full"):
                    continue
                pool = ids[ev_arr != event]
                n = n_full if kind == "loeo_full" else min(rec["cap"], len(pool))
                train = np.sort(run_rng(rec["run_key"], rec["seed"]).choice(pool, size=n, replace=False))
                sha = hashlib.sha256(",".join(map(str, train)).encode()).hexdigest()
                checked += 1
                if sha != rec["train_ids_sha256"] or n != rec["n_train"]:
                    bad.append(path)
    print(f"runs checked {checked}  mismatches {len(bad)}")
    for p in bad[:20]:
        print("  mismatch", p)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"runs_checked": checked, "mismatches": len(bad), "mismatched_runs": bad, "n_full": n_fulls,
               "plans": list(PLANS)}, open(OUT, "w"), indent=1)
    sys.exit(1 if bad or not checked else 0)


if __name__ == "__main__":
    main()
