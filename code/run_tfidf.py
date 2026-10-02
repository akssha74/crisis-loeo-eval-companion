"""Exploratory X3: TF-IDF + logistic regression through every base condition of `--plan confirm`.

Training samples are drawn exactly as in code/run_protocol.py (same run key, seed and pools); where the
transformer run of the same key and seed exists, its recorded train_ids_sha256 must match. Settings are fixed
and untuned. Predictions use the runner's format, under experiments/runs/x3_tfidf/<corpus>/tfidf-lr/seed<s>/.

    python code/run_tfidf.py --corpus humaid19 --seeds 42,1,2,3,4
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from run_protocol import CORPORA, PAIRED_DEDUP, build_plan, run_rng, sha256_file

TAG, PLAN = "tfidf-lr", "x3_tfidf"
SETTINGS = {"ngram_range": [1, 2], "lowercase": True, "min_df": 2, "sublinear_tf": True, "C": 1.0,
            "solver": "lbfgs", "max_iter": 2000}


def train_ids_for(run_key, pool, n_override, pools, cap, seed):
    n_train = min(cap, len(pool)) if n_override is None else min(n_override, cap)
    paired = PAIRED_DEDUP.get(run_key)
    if paired is None:
        return np.sort(run_rng(run_key, seed).choice(pool, size=n_train, replace=False))
    b_pool, b_nov = pools[paired]
    b_n = min(cap, len(b_pool)) if b_nov is None else min(b_nov, cap)
    base_ids = run_rng(paired, seed).choice(b_pool, size=b_n, replace=False)
    keep = base_ids[np.isin(base_ids, pool)]
    fill_pool = pool[~np.isin(pool, base_ids)]
    fill = run_rng(run_key + "|fill", seed).choice(fill_pool, size=n_train - len(keep), replace=False)
    return np.sort(np.concatenate([keep, fill]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--chrono_min", type=int, default=3000)
    ap.add_argument("--ledger", default="experiments/run-ledger.jsonl")
    args = ap.parse_args()
    corpus_file, manifest_file = CORPORA[args.corpus]
    df = pd.read_csv(corpus_file, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    manifest = json.load(open(manifest_file))
    assert manifest["canonical_sha256"] == sha256_file(corpus_file), "corpus changed since manifest"
    labels = sorted(df["label"].unique())
    df_idx = df.set_index("tweet_id")
    plan = build_plan(df, manifest, args.cap, args.chrono_min)
    pools = {k: (pool, n_ov) for k, pool, n_ov, _ in plan}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    code_sha = sha256_file(__file__)
    for seed in map(int, args.seeds.split(",")):
        out_root = os.path.join("experiments", "runs", PLAN, args.corpus, TAG, f"seed{seed}")
        for run_key, pool, n_override, eval_ids in plan:
            safe = run_key.replace(":", "__")
            out_dir = os.path.join(out_root, safe)
            if os.path.exists(os.path.join(out_dir, "run.json")):
                continue
            os.makedirs(out_dir, exist_ok=True)
            started, t0 = datetime.now(timezone.utc), time.time()
            train_ids = train_ids_for(run_key, pool, n_override, pools, args.cap, seed)
            ids_sha = hashlib.sha256(",".join(map(str, train_ids)).encode()).hexdigest()
            refs = glob.glob(f"experiments/runs/confirm/{args.corpus}/*/seed{seed}/{safe}/run.json")
            for ref in refs:
                assert json.load(open(ref))["train_ids_sha256"] == ids_sha, f"training sample differs from {ref}"
            tr, ev = df_idx.loc[train_ids], df_idx.loc[np.asarray(eval_ids)]
            vec = TfidfVectorizer(ngram_range=tuple(SETTINGS["ngram_range"]), lowercase=True,
                                  min_df=SETTINGS["min_df"], sublinear_tf=True)
            clf = LogisticRegression(C=SETTINGS["C"], solver="lbfgs", max_iter=SETTINGS["max_iter"])
            clf.fit(vec.fit_transform(tr["text"]), tr["label"])
            probs = np.zeros((len(ev), len(labels)))
            probs[:, [labels.index(c) for c in clf.classes_]] = clf.predict_proba(vec.transform(ev["text"]))
            pred = pd.DataFrame({"tweet_id": ev.index.astype(str), "event": ev["event"].to_numpy(),
                                 "y_true": ev["label"].to_numpy(), "y_pred": [labels[i] for i in probs.argmax(1)]})
            for i, l in enumerate(labels):
                pred[f"p_{l}"] = np.round(probs[:, i], 4)
            pred_path = os.path.join(out_dir, "preds.tsv.gz")
            pred.to_csv(pred_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
            meta = {"run_key": run_key, "plan": PLAN, "corpus": args.corpus, "model": TAG, "seed": seed,
                    "settings": SETTINGS, "sklearn": sklearn.__version__, "python": sys.version.split()[0],
                    "cap": args.cap, "pool_n": int(len(pool)), "n_train": int(len(tr)), "n_eval": int(len(ev)),
                    "train_ids_sha256": ids_sha, "train_sample_checked_against": len(refs),
                    "train_event_counts": tr["event"].value_counts().to_dict(), "labels": labels,
                    "n_iter": int(clf.n_iter_.max()), "elapsed_sec": round(time.time() - t0, 1),
                    "manifest_sha256": sha256_file(manifest_file), "corpus_sha256": manifest["canonical_sha256"],
                    "code_sha256": code_sha, "commit": commit}
            json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
            with open(args.ledger, "a") as f:
                f.write(json.dumps({
                    "run_id": f"{PLAN}-{args.corpus}-{TAG}-s{seed}-{safe}", "node_id": "n-x3-tfidf",
                    "execution_kind": "training", "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
                    "cwd": ".", "run_key": run_key, "started_at": started.isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(), "status": "succeeded", "exit_code": 0,
                    "output_artifacts": [{"path": pred_path, "sha256": sha256_file(pred_path)}]}) + "\n")
            print(f"{args.corpus} s{seed} {run_key}: n_train={len(tr)} checked={len(refs)} {time.time() - t0:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
