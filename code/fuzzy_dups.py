"""Label-free near-duplicate detection between the id_random training pool and the random test union R.

A pool tweet is a near-duplicate of R if the cosine similarity of character 3-5-gram TF-IDF vectors of
the normalized texts (code/prepare_humaid.dedup_key) is at least --threshold to some tweet in R.
Exact normalized duplicates are included. Output: experiments/splits/fuzzydup_<corpus>.json

    python code/fuzzy_dups.py --corpus crisislext26 --threshold 0.8
    python code/fuzzy_dups.py --corpus crisislext26 --inspect        (print sample pairs per band; no labels)
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from run_protocol import CORPORA, sha256_file


def max_sim(pool_texts, test_texts, chunk=2000):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    vec.fit(np.concatenate([pool_texts, test_texts]))
    P, T = vec.transform(pool_texts), vec.transform(test_texts).T.tocsc()
    best, arg = np.zeros(P.shape[0]), np.zeros(P.shape[0], dtype=int)
    for i in range(0, P.shape[0], chunk):
        S = (P[i:i + chunk] @ T).toarray()
        best[i:i + chunk], arg[i:i + chunk] = S.max(1), S.argmax(1)
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    cf, mf = CORPORA[args.corpus]
    df = pd.read_csv(cf, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False,
                     usecols=["tweet_id", "text_norm"])
    man = json.load(open(mf))
    R = np.concatenate([np.array(e["R"], dtype=np.int64) for e in man["events"].values()])
    in_R = df["tweet_id"].isin(R).to_numpy()
    pool, test = df[~in_R].reset_index(drop=True), df[in_R].reset_index(drop=True)
    best, arg = max_sim(pool["text_norm"].to_numpy(), test["text_norm"].to_numpy())
    if args.inspect:
        rng = np.random.default_rng(0)
        for lo, hi in [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.999), (0.999, 1.01)]:
            idx = np.where((best >= lo) & (best < hi))[0]
            print(f"\n== cosine [{lo}, {hi}): {len(idx)} pool tweets")
            for i in rng.choice(idx, size=min(4, len(idx)), replace=False):
                print(f"  POOL: {pool['text_norm'][i][:110]}\n  TEST: {test['text_norm'][arg[i]][:110]}")
        return
    near = pool.loc[best >= args.threshold, "tweet_id"].astype(int).tolist()
    out = {"corpus": args.corpus, "threshold": args.threshold, "representation": "char_wb 3-5-gram TF-IDF, sublinear tf",
           "labels_read": False, "n_pool": int(len(pool)), "n_near": len(near),
           "n_exact": int((best >= 0.9999).sum()), "corpus_sha256": sha256_file(cf),
           "code_sha256": sha256_file(__file__), "pool_ids_near_R": sorted(near)}
    path = f"experiments/splits/fuzzydup_{args.corpus}.json"
    json.dump(out, open(path, "w"))
    print({k: v for k, v in out.items() if k != "pool_ids_near_R"}, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
