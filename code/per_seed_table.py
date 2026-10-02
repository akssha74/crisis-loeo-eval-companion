"""Per-seed values of every registered estimand (for independent recomputation of the tables).

Reuses the frozen per-seed computation in code/analyze.py without modifying it.
    python code/per_seed_table.py --plan confirm
"""
import argparse
import glob
import json
import os

from analyze import CORPORA, diffs, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="confirm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for corpus, man_path in CORPORA.items():
        man = json.load(open(man_path))
        for model_dir in sorted(glob.glob(os.path.join("experiments", "runs", args.plan, corpus, "*"))):
            for sd in sorted(glob.glob(os.path.join(model_dir, "seed*"))):
                tab, pool = per_seed(sd, man)
                row = {"corpus": corpus, "model": os.path.basename(model_dir), "seed": os.path.basename(sd)[4:]}
                for conv in ("D", "P"):
                    d = diffs(tab, conv)
                    row[f"E0_{conv}"] = pool[conv] - d["c"].mean()
                    row[f"E1_{conv}"] = pool[conv] - d["a"].mean()
                    for est in ("E2", "E3", "E4", "E5", "DUP"):
                        if est in d:
                            row[f"{est}_{conv}"] = float(d[est].mean())
                    row[f"pooled_id_{conv}"] = pool[conv]
                    row[f"mean_loeo_all_{conv}"] = float(d["c"].mean())
                row["MC"] = float(diffs(tab, "P")["MC"].mean())
                rows.append(row)
                print(corpus, row["model"], row["seed"], round(row["E2_P"], 4), round(row["MC"], 4))
    out = args.out or os.path.join("experiments", "derived", f"per_seed_{args.plan}.json")
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
