"""Snapshot of completed training runs (count and summed per-run wall-clock hours) for the paper's
implementation details. Writes experiments/derived/run_stats.json; rerun when more runs finish.

    python code/run_stats.py
"""
import datetime
import glob
import json


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob("experiments/runs/confirm*/*/*/*/*/run.json"))]
    tfidf = sorted(glob.glob("experiments/runs/x3_tfidf/*/*/*/*/run.json"))
    out = {"as_of_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_runs": len(runs), "hours": sum(r["elapsed_sec"] for r in runs) / 3600, "n_tfidf_runs": len(tfidf)}
    json.dump(out, open("experiments/derived/run_stats.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
