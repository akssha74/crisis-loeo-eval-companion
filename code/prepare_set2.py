"""Canonical table for the six HumAID events outside set1 (the protected confirmation corpus).

Reads the pinned Hugging Face mirror, checks that its 13 set1 events carry exactly the tweet IDs of
the CrisisNLP set1 archive, and writes the six remaining events with posting times decoded from the
tweet IDs. Labels are copied through but never summarised or printed (metadata_only exposure).

    python code/prepare_set2.py
"""
import glob
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from prepare_humaid import dedup_key, normalize, snowflake_to_utc  # noqa: E402

HF_DIR = "data/humaid_hf/2e7ea23332006f075068b90d401b178d847b447a"
SET2_HAZARD = {
    "kerala_floods_2018": "flood",
    "midwestern_us_floods_2019": "flood",
    "hurricane_florence_2018": "tropical_cyclone",
    "hurricane_dorian_2019": "tropical_cyclone",
    "california_wildfires_2018": "wildfire",
    "pakistan_earthquake_2019": "earthquake",
}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(HF_DIR, "*", "*.json"))):
        event = os.path.basename(os.path.dirname(path))
        split = os.path.splitext(os.path.basename(path))[0]
        for r in json.load(open(path)):
            rows.append((int(r["tweet_id"]), event, split, r["tweet_text"], r["class_label"].strip()))
    hf = pd.DataFrame(rows, columns=["tweet_id", "event", "official_split", "text_raw", "label"])

    set1 = pd.read_csv("data/humaid_canonical.tsv", sep="\t", dtype={"tweet_id": "int64"},
                       keep_default_na=False, usecols=["tweet_id", "event"])
    hf1 = hf[hf["event"].isin(set(set1["event"]))]
    id_check = {
        "hf_set1_rows": int(len(hf1)),
        "archive_set1_rows": int(len(set1)),
        "same_tweet_id_set": bool(set(hf1["tweet_id"]) == set(set1["tweet_id"])),
        "same_event_assignment": bool(
            hf1.set_index("tweet_id")["event"].sort_index().equals(set1.set_index("tweet_id")["event"].sort_index())),
    }

    s2 = hf[hf["event"].isin(SET2_HAZARD)].copy()
    n_rows = int(len(s2))
    s2 = s2[(s2["text_raw"].str.len() > 0) & (s2["label"].str.len() > 0)]
    dup_ids = int(s2.duplicated("tweet_id").sum())
    s2 = s2.drop_duplicates("tweet_id", keep="first")
    s2["posted_at_utc"] = s2["tweet_id"].map(lambda i: snowflake_to_utc(int(i)).isoformat())
    s2["hazard"] = s2["event"].map(SET2_HAZARD)
    s2["text"] = s2["text_raw"].map(normalize)
    s2["text_norm"] = s2["text_raw"].map(dedup_key)
    s2["is_retweet"] = s2["text_raw"].str.match(r"^RT\s+@").astype(int)
    out = s2[["tweet_id", "posted_at_utc", "event", "hazard", "official_split", "text", "label",
              "text_norm", "is_retweet"]].sort_values(["event", "tweet_id"]).reset_index(drop=True)
    out_path = "data/humaid_set2_canonical.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    set1_norm = set(pd.read_csv("data/humaid_canonical.tsv", sep="\t", keep_default_na=False,
                                usecols=["text_norm"])["text_norm"])
    events = {}
    for ev, g in out.groupby("event"):
        t = pd.to_datetime(g["posted_at_utc"], format="ISO8601")
        events[ev] = {
            "hazard": SET2_HAZARD[ev], "n": int(len(g)),
            "first_post_utc": t.min().isoformat(), "last_post_utc": t.max().isoformat(),
            "span_days": round((t.max() - t.min()).total_seconds() / 86400, 2),
            "retweet_share": round(float(g["is_retweet"].mean()), 4),
            "within_event_normalized_duplicate_rows": int(g.duplicated("text_norm").sum()),
            "normalized_texts_also_in_set1": int(g["text_norm"].isin(set1_norm).sum()),
        }
    meta = {
        "exposure_level": "metadata_only: counts, posting times and duplicate statistics; labels never "
                          "summarised or printed. Disclosed incidental exposure: ~40 California-wildfire "
                          "rows in the Hub dataset-card preview and one Pakistan-earthquake dev row seen "
                          "while checking the JSON format.",
        "source": {"repo": "QCRI/HumAID-events", "revision": HF_DIR.split("/")[-1],
                   "fetch_manifest_sha256": hashlib.sha256(
                       open(os.path.join(HF_DIR, "fetch_manifest.json"), "rb").read()).hexdigest()},
        "set1_cross_check": id_check,
        "rows_read": n_rows, "duplicate_tweet_id_rows_dropped": dup_ids, "unique_tweets": int(len(out)),
        "events": events,
        "canonical_sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
    }
    os.makedirs("research/prelock", exist_ok=True)
    json.dump(meta, open("research/prelock/humaid_set2_metadata.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "events"}, indent=1))
    for ev, e in sorted(events.items(), key=lambda kv: kv[1]["first_post_utc"]):
        print(f"{ev:28s} n={e['n']:5d} {e['first_post_utc'][:16]} -> {e['last_post_utc'][:16]} "
              f"span={e['span_days']:6.1f}d rt={e['retweet_share']:.2f} dup={e['within_event_normalized_duplicate_rows']} "
              f"in_set1={e['normalized_texts_also_in_set1']}")


if __name__ == "__main__":
    main()
