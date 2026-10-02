"""Build the canonical HumAID (13-event set1) table with posting times decoded from tweet IDs.

Output: data/humaid_canonical.tsv with one row per unique tweet_id and columns
tweet_id, posted_at_utc, event, hazard, official_split, text, label, text_norm, is_retweet.
Also writes research/prelock/humaid_metadata.json (metadata and aggregate counts only).
"""
import argparse
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

TWITTER_EPOCH_MS = 1288834974657

EVENT_TO_HAZARD = {
    "canada_wildfires_2016": "wildfire",
    "greece_wildfires_2018": "wildfire",
    "cyclone_idai_2019": "tropical_cyclone",
    "hurricane_harvey_2017": "tropical_cyclone",
    "hurricane_irma_2017": "tropical_cyclone",
    "hurricane_maria_2017": "tropical_cyclone",
    "hurricane_matthew_2016": "tropical_cyclone",
    "ecuador_earthquake_2016": "earthquake",
    "italy_earthquake_aug_2016": "earthquake",
    "kaikoura_earthquake_2016": "earthquake",
    "puebla_mexico_earthquake_2017": "earthquake",
    "maryland_floods_2018": "flood",
    "srilanka_floods_2017": "flood",
}


def snowflake_to_utc(tweet_id: int) -> datetime:
    return datetime.fromtimestamp(((tweet_id >> 22) + TWITTER_EPOCH_MS) / 1000, tz=timezone.utc)


def normalize(t: str) -> str:
    t = re.sub(r"http\S+", "[URL]", str(t))
    t = re.sub(r"@\w+", "[USER]", t)
    return t.strip()


def dedup_key(t: str) -> str:
    t = normalize(t).lower()
    t = re.sub(r"^rt\s+\[user\]:?\s*", "", t)
    t = re.sub(r"\[url\]|\[user\]", " ", t)
    t = re.sub(r"[^a-z0-9#\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/events_set1")
    ap.add_argument("--out", default="data/humaid_canonical.tsv")
    ap.add_argument("--meta_out", default="research/prelock/humaid_metadata.json")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tsv"), recursive=True)):
        fname = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"(.+)_(train|dev|test)$", fname)
        event, split = m.group(1), m.group(2)
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        for r in df.itertuples(index=False):
            rows.append((r.tweet_id, event, split, r.tweet_text, r.class_label.strip()))
    raw = pd.DataFrame(rows, columns=["tweet_id", "event", "official_split", "text_raw", "label"])
    n_raw = len(raw)
    raw = raw[(raw["text_raw"].str.len() > 0) & (raw["label"].str.len() > 0)]
    raw["tweet_id"] = raw["tweet_id"].astype("int64")
    n_nonempty = int(len(raw))

    dup_ids = int(raw.duplicated("tweet_id").sum())
    conflicting = int(raw.groupby("tweet_id")["label"].nunique().gt(1).sum())
    raw = raw.drop_duplicates("tweet_id", keep="first").copy()

    raw["posted_at_utc"] = raw["tweet_id"].map(lambda i: snowflake_to_utc(int(i)).isoformat())
    raw["hazard"] = raw["event"].map(EVENT_TO_HAZARD)
    raw["text"] = raw["text_raw"].map(normalize)
    raw["text_norm"] = raw["text_raw"].map(dedup_key)
    raw["is_retweet"] = raw["text_raw"].str.match(r"^RT\s+@").astype(int)
    out = raw[["tweet_id", "posted_at_utc", "event", "hazard", "official_split", "text", "label",
               "text_norm", "is_retweet"]].sort_values(["event", "tweet_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)

    events = {}
    for ev, g in out.groupby("event"):
        t = pd.to_datetime(g["posted_at_utc"], format="ISO8601")
        within_dup = int(g.duplicated("text_norm").sum())
        events[ev] = {
            "hazard": EVENT_TO_HAZARD[ev],
            "n": int(len(g)),
            "first_post_utc": t.min().isoformat(),
            "median_post_utc": t.median().isoformat(),
            "last_post_utc": t.max().isoformat(),
            "span_days": round((t.max() - t.min()).total_seconds() / 86400, 2),
            "retweet_share": round(float(g["is_retweet"].mean()), 4),
            "within_event_normalized_duplicate_rows": within_dup,
            "official_split_counts": g["official_split"].value_counts().to_dict(),
        }
    cross_event_dup = int(out.groupby("text_norm")["event"].nunique().gt(1).sum())
    official_leak = 0
    for ev, g in out.groupby("event"):
        tr = set(g.loc[g["official_split"] != "test", "text_norm"])
        official_leak += int(g.loc[g["official_split"] == "test", "text_norm"].isin(tr).sum())

    meta = {
        "exposure_level": "metadata_only (plus corpus-level label counts; HumAID labels were already fully "
                          "outcome-exposed by the rejected SNCS study, see prior-work.jsonl)",
        "source_archive": "AI_Disaster_IEEE/humaid_set1.tar.gz",
        "source_archive_sha256": hashlib.sha256(
            open("../../AI_Disaster_IEEE/humaid_set1.tar.gz", "rb").read()).hexdigest(),
        "rows_read": n_raw,
        "rows_after_empty_filter_before_id_dedup": n_nonempty,
        "duplicate_tweet_id_rows_dropped": dup_ids,
        "tweet_ids_with_conflicting_labels": int(conflicting),
        "unique_tweets": int(len(out)),
        "labels": out["label"].value_counts().to_dict(),
        "cross_event_shared_normalized_texts": cross_event_dup,
        "official_test_rows_with_normalized_text_in_same_event_train_or_dev": official_leak,
        "events": events,
        "canonical_sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
    }
    os.makedirs(os.path.dirname(args.meta_out), exist_ok=True)
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "events"}, indent=1))
    for ev, e in sorted(events.items(), key=lambda kv: kv[1]["first_post_utc"]):
        print(f"{ev:32s} n={e['n']:5d} {e['first_post_utc'][:16]} -> {e['last_post_utc'][:16]} "
              f"span={e['span_days']:6.1f}d rt={e['retweet_share']:.2f} dup={e['within_event_normalized_duplicate_rows']}")


if __name__ == "__main__":
    main()
