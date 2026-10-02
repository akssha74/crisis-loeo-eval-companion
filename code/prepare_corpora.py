"""Build the two confirmation corpora in one schema.

data/corpus_humaid19.tsv  HumAID set1 (13 events, development-visible) + set2 (6 events, protected=1)
data/corpus_crisislext26.tsv  CrisisLexT26 information-type task, 26 events (outcome-exposed)

Columns: tweet_id, posted_at_utc, event, hazard, protected, text, label, text_norm, is_retweet.
Only metadata is printed; CrisisLexT26 label counts are printed because that corpus was already
outcome-exposed by the rejected study, HumAID set2 labels are never summarised.

    python code/prepare_corpora.py
"""
import glob
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from prepare_humaid import dedup_key, normalize, snowflake_to_utc  # noqa: E402

CLX_DIR = "data/crisislext26/d67cddd51131dbb28842209014f67cffe359ff42"
CLX_DROP = {"Not labeled", "Not applicable"}
CLX_LABELS = {
    "Affected individuals": "affected_individuals",
    "Caution and advice": "caution_and_advice",
    "Donations and volunteering": "donations_and_volunteering",
    "Infrastructure and utilities": "infrastructure_and_utilities",
    "Other Useful Information": "other_useful_information",
    "Sympathy and support": "sympathy_and_support",
}
COLS = ["tweet_id", "posted_at_utc", "event", "hazard", "protected", "text", "label", "text_norm", "is_retweet"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def humaid19():
    s1 = pd.read_csv("data/humaid_canonical.tsv", sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    s2 = pd.read_csv("data/humaid_set2_canonical.tsv", sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    s1["protected"], s2["protected"] = 0, 1
    df = pd.concat([s1, s2])[COLS].sort_values(["event", "tweet_id"]).reset_index(drop=True)
    assert df["tweet_id"].is_unique
    out = "data/corpus_humaid19.tsv"
    df.to_csv(out, sep="\t", index=False)
    return out, {"sources": {"set1": sha("data/humaid_canonical.tsv"), "set2": sha("data/humaid_set2_canonical.tsv")},
                 "n": int(len(df)), "events": int(df["event"].nunique()),
                 "protected_events": sorted(df.loc[df["protected"] == 1, "event"].unique().tolist()),
                 "sha256": sha(out)}


def crisislex():
    rows, n_read, dropped = [], 0, {}
    for path in sorted(glob.glob(os.path.join(CLX_DIR, "*", "*-tweets_labeled.csv"))):
        ev = os.path.basename(os.path.dirname(path))
        desc = json.load(open(path.replace("-tweets_labeled.csv", "-event_description.json")))
        hazard = desc["categorization"]["type"].strip().lower().replace(" ", "_")
        df = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        n_read += len(df)
        for r in df.itertuples(index=False):
            lab = getattr(r, "_3").strip()
            if lab in CLX_DROP or lab not in CLX_LABELS:
                dropped[lab] = dropped.get(lab, 0) + 1
                continue
            tid = int(str(getattr(r, "_0")).strip().strip('"'))
            text = getattr(r, "_1")
            rows.append((tid, ev.lower(), hazard, text, CLX_LABELS[lab]))
    df = pd.DataFrame(rows, columns=["tweet_id", "event", "hazard", "text_raw", "label"])
    dup_ids = int(df.duplicated("tweet_id").sum())
    conflicting = int(df.groupby("tweet_id")["label"].nunique().gt(1).sum())
    df = df.drop_duplicates("tweet_id", keep="first").copy()
    df["posted_at_utc"] = df["tweet_id"].map(lambda i: snowflake_to_utc(int(i)).isoformat())
    df["protected"] = 0
    df["text"] = df["text_raw"].map(normalize)
    df["text_norm"] = df["text_raw"].map(dedup_key)
    df["is_retweet"] = df["text_raw"].str.match(r"^RT\s+@").astype(int)
    df = df[df["text"].str.len() > 0][COLS].sort_values(["event", "tweet_id"]).reset_index(drop=True)
    out = "data/corpus_crisislext26.tsv"
    df.to_csv(out, sep="\t", index=False)
    within_dup = int(sum(g.duplicated("text_norm").sum() for _, g in df.groupby("event")))
    per_event = {}
    for ev, g in df.groupby("event"):
        t = pd.to_datetime(g["posted_at_utc"], format="ISO8601")
        per_event[ev] = {"n": int(len(g)), "hazard": g["hazard"].iloc[0], "first_post_utc": t.min().isoformat(),
                         "last_post_utc": t.max().isoformat(), "retweet_share": round(float(g["is_retweet"].mean()), 3),
                         "within_event_normalized_duplicate_rows": int(g.duplicated("text_norm").sum()),
                         "n_label_classes": int(g["label"].nunique())}
    return out, {"source": {"repo": "sajao/CrisisLex", "commit": CLX_DIR.split("/")[-1],
                            "fetch_manifest_sha256": sha(os.path.join(CLX_DIR, "fetch_manifest.json"))},
                 "rows_read": n_read, "dropped_by_label": dropped, "duplicate_tweet_id_rows_dropped": dup_ids,
                 "tweet_ids_with_conflicting_labels": conflicting, "n": int(len(df)),
                 "labels": df["label"].value_counts().to_dict(), "within_event_normalized_duplicate_rows": within_dup,
                 "events": per_event, "sha256": sha(out)}


def main():
    h_out, h_meta = humaid19()
    c_out, c_meta = crisislex()
    meta = {"humaid19": h_meta, "crisislext26": c_meta}
    json.dump(meta, open("research/prelock/corpora_metadata.json", "w"), indent=1)
    print(json.dumps(h_meta, indent=1))
    print(json.dumps({k: v for k, v in c_meta.items() if k != "events"}, indent=1))
    for ev, e in sorted(c_meta["events"].items(), key=lambda kv: kv[1]["first_post_utc"]):
        print(f"{ev:34s} n={e['n']:4d} {e['first_post_utc'][:16]} cls={e['n_label_classes']} "
              f"rt={e['retweet_share']:.2f} dup={e['within_event_normalized_duplicate_rows']} {e['hazard']}")


if __name__ == "__main__":
    main()
