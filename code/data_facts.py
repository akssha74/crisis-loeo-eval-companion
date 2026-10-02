"""Descriptive facts for each corpus (no model outputs): classes per event, the size of event-class cells in
the matched test subsets R_e, and rows repeating an earlier tweet of the same event after normalization. Writes experiments/derived/data_facts.json.

    python code/data_facts.py
"""
import json

import pandas as pd

from prepare_humaid import dedup_key


def main():
    out = {}
    for corpus in ("humaid19", "crisislext26"):
        df = pd.read_csv(f"data/corpus_{corpus}.tsv", sep="\t", keep_default_na=False, usecols=["tweet_id", "event", "label", "text"])
        man = json.load(open(f"experiments/splits/confirm_{corpus}.json"))
        k = df["label"].nunique()
        per_event = df.groupby("event")["label"].nunique()
        r_ids = {int(x) for e in man["events"] for x in man["events"][e]["R"]}
        d_r = df[df["tweet_id"].isin(r_ids)]
        cells = d_r.groupby(["event", "label"]).size()
        dup_groups = df.assign(k=df["text"].map(dedup_key)).groupby(["event", "k"]).size()
        out[corpus] = {
            "n_classes": int(k),
            "n_events": int(len(per_event)),
            "classes_per_event_min": int(per_event.min()),
            "classes_per_event_max": int(per_event.max()),
            "events_lacking_a_class": int((per_event < k).sum()),
            "mean_absent_classes": float((k - per_event).mean()),
            "R_e_size_median": int(d_r.groupby("event").size().median()),
            "R_e_cells_present": int(len(cells)),
            "R_e_cells_lt5_share": float((cells < 5).mean()),
            "pooled_R_min_class_size": int(d_r.groupby("label").size().min()),
            "within_event_duplicate_rows": int((dup_groups - 1).sum()),
        }
    json.dump(out, open("experiments/derived/data_facts.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
