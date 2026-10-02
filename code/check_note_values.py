"""Recompute every generated value of the Project Note (keys note-*, the result tables and the usage example) from
experiments/derived and compare it with note_values.json, the values printed in the note.

    python code/check_note_values.py            # compare
    python code/check_note_values.py --write    # record the current values
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_paper_assets as mpa  # noqa: E402

DERIVED = "experiments/derived"


def current():
    load = lambda name: json.load(open(os.path.join(DERIVED, name)))  # noqa: E731
    x9, x10, ids = load("revision_x9.json"), load("x10_zeroshot.json"), load("training_ids_check.json")
    out = mpa.note_keys(x9, x10, ids)
    out["table:note-results"] = mpa.table_note_results(x9, x10)
    out["table:note-fixtures"] = mpa.table_note_fixtures(x9)
    out["table:note-events"] = mpa.table_note_events(x9)
    out["usage-example"] = mpa.usage_example()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default="note_values.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    now = current()
    if a.write:
        json.dump(now, open(a.values, "w"), indent=1, sort_keys=True)
        print(f"wrote {len(now)} values to {a.values}")
        return
    ref = json.load(open(a.values))
    bad = sorted(k for k in ref.keys() | now.keys() if ref.get(k) != now.get(k))
    print(f"values compared {len(ref)}  differing {len(bad)}")
    for k in bad[:20]:
        print(f"  {k}: note {ref.get(k)!r:.60}  now {now.get(k)!r:.60}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
