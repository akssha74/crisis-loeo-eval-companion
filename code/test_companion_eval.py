"""Tests of the evaluation companion on small synthetic corpora. Run with pytest, or directly:

    python code/test_companion_eval.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import companion_eval as ce  # noqa: E402

CLASSES = ["a", "b", "c"]


def toy():
    """Two events; event x lacks class c, event y lacks class b and receives one prediction of b."""
    rows = [("1", "x", "a", "a"), ("2", "x", "a", "b"), ("3", "x", "b", "b"), ("4", "x", "b", "b"),
            ("5", "y", "a", "a"), ("6", "y", "c", "c"), ("7", "y", "c", "b"), ("8", "y", "c", "c")]
    pred = pd.DataFrame(rows, columns=["tweet_id", "event", "y_true", "y_pred"])
    meta = pred.rename(columns={"y_true": "label"})[["tweet_id", "event", "label"]].assign(hazard="h")
    return pred, meta


def random_corpus(seed, n_events=6, n_classes=4, size=60):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for e in range(n_events):
        present = rng.choice(n_classes, size=rng.integers(1, n_classes + 1), replace=False)
        for _ in range(size):
            y = int(rng.choice(present))
            p = y if rng.random() < 0.6 else int(rng.integers(n_classes))
            rows.append((str(k), f"e{e}", f"k{y}", f"k{p}"))
            k += 1
    pred = pd.DataFrame(rows, columns=["tweet_id", "event", "y_true", "y_pred"])
    for j in range(n_classes):
        if f"k{j}" not in set(pred["y_true"]):
            pred.loc[len(pred)] = (str(k + j), "e0", f"k{j}", f"k{j}")
    meta = pred.rename(columns={"y_true": "label"})[["tweet_id", "event", "label"]].assign(hazard="h")
    return pred, meta


def test_label_lists_by_hand():
    pred, meta = toy()
    p, _ = ce.score(pred, meta, "P")
    d, _ = ce.score(pred, meta, "D")
    o, _ = ce.score(pred, meta, "O")
    per = {r["event"]: r["macro_f1"] for r in p["events"]}
    assert np.isclose(per["x"], (2 / 3 + 4 / 5) / 2)
    assert np.isclose(per["y"], (1 + 4 / 5) / 2)
    assert np.isclose({r["event"]: r["macro_f1"] for r in d["events"]}["y"], (1 + 4 / 5 + 0) / 3)
    assert np.isclose({r["event"]: r["macro_f1"] for r in o["events"]}["x"], (2 / 3 + 4 / 5) / 3)
    assert np.isclose(p["pooled_macro_f1"], d["pooled_macro_f1"]) and np.isclose(p["pooled_macro_f1"],
                                                                                  o["pooled_macro_f1"])


def test_matches_sklearn():
    pred, meta = random_corpus(1)
    for conv in ("P", "D", "O"):
        res, _ = ce.score(pred, meta, conv)
        for r in res["events"]:
            g = pred[pred["event"] == r["event"]]
            labels = {"P": sorted(set(g["y_true"])), "D": None, "O": sorted(set(meta["label"]))}[conv]
            ref = f1_score(g["y_true"], g["y_pred"], labels=labels, average="macro", zero_division=0)
            assert np.isclose(r["macro_f1"], ref), (conv, r["event"])


def test_closed_form_shifts():
    for seed in range(5):
        pred, meta = random_corpus(seed)
        classes = sorted(meta["label"].unique())
        for e, g in pred.groupby("event"):
            cnt = ce.counts(g, classes)
            present = int(((cnt[:, 0] + cnt[:, 2]) > 0).sum())
            k = int((((cnt[:, 0] + cnt[:, 2]) == 0) & (cnt[:, 1] > 0)).sum())
            fp = ce.macro_f1(cnt, "P")
            assert np.isclose(ce.macro_f1(cnt, "O"), present / len(classes) * fp)
            assert np.isclose(ce.macro_f1(cnt, "D"), present / (present + k) * fp)


def test_split_sums_to_a_p():
    for seed in range(5):
        pred, meta = random_corpus(seed)
        res, _ = ce.score(pred, meta, "P")
        t = res["split_of_A_P"]
        assert np.isclose(t["T1"] + t["T2"] + t["T3"] + t["T4"], res["A_P"])
        assert np.isclose(t["T2alt"] + t["T3alt"], t["T2"] + t["T3"])
        assert t["T1"] <= 1e-12


def test_t4_zero_when_every_event_has_every_class():
    pred, meta = random_corpus(3, n_classes=1)
    res, _ = ce.score(pred, meta, "P")
    assert abs(res["split_of_A_P"]["T4"]) < 1e-12


def test_validation():
    pred, meta = toy()
    bad = {"duplicate": pd.concat([pred, pred.iloc[[0]]]),
           "unknown id": pred.assign(tweet_id=pred["tweet_id"].replace("1", "99")),
           "wrong label": pred.assign(y_true=pred["y_true"].where(pred["tweet_id"] != "1", "b")),
           "stray prediction": pred.assign(y_pred=pred["y_pred"].where(pred["tweet_id"] != "1", "z")),
           "incomplete event": pred.iloc[1:],
           "absent event": pred[pred["event"] == "x"]}
    for name, df in bad.items():
        try:
            ce.score(df, meta, "P")
        except ValueError:
            continue
        raise AssertionError(f"accepted {name}")
    res, _ = ce.score(pred[pred["event"] == "x"], meta, "P", partial=True)
    assert res["partial"] and res["events_absent"] == ["y"]


def test_signature():
    pred, meta = toy()
    res, _ = ce.score(pred, meta, "D")
    sig = ce.signature(res, "0123456789abcdef")
    assert sig == f"loeo-macro-f1|labels:D|zero-division:0|events:2|meta:0123456789ab|v:{ce.VERSION}"
    res, _ = ce.score(pred[pred["event"] == "x"], meta, "P", partial=True)
    assert "|events:1+partial|" in ce.signature(res, "0" * 64)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
