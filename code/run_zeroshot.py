"""Exploratory X10 (research/preregistration.md): a public system trained on no event, scored through the
companion. Zero-shot NLI classification with facebook/bart-large-mnli at a pinned revision; one hypothesis per
class, "This message is about {class}.", prediction = class with the largest entailment logit.

    ../../.venv/bin/python code/run_zeroshot.py predict crisislext26
    ../../.venv/bin/python code/run_zeroshot.py predict humaid19
    python3 code/run_zeroshot.py score
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO, REV = "facebook/bart-large-mnli", "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
MODEL_DIR = os.path.join("data", "models", REPO, REV)
CORPUS_TSV = {"crisislext26": "data/corpus_crisislext26.tsv", "humaid19": "data/corpus_humaid19.tsv"}
OUT_DIR = os.path.join("experiments", "runs", "x10_zeroshot")
DERIVED = "experiments/derived/x10_zeroshot.json"
MAX_LEN = 128
CHUNK = 2048
BATCH_PAIRS = 256


def hypothesis(c):
    return f"This message is about {c.replace('_', ' ')}."


def entailment_logits(model, tok, dev, ent, prem, hyps, sort=True):
    """(len(prem), len(hyps)) entailment logits; premises are batched in token-length order and restored."""
    import torch
    lens = [len(ids) for ids in tok(prem, truncation=True, max_length=MAX_LEN)["input_ids"]]
    order = np.argsort(lens, kind="stable") if sort else np.arange(len(prem))
    pairs = [(prem[i], h) for i in order for h in hyps]
    logits = []
    with torch.no_grad():
        for b in range(0, len(pairs), BATCH_PAIRS):
            chunk = pairs[b:b + BATCH_PAIRS]
            enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            logits.append(model(**enc).logits[:, ent].float().cpu().numpy())
    z = np.empty((len(prem), len(hyps)), dtype=np.float32)
    z[order] = np.concatenate(logits).reshape(len(prem), len(hyps))
    return z


def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev).eval()
    return model, tok, dev, {k.lower(): v for k, v in model.config.label2id.items()}["entailment"]


def predict(corpus):
    df = pd.read_csv(CORPUS_TSV[corpus], sep="\t", dtype={"tweet_id": str}, keep_default_na=False)
    classes = sorted(df["label"].unique())
    hyps = [hypothesis(c) for c in classes]
    model, tok, dev, ent = load_model()
    run = os.path.join(OUT_DIR, corpus, "bart-large-mnli")
    parts = os.path.join(run, "parts")
    os.makedirs(parts, exist_ok=True)
    texts = df["text"].tolist()
    for start in range(0, len(df), CHUNK):
        path = os.path.join(parts, f"{start:07d}.npy")
        if os.path.exists(path):
            continue
        np.save(path, entailment_logits(model, tok, dev, ent, texts[start:start + CHUNK], hyps))
        print(f"{corpus} {min(start + CHUNK, len(df))}/{len(df)}", flush=True)
    z = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(parts, "*.npy")))])
    prob = np.exp(z - z.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    out = pd.DataFrame({"tweet_id": df["tweet_id"], "event": df["event"], "y_true": df["label"],
                        "y_pred": [classes[i] for i in z.argmax(1)]})
    for j, c in enumerate(classes):
        out[f"p_{c}"] = prob[:, j].round(4)
    out.to_csv(os.path.join(run, "preds.tsv.gz"), sep="\t", index=False)
    json.dump({"repo": REPO, "revision": REV, "hypotheses": dict(zip(classes, hyps)), "max_length": MAX_LEN,
               "device": dev, "n": len(df)}, open(os.path.join(run, "run.json"), "w"), indent=1)
    print("wrote", os.path.join(run, "preds.tsv.gz"))


def score():
    import analyze_v2 as av
    import companion_eval as ce
    from revision_x7 import boot_ci, loeo_aggregation
    res = {"repo": REPO, "revision": REV, "corpora": {}}
    for corpus in CORPUS_TSV:
        meta = pd.read_csv(av.CORPORA[corpus][0], sep="\t", dtype=str, keep_default_na=False)
        pred = ce.read_predictions([os.path.join(OUT_DIR, corpus, "bart-large-mnli", "preds.tsv.gz")])
        rec = {}
        for conv in ("P", "D", "O"):
            r, cells = ce.score(pred, meta, conv)
            events = [e["event"] for e in r["events"]]
            c = np.stack([cells[cells["event"] == e][["tp", "fp", "fn"]].to_numpy(float) for e in events])[:, None]
            rec[conv] = {"pooled": r["pooled_macro_f1"], "per_event_mean": r["per_event_mean_macro_f1"], "A": r["A"],
                         "A_ci95": boot_ci(lambda ev, sd, cv=conv: loeo_aggregation({"c": c}, ev, sd, cv), len(events), 1)}
        res["corpora"][corpus] = rec
        print(corpus, {k: (round(v["pooled"], 3), round(v["per_event_mean"], 3), round(v["A"], 3),
                           [round(x, 3) for x in v["A_ci95"]]) for k, v in rec.items()})
    json.dump(res, open(DERIVED, "w"), indent=1)
    print("wrote", DERIVED)


def check_order(corpus="crisislext26", n=256):
    """Length-ordered batching against the first saved chunk, which was computed in corpus order."""
    df = pd.read_csv(CORPUS_TSV[corpus], sep="\t", dtype={"tweet_id": str}, keep_default_na=False)
    hyps = [hypothesis(c) for c in sorted(df["label"].unique())]
    ref = np.load(os.path.join(OUT_DIR, corpus, "bart-large-mnli", "parts", "0000000.npy"))[:n]
    z = entailment_logits(*load_model(), df["text"].tolist()[:n], hyps)
    print(f"max |diff| {np.abs(z - ref).max():.2e}  argmax agreement {(z.argmax(1) == ref.argmax(1)).mean():.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2])
    elif sys.argv[1] == "check_order":
        check_order()
    else:
        score()
