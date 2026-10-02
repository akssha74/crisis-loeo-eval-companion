"""Source-validation learning curves for the fixed recipe (lesson L26). No target event is scored.

For each listed LOEO fold: from the fold's training pool (all other events) draw 6,000 training tweets
and a disjoint 600-tweet source validation set, train for --epochs epochs with the registered recipe
(linear schedule over the full budget), and record source-validation loss and present-class macro-F1
after every epoch. Output: experiments/derived/convergence/<corpus>/<model>/<event>_s<seed>.json

    python code/convergence.py --corpus humaid19 --model data/models/<repo>/<rev> --events a,b,c
"""
import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from run_protocol import CORPORA, DEVICE, collate_fn, sha256_file


def curve(model_name, train_df, val_df, label2id, seed, epochs, lr=2e-5, bs=32, max_len=96):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id)).to(DEVICE)
    tr = list(zip(train_df["text"].tolist(), train_df["label"].map(label2id).tolist()))
    va = list(zip(val_df["text"].tolist(), val_df["label"].map(label2id).tolist()))
    g = torch.Generator()
    g.manual_seed(seed)
    tr_dl = DataLoader(tr, batch_size=bs, shuffle=True, generator=g, collate_fn=collate_fn(tok, max_len))
    va_dl = DataLoader(va, batch_size=128, shuffle=False, collate_fn=collate_fn(tok, max_len))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(tr_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss()
    out = []
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for batch in tr_dl:
            y = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, y)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            tl += float(loss.detach().cpu()) * len(y)
            n += len(y)
        model.eval()
        vl, preds, gold = 0.0, [], []
        with torch.no_grad():
            for batch in va_dl:
                y = batch.pop("labels")
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits.float().cpu()
                vl += float(loss_fn(logits, y)) * len(y)
                preds += logits.argmax(-1).tolist()
                gold += y.tolist()
        out.append({"epoch": ep + 1, "train_loss": round(tl / n, 5), "val_loss": round(vl / len(gold), 5),
                    "val_macro_f1_present": round(f1_score(gold, preds, average="macro",
                                                           labels=sorted(set(gold)), zero_division=0), 5)})
        print(json.dumps(out[-1]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--model", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    corpus_file, _ = CORPORA[args.corpus]
    df = pd.read_csv(corpus_file, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    parts = args.model.rstrip("/").split("/")
    tag = parts[parts.index("models") + 1] if "models" in parts[:-2] else parts[-1]
    for ev in args.events.split(","):
        out_path = os.path.join("experiments", "derived", "convergence", args.corpus, tag, f"{ev}_s{args.seed}.json")
        if os.path.exists(out_path):
            continue
        pool = df[df["event"] != ev]
        if "protected" in pool:
            pool = pool[pool["protected"] == 0]  # protected events never enter training or validation here
        rng = np.random.default_rng(int(hashlib.sha256(f"conv|{ev}|{args.seed}".encode()).hexdigest()[:16], 16))
        idx = rng.choice(len(pool), size=6600, replace=False)
        train_df, val_df = pool.iloc[idx[:6000]], pool.iloc[idx[6000:]]
        t0 = time.time()
        print(f"fold={ev} corpus={args.corpus} model={tag} seed={args.seed}", flush=True)
        rows = curve(args.model, train_df, val_df, label2id, args.seed, args.epochs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump({"corpus": args.corpus, "model": args.model, "held_out_event_not_scored": ev, "seed": args.seed,
                   "epochs": args.epochs, "n_train": 6000, "n_source_val": 600, "curve": rows,
                   "elapsed_sec": round(time.time() - t0, 1), "code_sha256": sha256_file(__file__)},
                  open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
