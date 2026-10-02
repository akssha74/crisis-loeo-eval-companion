"""Pilot runner: fine-tune a text encoder under each registered training condition and save
per-instance predictions for every evaluation tweet. Metrics are computed later, offline, by
code/analyze_pilot.py from the saved predictions only.

Conditions (training pool -> evaluation set), every training set capped at --cap tweets:
  id_random       all tweets not in any R_e            -> union of R_e
  id_temporal     all tweets not in any T_e            -> union of T_e
  loeo:<e>        tweets of every other event          -> all tweets of e
  chrono:<e>      tweets posted before e's first post  -> all tweets of e   (pool >= --chrono_min)
  loeon:<e>       loeo pool subsampled to |chrono pool| -> all tweets of e  (only when that is < cap)

    python code/finetune.py --model distilbert-base-uncased --seed 42 --plan pilot
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import transformers

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_rng(run_key: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{run_key}|{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def build_plan(df, manifest, cap, chrono_min):
    all_ids = df["tweet_id"].to_numpy()
    by_event = {ev: g["tweet_id"].to_numpy() for ev, g in df.groupby("event")}
    ev_info = manifest["events"]
    R_all = np.concatenate([np.array(e["R"], dtype=np.int64) for e in ev_info.values()])
    T_all = np.concatenate([np.array(e["T"], dtype=np.int64) for e in ev_info.values()])
    t = pd.to_datetime(df["posted_at_utc"], format="ISO8601")

    plan = [
        ("id_random", all_ids[~np.isin(all_ids, R_all)], None, np.sort(R_all)),
        ("id_temporal", all_ids[~np.isin(all_ids, T_all)], None, np.sort(T_all)),
    ]
    for ev in sorted(by_event):
        plan.append((f"loeo:{ev}", all_ids[df["event"].to_numpy() != ev], None, by_event[ev]))
    for ev in sorted(by_event):
        first = pd.Timestamp(ev_info[ev]["first_post_utc"])
        pool = all_ids[(df["event"].to_numpy() != ev) & (t < first).to_numpy()]
        if len(pool) < chrono_min:
            continue
        plan.append((f"chrono:{ev}", pool, None, by_event[ev]))
        if len(pool) < cap:
            plan.append((f"loeon:{ev}", all_ids[df["event"].to_numpy() != ev], len(pool), by_event[ev]))
    return plan


def collate_fn(tok, max_len):
    def _c(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return _c


def train_and_predict(model_name, train_df, eval_df, label2id, seed, epochs, lr, bs, max_len, log):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id)).to(DEVICE)
    train_items = list(zip(train_df["text"].tolist(), train_df["label"].map(label2id).tolist()))
    eval_items = list(zip(eval_df["text"].tolist(), eval_df["label"].map(label2id).tolist()))
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(train_items, batch_size=bs, shuffle=True, generator=g, collate_fn=collate_fn(tok, max_len))
    eval_dl = DataLoader(eval_items, batch_size=128, shuffle=False, collate_fn=collate_fn(tok, max_len))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    step = 0
    for ep in range(epochs):
        run_loss, n = 0.0, 0
        for batch in train_dl:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            run_loss += float(loss.detach().cpu()) * len(labels)
            n += len(labels)
            step += 1
        log(f"  epoch {ep + 1}/{epochs} mean_train_loss={run_loss / max(n, 1):.4f} steps={step}")
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in eval_dl:
            batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            probs.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="data/humaid_canonical.tsv")
    ap.add_argument("--manifest", default="experiments/splits/pilot_splits.json")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan", default="pilot")
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--chrono_min", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--only", default="", help="comma-separated run keys (debug/smoke only)")
    ap.add_argument("--train_limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--eval_limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--node_id", default="n-pilot-01")
    ap.add_argument("--ledger", default="experiments/run-ledger.jsonl")
    ap.add_argument("--no_ledger", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.canonical, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    manifest = json.load(open(args.manifest))
    assert manifest["canonical_sha256"] == sha256_file(args.canonical), "canonical table changed since manifest"
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    df_idx = df.set_index("tweet_id")

    plan = build_plan(df, manifest, args.cap, args.chrono_min)
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p[0] in keep]
    parts = args.model.rstrip("/").split("/")
    tag = parts[parts.index("models") + 1] if "models" in parts[:-2] else parts[-1]
    out_root = os.path.join("experiments", "runs", args.plan, tag, f"seed{args.seed}")
    os.makedirs(out_root, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    code_sha = sha256_file(__file__)

    for run_key, pool, n_override, eval_ids in plan:
        safe = run_key.replace(":", "__")
        out_dir = os.path.join(out_root, safe)
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print(f"skip {run_key} (done)")
            continue
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "train.log")
        logf = open(log_path, "w")

        def log(msg):
            line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        started = datetime.now(timezone.utc)
        t0 = time.time()
        rng = run_rng(run_key, args.seed)
        n_train = min(args.cap, len(pool)) if n_override is None else min(n_override, args.cap)
        train_ids = np.sort(rng.choice(pool, size=n_train, replace=False))
        if args.train_limit:
            train_ids = train_ids[: args.train_limit]
        ev_ids = np.asarray(eval_ids)
        if args.train_limit:
            # smoke test evaluates on its own training tweets so no evaluation outcome is opened
            ev_ids = train_ids[: args.eval_limit or len(train_ids)]
        train_df = df_idx.loc[train_ids].reset_index()
        eval_df = df_idx.loc[ev_ids].reset_index()
        log(f"run={run_key} model={args.model} seed={args.seed} device={DEVICE} pool={len(pool)} "
            f"n_train={len(train_df)} n_eval={len(eval_df)} train_events={train_df['event'].nunique()}")
        exit_code = 0
        try:
            probs = train_and_predict(args.model, train_df, eval_df, label2id, args.seed, args.epochs,
                                      args.lr, args.bs, args.max_len, log)
        except Exception as exc:  # recorded as a failed execution, never as a scientific result
            log(f"FAILED: {exc!r}")
            exit_code = 1
        elapsed = time.time() - t0
        outputs = []
        if exit_code == 0:
            pred = pd.DataFrame({
                "tweet_id": eval_df["tweet_id"].astype(str),
                "event": eval_df["event"],
                "y_true": eval_df["label"],
                "y_pred": [labels[i] for i in probs.argmax(1)],
            })
            for i, l in enumerate(labels):
                pred[f"p_{l}"] = np.round(probs[:, i], 6)
            pred_path = os.path.join(out_dir, "preds.tsv")
            pred.to_csv(pred_path, sep="\t", index=False)
            meta = {
                "run_key": run_key, "plan": args.plan, "model": args.model, "seed": args.seed,
                "device": DEVICE, "torch": torch.__version__, "transformers": transformers.__version__,
                "python": sys.version.split()[0], "platform": platform.platform(),
                "epochs": args.epochs, "lr": args.lr, "bs": args.bs, "max_len": args.max_len, "cap": args.cap,
                "pool_n": int(len(pool)), "n_train": int(len(train_df)), "n_eval": int(len(eval_df)),
                "train_ids_sha256": hashlib.sha256(",".join(map(str, train_ids)).encode()).hexdigest(),
                "train_event_counts": train_df["event"].value_counts().to_dict(),
                "labels": labels, "elapsed_sec": round(elapsed, 1),
                "manifest_sha256": sha256_file(args.manifest), "code_sha256": code_sha, "commit": commit,
                "smoke": bool(args.train_limit or args.eval_limit),
            }
            json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
            outputs = [{"path": pred_path, "sha256": sha256_file(pred_path)},
                       {"path": os.path.join(out_dir, "run.json"),
                        "sha256": sha256_file(os.path.join(out_dir, "run.json"))}]
        log(f"done exit={exit_code} elapsed={elapsed:.1f}s")
        logf.close()
        if not args.no_ledger:
            rec = {
                "run_id": f"{args.plan}-{tag}-s{args.seed}-{safe}",
                "node_id": args.node_id,
                "execution_kind": "training",
                "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
                "cwd": ".",
                "run_key": run_key,
                "started_at": started.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "succeeded" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "log_path": log_path,
                "log_sha256": sha256_file(log_path),
                "output_artifacts": outputs,
            }
            with open(args.ledger, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
