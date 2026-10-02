"""Confirmation runner: fine-tune an encoder under every registered training condition for one corpus,
model and seed, and save per-instance predictions (gzip TSV). Metrics are computed offline by
code/analyze.py from the saved predictions only.

Conditions (training pool -> evaluation set); every training set is capped at --cap tweets:
  id_random         all tweets not in any R_e                               -> union of R_e
  id_random_dedup   id_random pool minus tweets whose normalized text is in union R -> union of R_e
  id_temporal       all tweets not in any T_e                               -> union of T_e
  id_temporal_dedup id_temporal pool minus tweets whose normalized text is in union T -> union of T_e
  loeo:<e>          tweets of every other event                             -> all tweets of e
  chrono:<e>        tweets posted before e's first tweet (pool >= --chrono_min) -> all tweets of e
  loeon:<e>         loeo pool subsampled to |chrono pool| where that is < cap -> all tweets of e

    python code/run_protocol.py --corpus humaid19 --model data/models/<repo>/<rev> --seed 42 --plan confirm
"""
import argparse
import gc
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
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CORPORA = {
    "humaid19": ("data/corpus_humaid19.tsv", "experiments/splits/confirm_humaid19.json"),
    "crisislext26": ("data/corpus_crisislext26.tsv", "experiments/splits/confirm_crisislext26.json"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_rng(run_key: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{run_key}|{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def build_plan(df, manifest, cap, chrono_min, extras=(), corpus=None):
    ids = df["tweet_id"].to_numpy()
    ev_arr = df["event"].to_numpy()
    norm = df["text_norm"].to_numpy()
    t = pd.to_datetime(df["posted_at_utc"], format="ISO8601")
    info = manifest["events"]
    by_event = {ev: ids[ev_arr == ev] for ev in sorted(info)}
    R_all = np.concatenate([np.array(e["R"], dtype=np.int64) for e in info.values()])
    T_all = np.concatenate([np.array(e["T"], dtype=np.int64) for e in info.values()])
    in_R, in_T = np.isin(ids, R_all), np.isin(ids, T_all)
    norm_R, norm_T = set(norm[in_R]), set(norm[in_T])
    dup_R = np.array([n in norm_R for n in norm])
    dup_T = np.array([n in norm_T for n in norm])
    plan = [
        ("id_random", ids[~in_R], None, np.sort(R_all)),
        ("id_random_dedup", ids[~in_R & ~dup_R], None, np.sort(R_all)),
        ("id_temporal", ids[~in_T], None, np.sort(T_all)),
        ("id_temporal_dedup", ids[~in_T & ~dup_T], None, np.sort(T_all)),
    ]
    for ev in sorted(info):
        plan.append((f"loeo:{ev}", ids[ev_arr != ev], None, by_event[ev]))
    for ev in sorted(info):
        first = pd.Timestamp(info[ev]["first_post_utc"])
        pool = ids[(ev_arr != ev) & (t < first).to_numpy()]
        if len(pool) < chrono_min:
            continue
        plan.append((f"chrono:{ev}", pool, None, by_event[ev]))
        if len(pool) < cap:
            plan.append((f"loeon:{ev}", ids[ev_arr != ev], len(pool), by_event[ev]))
    if "fuzzy" in extras:
        near = set(json.load(open(f"experiments/splits/fuzzydup_{corpus}.json"))["pool_ids_near_R"])
        near_R = np.array([i in near for i in ids])
        plan.append(("id_random_fuzzydedup", ids[~in_R & ~dup_R & ~near_R], None, np.sort(R_all)))
    if "mask" in extras:
        plan.append(("id_random_mask", ids[~in_R], None, np.sort(R_all)))
        for ev in sorted(info):
            plan.append((f"loeo_mask:{ev}", ids[ev_arr != ev], None, by_event[ev]))
    if "divk" in extras:
        # diversity-matched control: LOEO restricted to k random other events, k = number of source events in
        # chrono(e)'s pool, trained on as many tweets as chrono(e); the k-event set is fixed per event.
        others_all = sorted(info)
        for ev in others_all:
            first = pd.Timestamp(info[ev]["first_post_utc"])
            cmask = (ev_arr != ev) & (t < first).to_numpy()
            if cmask.sum() < chrono_min:
                continue
            k = len(set(ev_arr[cmask]))
            n_c = min(cap, int(cmask.sum()))
            others = [x for x in others_all if x != ev]
            rng = np.random.default_rng(int(hashlib.sha256(f"divk|{ev}".encode()).hexdigest()[:16], 16))
            for _ in range(200):
                chosen = rng.choice(others, size=k, replace=False)
                kpool = ids[np.isin(ev_arr, chosen)]
                if len(kpool) >= n_c:
                    plan.append((f"loeok:{ev}", kpool, n_c, by_event[ev]))
                    break
    if "full" in extras:
        # full-pool arm: id_random on its whole pool, LOEO subsampled to the same size (no 6,000 cap)
        n_full = int((~in_R).sum())
        plan.append(("id_random_full", ids[~in_R], n_full, np.sort(R_all)))
        for ev in sorted(info):
            plan.append((f"loeo_full:{ev}", ids[ev_arr != ev], n_full, by_event[ev]))
    return plan


# Paired conditions (amendment A2): a dedup arm keeps every non-duplicate tweet of its base arm's sample and
# tops up from the same pool; a mask arm trains on exactly its base arm's sample with masked text.
PAIRED_DEDUP = {"id_random_dedup": "id_random", "id_temporal_dedup": "id_temporal",
                "id_random_fuzzydedup": "id_random"}


def mask_base(run_key):
    if run_key == "id_random_mask":
        return "id_random"
    if run_key.startswith("loeo_mask:"):
        return "loeo:" + run_key.split(":", 1)[1]
    return None


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
    step, curve = 0, []
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
        curve.append(round(run_loss / max(n, 1), 5))
        log(f"  epoch {ep + 1}/{epochs} mean_train_loss={curve[-1]:.4f} steps={step}")
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in eval_dl:
            batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            probs.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    return np.concatenate(probs), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan", default="confirm")
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--chrono_min", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--only", default="", help="comma-separated run keys")
    ap.add_argument("--extras", default="", help="opt-in extra conditions: fuzzy,mask (amendment A2)")
    ap.add_argument("--train_limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--node_id", default="n-confirm")
    ap.add_argument("--ledger", default="experiments/run-ledger.jsonl")
    ap.add_argument("--no_ledger", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    corpus_file, manifest_file = CORPORA[args.corpus]
    df = pd.read_csv(corpus_file, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False)
    manifest = json.load(open(manifest_file))
    assert manifest["canonical_sha256"] == sha256_file(corpus_file), "corpus changed since manifest"
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    df_idx = df.set_index("tweet_id")

    extras = tuple(x for x in args.extras.split(",") if x)
    plan = build_plan(df, manifest, args.cap, args.chrono_min, extras, args.corpus)
    pools = {k: (pool, n_ov) for k, pool, n_ov, _ in plan}
    masked = None
    if "mask" in extras:
        masked = pd.read_csv(f"data/corpus_{args.corpus}_masked.tsv", sep="\t", dtype={"tweet_id": "int64"},
                             keep_default_na=False).set_index("tweet_id")["text_masked"]
    if args.only:
        keep = set(args.only.split(","))  # entries ending in ":" select every run key with that prefix
        plan = [p for p in plan if p[0] in keep or any(k.endswith(":") and p[0].startswith(k) for k in keep)]
    if args.list:
        for k, pool, n_ov, ev in plan:
            print(f"{k:45s} pool={len(pool):6d} n_override={n_ov} n_eval={len(ev)}")
        print(len(plan), "runs")
        return
    parts = args.model.rstrip("/").split("/")
    tag = parts[parts.index("models") + 1] if "models" in parts[:-2] else parts[-1]
    out_root = os.path.join("experiments", "runs", args.plan, args.corpus, tag, f"seed{args.seed}")
    os.makedirs(out_root, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    code_sha = sha256_file(__file__)

    for run_key, pool, n_override, eval_ids in plan:
        safe = run_key.replace(":", "__")
        out_dir = os.path.join(out_root, safe)
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print(f"skip {run_key} (done)", flush=True)
            continue
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        try:
            os.mkdir(out_dir)  # atomic claim: parallel processes on the same seed never train the same run
        except FileExistsError:
            print(f"skip {run_key} (claimed by another process, or a stale directory to clear)", flush=True)
            continue
        log_path = os.path.join(out_dir, "train.log")
        logf = open(log_path, "w")

        def log(msg):
            line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        started = datetime.now(timezone.utc)
        t0 = time.time()
        n_train = min(args.cap, len(pool)) if n_override is None else min(n_override, args.cap)
        if run_key.split(":")[0].endswith("_full"):
            n_train = n_override
        paired = PAIRED_DEDUP.get(run_key) or mask_base(run_key)
        if paired is None:
            rng = run_rng(run_key, args.seed)
            train_ids = np.sort(rng.choice(pool, size=n_train, replace=False))
        else:
            b_pool, b_nov = pools[paired]
            b_n = min(args.cap, len(b_pool)) if b_nov is None else min(b_nov, args.cap)
            base_ids = run_rng(paired, args.seed).choice(b_pool, size=b_n, replace=False)
            if run_key in PAIRED_DEDUP:
                keep = base_ids[np.isin(base_ids, pool)]
                fill_pool = pool[~np.isin(pool, base_ids)]
                fill = run_rng(run_key + "|fill", args.seed).choice(fill_pool, size=n_train - len(keep), replace=False)
                train_ids = np.sort(np.concatenate([keep, fill]))
            else:
                train_ids = np.sort(base_ids)
        ev_ids = np.asarray(eval_ids)
        if args.train_limit:
            train_ids = train_ids[: args.train_limit]
            ev_ids = train_ids[:32]
        train_df = df_idx.loc[train_ids].reset_index()
        eval_df = df_idx.loc[ev_ids].reset_index()
        if mask_base(run_key):
            train_df["text"] = masked.loc[train_df["tweet_id"]].to_numpy()
            eval_df["text"] = masked.loc[eval_df["tweet_id"]].to_numpy()
        log(f"run={run_key} corpus={args.corpus} model={args.model} seed={args.seed} device={DEVICE} "
            f"pool={len(pool)} n_train={len(train_df)} n_eval={len(eval_df)} train_events={train_df['event'].nunique()}")
        exit_code, curve = 0, []
        try:
            probs, curve = train_and_predict(args.model, train_df, eval_df, label2id, args.seed, args.epochs,
                                             args.lr, args.bs, args.max_len, log)
        except Exception as exc:  # recorded as a failed execution, never as a scientific result
            log(f"FAILED: {exc!r}")
            exit_code = 1
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()  # long-lived streams otherwise keep every run's cached MPS memory
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
                pred[f"p_{l}"] = np.round(probs[:, i], 4)
            pred_path = os.path.join(out_dir, "preds.tsv.gz")
            pred.to_csv(pred_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
            meta = {
                "run_key": run_key, "plan": args.plan, "corpus": args.corpus, "model": args.model,
                "seed": args.seed, "device": DEVICE, "torch": torch.__version__,
                "transformers": transformers.__version__, "python": sys.version.split()[0],
                "platform": platform.platform(), "epochs": args.epochs, "lr": args.lr, "bs": args.bs,
                "max_len": args.max_len, "cap": args.cap, "pool_n": int(len(pool)), "n_train": int(len(train_df)),
                "n_eval": int(len(eval_df)), "train_loss_per_epoch": curve,
                "train_ids_sha256": hashlib.sha256(",".join(map(str, train_ids)).encode()).hexdigest(),
                "train_event_counts": train_df["event"].value_counts().to_dict(), "labels": labels,
                "elapsed_sec": round(elapsed, 1), "manifest_sha256": sha256_file(manifest_file),
                "corpus_sha256": manifest["canonical_sha256"], "code_sha256": code_sha, "commit": commit,
                "smoke": bool(args.train_limit), "paired_base": paired,
                "extras": list(extras),
            }
            json.dump(meta, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
            outputs = [{"path": pred_path, "sha256": sha256_file(pred_path)},
                       {"path": os.path.join(out_dir, "run.json"),
                        "sha256": sha256_file(os.path.join(out_dir, "run.json"))}]
        log(f"done exit={exit_code} elapsed={elapsed:.1f}s")
        logf.close()
        if not args.no_ledger:
            rec = {
                "run_id": f"{args.plan}-{args.corpus}-{tag}-s{args.seed}-{safe}",
                "node_id": args.node_id, "execution_kind": "training",
                "command": " ".join([os.path.basename(sys.executable)] + sys.argv), "cwd": ".",
                "run_key": run_key, "started_at": started.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "succeeded" if exit_code == 0 else "failed", "exit_code": exit_code,
                "log_path": log_path, "log_sha256": sha256_file(log_path), "output_artifacts": outputs,
            }
            with open(args.ledger, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
