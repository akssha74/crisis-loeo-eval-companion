"""Write experiments/systems.tsv, the index of reference systems that code/companion_eval.py --system reads:
system name, corpus, description, complete seeds, prediction-file pattern and metadata path.

    python code/systems_index.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v2 as av  # noqa: E402
from companion_eval import SYSTEMS  # noqa: E402
from revision_x9 import complete_seeds, specs  # noqa: E402
from revision_x11 import ZS  # noqa: E402

NAMES = {("2 epochs", "distilbert-base-uncased"): ("distilbert-2ep", "DistilBERT-base-uncased, 2 epochs, 6,000 messages"),
         ("2 epochs", "roberta-base"): ("roberta-2ep", "RoBERTa-base, 2 epochs, 6,000 messages"),
         ("4 epochs", "distilbert-base-uncased"): ("distilbert-4ep", "DistilBERT-base-uncased, 4 epochs, 6,000 messages"),
         ("4 epochs", "roberta-base"): ("roberta-4ep", "RoBERTa-base, 4 epochs, 6,000 messages"),
         ("TF-IDF", "tfidf-lr"): ("tfidf-lr", "TF-IDF word 1-2-grams + logistic regression, 6,000 messages"),
         ("uncapped", "distilbert-base-uncased"): ("distilbert-large", "DistilBERT-base-uncased, 2 epochs, 80% of corpus")}


def main():
    rows = []
    for arm, corpus, model, model_dir, prefix, seeds in specs():
        events = sorted(json.load(open(av.CORPORA[corpus][1]))["events"])
        ok, _ = complete_seeds(model_dir, events, prefix, seeds)
        name, desc = NAMES[arm, model]
        rows.append({"system": name, "corpus": corpus, "description": desc, "seeds": ",".join(ok),
                     "preds": os.path.join(model_dir, "seed{seed}", f"{prefix}__*", "preds.tsv.gz"),
                     "meta": av.CORPORA[corpus][0]})
    for corpus in av.CORPORA:
        rows.append({"system": "bart-mnli-zeroshot", "corpus": corpus,
                     "description": "BART-large-MNLI zero-shot entailment, trained on no event", "seeds": "",
                     "preds": os.path.join(ZS, corpus, "bart-large-mnli", "preds.tsv.gz"),
                     "meta": av.CORPORA[corpus][0]})
    pd.DataFrame(rows).sort_values(["corpus", "system"]).to_csv(SYSTEMS, sep="\t", index=False)
    print(f"wrote {SYSTEMS} ({len(rows)} systems)")


if __name__ == "__main__":
    main()
