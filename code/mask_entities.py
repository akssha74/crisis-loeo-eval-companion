"""Event-token masking for the consequence arm (amendment A2), label-free.

Following the event-specific-token masking baseline evaluated by Seeberger et al. (2025) -- named
entities, hashtags and numbers -- every spaCy entity span is replaced by its type in brackets (e.g.
"[GPE]"), remaining hashtags by "[HASHTAG]" and remaining digit strings by "[NUM]". The [URL] and
[USER] placeholders from normalization are kept. Runs in the isolated .venv-mask environment.

    .venv-mask/bin/python code/mask_entities.py --corpus humaid19
Output: data/corpus_<corpus>_masked.tsv (tweet_id, text_masked) and research/prelock/masking_<corpus>.json
"""
import argparse
import hashlib
import json
import re

import pandas as pd
import spacy

CORPORA = {"humaid19": "data/corpus_humaid19.tsv", "crisislext26": "data/corpus_crisislext26.tsv"}
PROTECT = re.compile(r"\[(URL|USER)\]")


def mask_doc(doc):
    out, last = [], 0
    for ent in doc.ents:
        if PROTECT.fullmatch(ent.text.strip()):
            continue
        out.append(doc.text[last:ent.start_char])
        out.append(f"[{ent.label_}]")
        last = ent.end_char
    out.append(doc.text[last:])
    text = "".join(out)
    text = re.sub(r"#\w+", "[HASHTAG]", text)
    text = re.sub(r"(?<![\w\[])\d[\d,.:]*", "[NUM]", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    args = ap.parse_args()
    path = CORPORA[args.corpus]
    df = pd.read_csv(path, sep="\t", dtype={"tweet_id": "int64"}, keep_default_na=False, usecols=["tweet_id", "text"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    masked, n_ent = [], []
    for doc in nlp.pipe(df["text"].tolist(), batch_size=512):
        masked.append(mask_doc(doc))
        n_ent.append(len(doc.ents))
    df["text_masked"] = masked
    out = f"data/corpus_{args.corpus}_masked.tsv"
    df[["tweet_id", "text_masked"]].to_csv(out, sep="\t", index=False)
    changed = (df["text_masked"] != df["text"]).mean()
    meta = {"corpus": args.corpus, "spacy": spacy.__version__, "model": "en_core_web_sm " + nlp.meta["version"],
            "labels_read": False, "n": int(len(df)), "share_tweets_changed": round(float(changed), 4),
            "mean_entities_per_tweet": round(float(pd.Series(n_ent).mean()), 3),
            "source_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "output_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
            "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "examples": [{"before": a, "after": b} for a, b in zip(df["text"].head(3), df["text_masked"].head(3))]}
    json.dump(meta, open(f"research/prelock/masking_{args.corpus}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "examples"}))
    for e in meta["examples"]:
        print(" ", e["before"][:100], "\n ->", e["after"][:100])


if __name__ == "__main__":
    main()
