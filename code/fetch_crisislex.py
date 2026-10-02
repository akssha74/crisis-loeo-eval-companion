"""Download the CrisisLexT26 labelled tweets and event descriptions at a pinned commit.

    python code/fetch_crisislex.py          (writes data/crisislext26/<commit>/ and fetch_manifest.json)
"""
import hashlib
import json
import os
import urllib.request

REPO = "sajao/CrisisLex"
COMMIT = "d67cddd51131dbb28842209014f67cffe359ff42"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main():
    listing = json.loads(get(f"https://api.github.com/repos/{REPO}/contents/data/CrisisLexT26?ref={COMMIT}"))
    events = sorted(x["name"] for x in listing if x["type"] == "dir")
    out = os.path.join("data", "crisislext26", COMMIT)
    manifest = {"repo": REPO, "commit": COMMIT, "licence": "MIT (repository)", "files": {}}
    for ev in events:
        for name in (f"{ev}-tweets_labeled.csv", f"{ev}-event_description.json"):
            dest = os.path.join(out, ev, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                open(dest, "wb").write(get(
                    f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/data/CrisisLexT26/{ev}/{name}"))
            manifest["files"][f"{ev}/{name}"] = {"sha256": hashlib.sha256(open(dest, "rb").read()).hexdigest(),
                                                 "bytes": os.path.getsize(dest)}
        print("ok", ev, flush=True)
    json.dump(manifest, open(os.path.join(out, "fetch_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
