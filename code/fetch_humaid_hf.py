"""Download every per-event split of QCRI/HumAID-events at a pinned Hub revision.

    python code/fetch_humaid_hf.py            (writes data/humaid_hf/<revision>/ and fetch_manifest.json)
"""
import hashlib
import json
import os
import urllib.request

REPO = "QCRI/HumAID-events"
REVISION = "2e7ea23332006f075068b90d401b178d847b447a"


def main():
    with urllib.request.urlopen(
            f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true", timeout=60) as r:
        tree = json.load(r)
    files = sorted(x["path"] for x in tree if x["type"] == "file" and x["path"].endswith((".json", ".md")))
    out = os.path.join("data", "humaid_hf", REVISION)
    manifest = {"repo": REPO, "revision": REVISION, "licence": "cc-by-nc-sa-4.0", "files": {}}
    for path in files:
        dest = os.path.join(out, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
            with urllib.request.urlopen(url, timeout=300) as r:
                open(dest, "wb").write(r.read())
        manifest["files"][path] = {"sha256": hashlib.sha256(open(dest, "rb").read()).hexdigest(),
                                   "bytes": os.path.getsize(dest)}
        print("ok", path, flush=True)
    json.dump(manifest, open(os.path.join(out, "fetch_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
