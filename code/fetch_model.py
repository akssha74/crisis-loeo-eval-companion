"""Download a pinned Hugging Face model revision with parallel HTTP range requests.

The CDN throttles each connection, so large weight files are fetched in parallel chunks and
verified against the SHA-256 the Hub publishes for LFS files (X-Linked-ETag).

    python code/fetch_model.py distilbert-base-uncased
"""
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FILES = {
    "distilbert-base-uncased": ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                                "model.safetensors"],
    "roberta-base": ["config.json", "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors"],
    "bert-base-uncased": ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                          "model.safetensors"],
}
CHUNK = 4 * 1024 * 1024
WORKERS = 24


def api_sha(repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        return json.load(r)["sha"]


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=30) as r:
            return r.headers
    except urllib.error.HTTPError as e:
        return e.headers


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def fetch_range(url, start, end, path):
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError("short read")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return
        except Exception as exc:
            if attempt == 7:
                raise
            print(f"  retry {start}-{end}: {exc!r}", flush=True)


def main():
    repo = sys.argv[1]
    rev = api_sha(repo)
    out = os.path.join("data", "models", repo, rev)
    os.makedirs(out, exist_ok=True)
    manifest = {"repo": repo, "revision": rev, "files": {}}
    for name in FILES[repo]:
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
        path = os.path.join(out, name)
        h = head(url)
        linked_sha = (h.get("X-Linked-ETag") or "").strip('"')
        loc = h.get("Location")
        if loc and loc.startswith("/"):
            loc = "https://huggingface.co" + loc
        size = int(h.get("X-Linked-Size") or 0)
        if not size:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            open(path, "wb").write(data)
        else:
            if not (os.path.exists(path) and os.path.getsize(path) == size and
                    hashlib.sha256(open(path, "rb").read()).hexdigest() == linked_sha):
                with open(path, "wb") as f:
                    f.truncate(size)
                ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
                print(f"{name}: {size / 1e6:.1f} MB in {len(ranges)} chunks", flush=True)
                with ThreadPoolExecutor(WORKERS) as ex:
                    for i, _ in enumerate(ex.map(lambda r: fetch_range(loc, r[0], r[1], path), ranges)):
                        if i % 10 == 0:
                            print(f"  {i + 1}/{len(ranges)}", flush=True)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if linked_sha and digest != linked_sha:
            raise SystemExit(f"sha256 mismatch for {name}: {digest} != {linked_sha}")
        manifest["files"][name] = {"sha256": digest, "bytes": os.path.getsize(path),
                                   "hub_linked_sha256": linked_sha or None}
        print(f"ok {name} {digest[:12]}", flush=True)
    json.dump(manifest, open(os.path.join(out, "fetch_manifest.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
