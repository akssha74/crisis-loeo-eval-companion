#!/usr/bin/env bash
# Build the LRE article (paper/main.tex) and Online Resource 1 (paper/esm.tex). Each document reads the other's
# labels, so both are built twice.
set -euo pipefail
cd "$(dirname "$0")/../paper"
TECTONIC=${TECTONIC:-/Users/akshay.sharma/Projects/paper-activities/studies-archive/disaster-resolution-reliability/environment/bin/tectonic}
OUT=${OUT:-/tmp/crisis_lre_build}
PY=${PY:-python3}
mkdir -p "$OUT"

build() {
  if ! "$TECTONIC" -X compile --keep-intermediates --keep-logs --outdir "$OUT" "$1.tex" > "$OUT/$1.out" 2>&1; then
    tail -40 "$OUT/$1.out"
    exit 1
  fi
}

for pass in 1 2; do
  build esm
  "$PY" ../code/xref_labels.py "$OUT/esm.aux" generated/esm_labels.tex
  build main
  "$PY" ../code/xref_labels.py "$OUT/main.aux" generated/main_labels.tex
done
cp "$OUT/main.pdf" main.pdf
cp "$OUT/esm.pdf" esm.pdf
for d in main esm; do
  echo "$d: $(pdfinfo "$d.pdf" | awk '/^Pages/{print $2}') pages"
  rg -i "overfull|undefined|multiply" "$OUT/$d.out" || true
done
