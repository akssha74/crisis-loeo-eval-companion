#!/bin/zsh
# Rebuild every generated asset and the PDF from a clean export of a commit (default HEAD) and compare the
# regenerated tables and \res values with the committed ones. With --full, first rerun the primary, four-epoch,
# as-registered, TF-IDF, aggregation, X5, X7 and X8 analyses from the committed predictions and metadata, and require them
# to reproduce the committed summaries.   Usage: code/repro_gate.sh [commit] [--full]
set -e
REV=${1:-HEAD}
FULL=${2:-}
STUDY=studies/disaster-crisis-text-matched-estimand-lre
ROOT=$(git rev-parse --show-toplevel)
TECTONIC=$ROOT/studies-archive/disaster-resolution-reliability/environment/bin/tectonic
PY=$ROOT/.venv/bin/python
TMP=$(mktemp -d /tmp/crisis_repro.XXXXXX)
cd "$ROOT"
PATHS=("$STUDY/code" "$STUDY/paper" "$STUDY/experiments/derived" ":(glob)$STUDY/experiments/runs/**/run.json")
[[ $FULL == --full ]] && PATHS+=("$STUDY/experiments/meta" "$STUDY/experiments/splits" ":(glob)$STUDY/experiments/runs/**/preds.tsv.gz")
git archive "$REV" "${PATHS[@]}" | tar -x -C "$TMP"
cd "$TMP/$STUDY"
cp -R paper/generated "$TMP/committed_generated"
if [[ $FULL == --full ]]; then
  cp -R experiments/derived "$TMP/committed_derived"
  "$PY" code/analyze_v2.py --plan confirm --skip_incomplete_extras >"$TMP/v2.log" 2>&1
  "$PY" code/analyze_v2.py --plan confirm_ep4 --out experiments/derived/summary_v2_confirm_ep4.json >"$TMP/ep4.log" 2>&1
  "$PY" code/analyze.py --plan confirm >"$TMP/reg.log" 2>&1
  "$PY" code/analyze_v2.py --plan x3_tfidf >"$TMP/x3.log" 2>&1
  "$PY" code/aggregation_mechanism.py >"$TMP/agg.log" 2>&1
  "$PY" code/revision_tmlr.py >"$TMP/rev.log" 2>&1
  "$PY" code/revision_x7.py >"$TMP/x7.log" 2>&1
  "$PY" code/revision_x8.py >"$TMP/x8.log" 2>&1
  "$PY" - "$TMP/committed_derived" experiments/derived <<'EOF'
import json, math, sys
# incomplete_seeds lists run directories still training when the summary was written; they are not committed.
IGNORE = {"incomplete_seeds"}
def same(a, b):
    if isinstance(a, dict):
        a = {k: v for k, v in a.items() if k not in IGNORE}
        b = {k: v for k, v in b.items() if k not in IGNORE} if isinstance(b, dict) else b
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b
bad = [f for f in ("summary_v2_confirm.json", "summary_v2_confirm_ep4.json", "summary_confirm.json",
                   "summary_v2_x3_tfidf.json", "aggregation_mechanism.json", "revision_tmlr.json",
                   "revision_x7.json", "revision_x8.json")
       if not same(json.load(open(f"{sys.argv[1]}/{f}")), json.load(open(f"{sys.argv[2]}/{f}")))]
print("REPRO analyses reproduce the committed summaries" if not bad else f"REPRO MISMATCH in {bad}")
sys.exit(1 if bad else 0)
EOF
fi
"$PY" code/make_paper_assets.py >/dev/null
if diff -r -q "$TMP/committed_generated" paper/generated; then
  echo "REPRO generated assets identical to $REV"
else
  echo "REPRO MISMATCH between committed and regenerated assets"; exit 1
fi
cd paper && "$TECTONIC" -X compile main.tex >/dev/null 2>&1 && echo "REPRO build ok: $(pdfinfo main.pdf | grep Pages)"
if pdftotext main.pdf - | grep -q '??'; then echo "REPRO unresolved references or keys"; exit 1; fi
echo "REPRO passed ($TMP)"
