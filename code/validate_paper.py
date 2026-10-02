"""Fail-closed checks on the built manuscript against the analysis output.

    python code/validate_paper.py            (after code/make_paper_assets.py and a PDF build)
Exit status 0 only if every check passes; writes paper/validation_report.json.
"""
import filecmp
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

PAPER = "paper"
SUMMARY = "experiments/derived/summary_v2_confirm.json"
AUTHORS = [r"akshay[ ._-]?sharma", r"lalji[ ._-]?prasad"]
LEAKS = [r"databricks", r"github\.com/[a-z0-9-]*sharma", r"paper-activities"]
PLACEHOLDERS = [r"\?\?", r"\bpending\b", r"TODO", r"XXX", r"TBD"]
# Lesson L11/L24: observational-benchmark language and protocol branding without the implied constraint.
CAUSAL = [r"explains why", r"pinpoint", r"establishes a", r"\bconfirms?\b", r"\bproves?\b", r"\bcauses\b",
          r"deployment-faithful", r"\bguarantee", r"first (systematic|study|to)", r"no prior work"]


def tex_sources():
    files = [os.path.join(PAPER, "main.tex"), os.path.join(PAPER, "prose.tex")]
    files += sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex")))
    files += sorted(glob.glob(os.path.join(PAPER, "generated", "tab_*.tex")))
    return files


def main():
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    src = {f: open(f).read() for f in tex_sources()}
    defined = set(re.findall(r"\\csname res@([^\\]+)\\endcsname", open(os.path.join(PAPER, "generated", "results.tex")).read()))
    used = {k for s in src.values() for k in re.findall(r"\\res\{([^}]+)\}", s)}
    check("every \\res key is defined", used <= defined, sorted(used - defined))

    bib = open(os.path.join(PAPER, "references.bib")).read()
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for s in src.values() for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^]]*\]){0,2}\{([^}]+)\}", s)
             for k in grp.split(",")}
    check("every citation key exists in references.bib", cited <= bib_keys, sorted(cited - bib_keys))
    check("every bibliography entry is cited", bib_keys <= cited, sorted(bib_keys - cited))

    summary = json.load(open(SUMMARY))
    bad = []
    for r in summary:
        for sub, blk in r["subsets"].items():
            for conv in ("P", "D", "O"):
                b = blk[conv]
                resid = b["E0"]["mean"] - (b["E1"]["mean"] + b["E2"]["mean"] + b["E3"]["mean"])
                if abs(resid) > 1e-9 or b["decomposition_residual_max"] > 1e-9:
                    bad.append((r["corpus"], r["model"], sub, conv, resid))
            if abs(blk["four_term_residual"]) > 1e-9:
                bad.append((r["corpus"], r["model"], sub, "four-term", blk["four_term_residual"]))
    check("E0 = E1 + E2 + E3 (every draw) and E0^D = E1^P + E2^P + E3^P + MC in the summary", not bad, bad)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "code/make_paper_assets.py", "--summary", SUMMARY, "--out", tmp]
        subprocess.run(cmd, check=True, capture_output=True)
        stale = [os.path.basename(f) for f in glob.glob(os.path.join(tmp, "generated", "*.tex"))
                 if not filecmp.cmp(f, os.path.join(PAPER, "generated", os.path.basename(f)), shallow=False)]
        check("generated tables and numbers match a fresh regeneration from the summary", not stale, stale)

    texts = {name: subprocess.run(["pdftotext", "-layout", os.path.join(PAPER, f"{name}.pdf"), "-"],
                                  capture_output=True, text=True).stdout for name in ("main", "esm")}
    for name, text in texts.items():
        found = [p for p in PLACEHOLDERS if re.search(p, text, re.I)]
        check(f"no placeholder text in {name}.pdf", not found, found)
        causal = [p for p in CAUSAL if re.search(p, text, re.I)]
        check(f"no causal or overclaiming phrases in {name}.pdf (L11, L24)", not causal, causal)
        check(f"no unresolved references in {name}.pdf", "??" not in text and "[?]" not in text,
              sorted(set(re.findall(r"\?\?[\w-]*", text))))
        leaks = [p for p in LEAKS if re.search(p, text, re.I)]
        check(f"no employer or private-repository strings in {name}.pdf", not leaks, leaks)
    # LRE review is single-blind: the article must name its authors.
    missing = [p for p in AUTHORS if not re.search(p, texts["main"], re.I)]
    check("author names on the article's title page (single-blind review)", not missing, missing)

    report = {"ok": all(c["ok"] for c in checks), "n_checks": len(checks), "checks": checks}
    json.dump(report, open(os.path.join(PAPER, "validation_report.json"), "w"), indent=1)
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"] + ("" if c["ok"] else f"  -> {c['detail']}"))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
