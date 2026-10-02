"""Values, tables and figure of the Project Note and its Online Resource 1, from experiments/derived.

    python code/note_assets.py            (writes paper/generated/note_*.tex, esm_*.tex and the figure)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_paper_assets as mpa  # noqa: E402
from make_paper_assets import CORP, f2, f3, plt  # noqa: E402
from revision_x7 import B  # noqa: E402

DERIVED = "experiments/derived"
CNAME = {"H": "HumAID", "C": "CrisisLexT26"}
SID = {("2 epochs", "distilbert-base-uncased"): "D2", ("2 epochs", "roberta-base"): "R2",
       ("4 epochs", "distilbert-base-uncased"): "D4", ("4 epochs", "roberta-base"): "R4",
       ("TF-IDF", "tfidf-lr"): "T", ("uncapped", "distilbert-base-uncased"): "L",
       ("zero-shot", "bart-large-mnli"): "Z"}
SNAME = {"D2": "DistilBERT, 2 ep.", "R2": "RoBERTa-base, 2 ep.", "D4": "DistilBERT, 4 ep.",
         "R4": "RoBERTa-base, 4 ep.", "T": "TF-IDF + LR", "L": "DistilBERT, large sample",
         "Z": "BART-MNLI zero-shot"}
TERMS = ("T1", "T2", "T3", "T4", "T2alt", "T3alt")
THRESHOLDS = ("5", "10", "15", "20", "30")
CORPORA = ("crisislext26", "humaid19")


def load():
    get = lambda n: json.load(open(os.path.join(DERIVED, n)))  # noqa: E731
    return (get("revision_x9.json"), get("x10_zeroshot.json"), get("revision_x11.json"),
            get("training_ids_check.json"), get("revision_x12.json"))


def sid_of(name):
    return SID[tuple(name.split("|"))]


def pct(x):
    return f"{100 * x:.0f}"


def x12_keys(kv, x11, x12):
    import pandas as pd
    for corpus in CORPORA:
        c = CORP[corpus]
        rows = dict(systems(x11, corpus))
        multi = [s for s, r in rows.items() if len(r["seeds"]) > 1]
        rng_keys(kv, f"note-{c}-APms", [rows[s]["A"]["P"]["mean"] for s in multi])
        kv[f"note-{c}-nms"] = str(len(multi))
        for cv, p in x12["pairs"][corpus].items():
            q = {(sid_of(r["i"]), sid_of(r["j"])): r for r in p["pairs"]}
            kv[f"note-{c}-{cv}-nclose"] = str(p["n_per_event_gap_below_abs_dA"])
            kv[f"note-{c}-{cv}-ndis"] = str(p["n_orders_disagree"])
            kv[f"note-{c}-{cv}-nexcl"] = str(sum(not r["dA_ci95"][0] <= 0 <= r["dA_ci95"][1] for r in q.values()))
            kv[f"note-{c}-{cv}-maxshare"] = pct(max(r["share_draws_orders_disagree"] for r in q.values()))
            ms = {k: r for k, r in q.items() if k[0] in multi and k[1] in multi}
            kv[f"note-{c}-{cv}-dAms-max"] = f3(max(abs(r["dA"]) for r in ms.values()))
            kv[f"note-{c}-{cv}-nexcl-ms"] = str(sum(not r["dA_ci95"][0] <= 0 <= r["dA_ci95"][1] for r in ms.values()))
            kv[f"note-{c}-{cv}-npairs-ms"] = str(len(ms))
            for (a, b), r in q.items():
                g = f"note-{c}-{cv}-{a}{b}"
                kv[g + "-share"] = pct(r["share_draws_orders_disagree"])
                kv[g + "-dA"], kv[g + "-dAabs"] = f3(r["dA"]), f3(abs(r["dA"]))
                kv[g + "-dAlo"], kv[g + "-dAhi"] = f3(r["dA_ci95"][0]), f3(r["dA_ci95"][1])
                kv[g + "-gpool"], kv[g + "-gper"] = f3(r["gap_pooled"]), f3(r["gap_per_event"])
        meta = pd.read_csv(f"experiments/meta/corpus_{corpus}_meta.tsv", sep="\t", dtype=str, keep_default_na=False)
        n_e = meta.groupby("event").size()
        share = 6000 / (len(meta) - n_e)
        kv[f"note-{c}-capshare-min"], kv[f"note-{c}-capshare-max"] = pct(share.min()), pct(share.max())
        wd = x12["within_event_duplicates"][corpus]
        kv[f"note-{c}-wdup-n"] = num(wd["dropped"])
        kv[f"note-{c}-wdup-shiftmax"] = f3(max(abs(s["A_P_all"] - s["A_P"]) for s in wd["systems"]))
        rng_keys(kv, f"note-{c}-wdup", [s["A_P"] for s in wd["systems"]])
        kv[f"note-{c}-wdup-cizero"] = str(sum(s["ci95"][0] <= 0 <= s["ci95"][1] for s in wd["systems"]))
        for pr in x12["precision_recall"]:
            if pr["corpus"] == corpus:
                m = mpa.MODEL[pr["model"]]
                kv[f"note-{c}-{m}-rhoP"], kv[f"note-{c}-{m}-rhoR"] = (f2(pr["median_rho_precision"]),
                                                                      f2(pr["median_rho_recall"]))
        zc = [z for z in x12["zero_cells"] if z["corpus"] == corpus]
        for m in ("D", "R"):
            kv[f"note-{c}-{m}-nzero"] = str(sum(mpa.MODEL[z["model"]] == m for z in zc))
        if corpus == "humaid19":
            k = "missing_or_found_people"
            kv["note-H-mofp-total"] = num(int((meta["label"] == k).sum()))
            for z in zc:
                if z["class"] == k and mpa.MODEL[z["model"]] == "R":
                    tag = {"california_wildfires_2018": "ca", "maryland_floods_2018": "md"}[z["event"]]
                    kv[f"note-H-mofp-{tag}"] = num(z["support"])
                    kv[f"note-H-mofp-{tag}-train"] = f"{z['train_sample_n_mean']:.0f}"
            kv["note-H-urgent-D-n"] = str(sum(z["class"] == "requests_or_urgent_needs" and mpa.MODEL[z["model"]] == "D"
                                              for z in zc))
    b = x12["bias"]
    kv["note-bias-min"], kv["note-bias-max"] = f3(min(r["bias"] for r in b)), f3(max(r["bias"] for r in b))
    kv["note-bias-absmax"] = f3(max(abs(r["bias"]) for r in b))
    flips = [r for r in b if (r["percentile_ci95"][0] <= 0 <= r["percentile_ci95"][1])
             != (r["bca"]["ci95"][0] <= 0 <= r["bca"]["ci95"][1])]
    kv["note-bca-nflip"] = str(len(flips))
    for r in flips:
        g = f"note-bca-{CORP[r['corpus']]}-{sid_of(r['system'])}"
        kv[g + "-lo"], kv[g + "-hi"] = f3(r["bca"]["ci95"][0]), f3(r["bca"]["ci95"][1])
    kv["note-incomplete-n"] = str(len(x12["incomplete_seed_runs"]))


def corpus_keys(kv):
    meta = json.load(open("research/prelock/corpora_metadata.json"))
    c = meta["crisislext26"]
    kv["note-C-rowsread"] = num(c["rows_read"])
    kv["note-C-dropNL"] = num(c["dropped_by_label"]["Not labeled"])
    kv["note-C-dropNA"] = num(c["dropped_by_label"]["Not applicable"])
    kv["note-C-dupid"] = str(c["duplicate_tweet_id_rows_dropped"])
    kv["note-C-withindup"] = num(c["within_event_normalized_duplicate_rows"])
    n = {"C": c["n"], "H": meta["humaid19"]["n"]}
    for k, v in n.items():
        kv[f"note-{k}-nmsg"] = num(v)
    kv["note-nmsg-total"] = num(sum(n.values()))
    return n


def systems(x11, corpus):
    return [(SID[r["arm"], r["model"]], r) for r in x11["systems"] if r["corpus"] == corpus]


def num(n):
    return f"{n:,}".replace(",", "{,}")


def exp10(x):
    return str(int(np.ceil(np.log10(max(x, 1e-300)))))


def rng_keys(kv, base, vals):
    kv[base + "-min"], kv[base + "-max"] = f3(min(vals)), f3(max(vals))


def size(n):
    return f"{n / 1e6:.1f}~MB" if n >= 1e6 else f"{max(1, round(n / 1e3))}~kB"


def inventory_keys(kv):
    path = os.path.join(DERIVED, "companion_inventory.json")
    if not os.path.exists(path):
        return
    inv = json.load(open(path))
    for g in inv["groups"]:
        slug = g["name"].split()[0].lower()
        kv[f"note-inv-{slug}-files"], kv[f"note-inv-{slug}-size"] = num(g["files"]), size(g["bytes"])
    kv["note-inv-total-files"], kv["note-inv-total-size"] = num(inv["total_files"]), size(inv["total_bytes"])


def keys(x9, x10, x11, ids, x12):
    kv = mpa.note_keys(x9, x10, ids)
    kv["note-runs-used"] = num(sum(len(a["seeds"]) * len(x9["per_event"][a["corpus"]]) for a in x9["arms"]))
    kv["note-runs-other"] = num(ids["runs_checked"] - sum(len(a["seeds"]) * len(x9["per_event"][a["corpus"]])
                                                          for a in x9["arms"]))
    x12_keys(kv, x11, x12)
    inventory_keys(kv)
    n_msgs = corpus_keys(kv)
    cf_dev, sk_dev, ndisc = 0.0, 0.0, 0
    for corpus in CORPORA:
        c = CORP[corpus]
        rows = systems(x11, corpus)
        kv[f"note-{c}-nsys"] = str(len(rows))
        for sid, r in rows:
            b = f"note-{c}-{sid}"
            kv[b + "-pool"], kv[b + "-per"] = f3(r["pooled"]["P"]), f3(r["per_event"]["P"])
            for cv in ("P", "D", "O"):
                mpa.ci_keys(kv, f"{b}-A{cv}", r["A"][cv])
                kv[f"{b}-per{cv}"], kv[f"{b}-pool{cv}"] = f3(r["per_event"][cv]), f3(r["pooled"][cv])
            for t in TERMS:
                kv[f"{b}-{t}"] = f3(r["terms"][t]["mean"])
            for t in THRESHOLDS:
                kv[f"{b}-t{t}"] = f3(r["thresholds"][t]["mean"])
            cfm = r["closed_forms"]
            kv[b + "-kmean"], kv[b + "-shiftO"], kv[b + "-shiftD"] = f2(cfm["k_mean"]), f3(cfm["shift_O"]), f3(cfm["shift_D"])
            cf_dev = max(cf_dev, cfm["max_abs_dev_event"], cfm["max_abs_dev_A"])
            if "sklearn_zero_division_nan" in r:
                sk_dev = max(sk_dev, r["sklearn_zero_division_nan"]["max_abs_diff"])
        for key, get in (("APall", lambda r: r["A"]["P"]["mean"]), ("ADall", lambda r: r["A"]["D"]["mean"]),
                         ("AOall", lambda r: r["A"]["O"]["mean"]),
                         ("shiftO", lambda r: r["closed_forms"]["shift_O"]),
                         ("shiftD", lambda r: r["closed_forms"]["shift_D"])):
            rng_keys(kv, f"note-{c}-{key}", [get(r) for _, r in rows])
        for t in TERMS:
            rng_keys(kv, f"note-{c}-{t}", [r["terms"][t]["mean"] for _, r in rows])
        for cv in ("P", "D", "O"):
            a_all = [r["A"][cv]["mean"] for _, r in rows]
            a_tr = [r["A"][cv]["mean"] for sid, r in rows if sid != "Z"]
            kv[f"note-{c}-A{cv}spread"] = f3(max(a_all) - min(a_all))
            kv[f"note-{c}-A{cv}spread-tr"] = f3(max(a_tr) - min(a_tr))
        by = dict(rows)
        for hi, lo in (("R4", "R2"), ("D4", "D2"), ("R2", "D2")):
            if hi in by and lo in by:
                g = f"note-{c}-{hi}{lo}-gap"
                kv[g + "-pool"] = f3(by[hi]["pooled"]["P"] - by[lo]["pooled"]["P"])
                for cv in ("P", "D", "O"):
                    kv[f"{g}-per{cv}"] = f3(by[hi]["per_event"][cv] - by[lo]["per_event"][cv])
                    kv[f"{g}-A{cv}"] = f3(by[hi]["A"][cv]["mean"] - by[lo]["A"][cv]["mean"])
                    kv[f"{g}-A{cv}abs"] = f3(abs(by[hi]["A"][cv]["mean"] - by[lo]["A"][cv]["mean"]))
        miss = max(r["share_draws_missing_a_class"] for _, r in rows)
        kv[f"note-{c}-missdraw-max"] = f"{100 * miss:.1f}"
        kv[f"note-{c}-missdraw-nmax"] = num(round(miss * B))
        kmeans = [r["closed_forms"]["k_mean"] for _, r in rows]
        kv[f"note-{c}-k-min"], kv[f"note-{c}-k-max"] = f2(min(kmeans)), f2(max(kmeans))
        kv[f"note-{c}-T4neg"] = str(sum(r["terms"]["T4"]["mean"] < 0 for _, r in rows))
        kv[f"note-{c}-T3pos"] = str(sum(r["terms"]["T3"]["mean"] > 0 for _, r in rows))
        kv[f"note-{c}-APcizero"] = str(sum(r["A"]["P"]["ci95"][0] <= 0 <= r["A"]["P"]["ci95"][1] for _, r in rows))
        jk = [max(abs(v - r["A"]["P"]["mean"]) for v in r["jackknife"]["values"].values()) for _, r in rows]
        kv[f"note-{c}-jkdev-max"] = f3(max(jk))
        kv[f"note-{c}-perfectAO"] = f3(x11["perfect_classifier_A_O"][corpus])
        for t in THRESHOLDS:
            rng_keys(kv, f"note-{c}-t{t}", [r["thresholds"][t]["mean"] for _, r in rows])
            th = x11["thresholds"][corpus][t]
            kv[f"note-{c}-t{t}-cells"], kv[f"note-{c}-t{t}-msgs"] = str(th["cells"]), num(th["messages"])
            kv[f"note-{c}-t{t}-pct"] = f"{100 * th['messages'] / n_msgs[c]:.1f}"
        d = x11["duplicates"][corpus]
        kv[f"note-{c}-dups"], kv[f"note-{c}-dupevents"] = str(d["messages"]), str(d["events_with_any"])
        dup_shift = max(abs(r["A_P_without_cross_event_duplicates"]["mean"] - r["A"]["P"]["mean"]) for _, r in rows)
        kv[f"note-{c}-dupshift-exp"] = exp10(dup_shift)
        for cv in ("P", "D", "O"):
            rk = x11["ranking"][corpus][cv]
            kv[f"note-{c}-{cv}-tau"], kv[f"note-{c}-{cv}-ndisc"] = f2(rk["kendall_tau"]), str(len(rk["discordant"]))
            ndisc += len(rk["discordant"])
        kv[f"note-{c}-npairs"] = str(x11["ranking"][corpus]["P"]["pairs"])
        for w in x11["within_class"]:
            if w["corpus"] == corpus:
                m = mpa.MODEL[w["model"]]
                kv[f"note-{c}-{m}-wrho"] = f2(w["median_rho"])
                kv[f"note-{c}-{m}-wrho-n"] = str(w["n_classes_defined"])
        small = {(s["event"], s["class"]) for s in x11["small_cells"][corpus]}
        for cls in {k for _, k in small}:
            kv[f"note-{c}-small-{cls.split('_')[0]}"] = str(sum(k == cls for _, k in small))
        zero = {(s["event"], s["class"]) for s in x11["small_cells"][corpus] if s["f1"] == 0}
        kv[f"note-{c}-small-f1zero-both"] = str(sum(
            all(s["f1"] == 0 for s in x11["small_cells"][corpus] if (s["event"], s["class"]) == cell) for cell in zero))
    kv["note-rank-ndisc"] = str(ndisc)
    kv["note-cf-exp"] = exp10(cf_dev)
    kv["note-sknan-exp"] = exp10(sk_dev)
    return kv


def table_results(x11):
    rows = []
    for corpus in CORPORA:
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        for sid, r in systems(x11, corpus):
            a, j = r["A"], r["jackknife"]
            rows.append(f"{SNAME[sid]} & {len(r['seeds'])} & {f3(r['per_event']['P'])} & {f3(r['pooled']['P'])} & "
                        f"{f3(a['P']['mean'])} [{f3(a['P']['ci95'][0])}, {f3(a['P']['ci95'][1])}] & "
                        f"{f3(j['min'])} to {f3(j['max'])} & {f3(a['D']['mean'])} & {f3(a['O']['mean'])} \\\\")
        if corpus == CORPORA[0]:
            rows.append("\\midrule")
    return "\n".join([
        "\\begin{table}[tbp]", "\\centering",
        "\\caption{Whole-event LOEO scores of every reference system. Per event and pooled are present-class (P) "
        "macro-F1 averaged over seeds. $A^P$, $A^D$ and $A^O$ are pooled minus per-event macro-F1 of the same "
        "predictions under the present-class, scikit-learn default and full-ontology label lists. The interval "
        "is a 95\\% bootstrap interval over events and seeds, over events only for one-seed systems; the "
        "jackknife column gives the range of $A^P$ when each event is left out in turn. ep.: training epochs; "
        "large sample: 80\\% of the corpus from the training events; LR: logistic regression.}",
        "\\label{tab:note-results}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{@{}lcrrcccc@{}}", "\\toprule",
        "System & Seeds & Per event & Pooled & $A^P$ [95\\% interval] & Jackknife & $A^D$ & $A^O$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_terms(x11):
    rows = []
    for corpus in CORPORA:
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        for sid, r in systems(x11, corpus):
            t = r["terms"]
            rows.append(f"{SNAME[sid]} & {f3(r['A']['P']['mean'])} & " +
                        " & ".join(stk(t[k]) for k in TERMS[:4]) + " & " +
                        " & ".join(f3(t[k]["mean"]) for k in TERMS[4:]) + " \\\\[1pt]")
        if corpus == CORPORA[0]:
            rows.append("\\midrule")
    return "\n".join([
        "\\begin{table}[tbp]", "\\centering",
        "\\caption{Split of $A^P$ (seed means, with 95\\% bootstrap intervals for $T_1$ to $T_4$): $T_1$, false "
        "positives in events that lack the class; $T_2$, non-additivity of F1 over events; $T_3$, support "
        "weighting; $T_4$, class composition (Section~\\ref{sec:method}). $T_2'$ and $T_3'$ are the "
        "non-additivity and support terms when the two steps are taken in the other order.}",
        "\\label{tab:note-terms}", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{@{}lrccccrr@{}}", "\\toprule",
        "System & $A^P$ & $T_1$ & $T_2$ & $T_3$ & $T_4$ & $T_2'$ & $T_3'$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


COLOURS = {"D2": "#1f77b4", "R2": "#d62728", "D4": "#17becf", "R4": "#ff7f0e", "T": "#2ca02c", "L": "#9467bd",
           "Z": "#7f7f7f"}


def fig_cells(x9, x11, path):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, corpus in zip(axes[:2], CORPORA):
        cells = x9["cells"][corpus]["roberta-base"]
        s = np.array([c["support"] for c in cells])
        f = np.array([c["f1"] for c in cells])
        small = s < mpa.MIN_CELL
        ax.scatter(s[~small], f[~small], s=8, color="#1b6ca8", alpha=0.7, linewidths=0)
        ax.scatter(s[small], f[small], s=13, color="#c0392b", marker="^", linewidths=0)
        ax.axvline(mpa.MIN_CELL, color="0.4", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.0)
        ax.set_title(f"({'ab'[CORPORA.index(corpus)]}) {CNAME[CORP[corpus]]} cells", fontsize=8)
        ax.set_xlabel("messages of the class in the event", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("per-class F1", fontsize=7)
    ax = axes[2]
    xs = [0] + [int(t) for t in THRESHOLDS]
    for corpus, ls in (("crisislext26", "-"), ("humaid19", "--")):
        for sid, r in systems(x11, corpus):
            ys = [r["A"]["P"]["mean"]] + [r["thresholds"][t]["mean"] for t in THRESHOLDS]
            ax.plot(xs, ys, ls=ls, color=COLOURS[sid], lw=0.9, marker="o", ms=2)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(xs)
    ax.set_title("(c) $A^P$ without small cells", fontsize=8)
    ax.set_xlabel("threshold $t$ (messages per cell)", fontsize=7)
    ax.set_ylabel("$A^P$", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=COLOURS[s], lw=1.2, label=SNAME[s]) for s in COLOURS]
    handles += [plt.Line2D([], [], color="0.2", ls="-", lw=0.9, label="CrisisLexT26 (c)"),
                plt.Line2D([], [], color="0.2", ls="--", lw=0.9, label="HumAID (c)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6, frameon=False, handlelength=2.2,
               columnspacing=1.2)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.17, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def usage_example():
    """Verbatim call of the evaluator on the seed-42 TF-IDF predictions of CrisisLexT26, and its output."""
    import subprocess
    import tempfile
    args = ["--corpus", "crisislext26", "--system", "tfidf-lr", "--seed", "42", "--convention", "P"]
    with tempfile.TemporaryDirectory() as tmp:
        stdout = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "companion_eval.py"), *args,
                                 "--out", os.path.join(tmp, "out")],
                                capture_output=True, text=True, check=True).stdout
        stdout = stdout.replace(os.path.join(tmp, "out"), "out")
    return "\n".join(["\\begin{verbatim}",
                      "$ python code/companion_eval.py --corpus crisislext26 --system tfidf-lr \\",
                      "    --seed 42 --convention P --out out",
                      stdout.rstrip("\n"), "\\end{verbatim}"])


def stk(blk):
    return f"\\stk{{{f3(blk['mean'])}}}{{[{f3(blk['ci95'][0])}, {f3(blk['ci95'][1])}]}}"


def esm_table(label, caption, colspec, header, corpus_rows, ncols, size="\\scriptsize"):
    rows = []
    for corpus in CORPORA:
        rows.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        rows += corpus_rows(corpus)
        if corpus == CORPORA[0]:
            rows.append("\\midrule")
    return "\n".join(["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      size, "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
                      header, "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def esm_conventions(x11):
    def rows(corpus):
        out = []
        for sid, r in systems(x11, corpus):
            cfm = r["closed_forms"]
            out.append(f"{SNAME[sid]} & " + " & ".join(f"{f3(r['per_event'][cv])} & {f3(r['pooled'][cv])}"
                                                       for cv in ("P", "D", "O")) +
                       f" & {stk(r['A']['D'])} & {stk(r['A']['O'])} & {f2(cfm['k_mean'])} & {cfm['k_max']} & "
                       f"{f3(cfm['shift_D'])} & {f3(cfm['shift_O'])} \\\\")
        return out
    return esm_table(
        "tab:esm-conventions",
        "Per-event mean and pooled macro-F1 under each label list, $A^D$ and $A^O$ with 95\\% bootstrap intervals, "
        "the mean and maximum over events and seeds of $k_e$ (absent classes predicted at least once in the event), "
        "and the closed-form shifts $A^D-A^P$ and $A^O-A^P$ of Eq.~(2) in the note.",
        "@{}lrrrrrrccrrrr@{}",
        "System & \\multicolumn{2}{c}{P} & \\multicolumn{2}{c}{D} & \\multicolumn{2}{c}{O} & $A^D$ & $A^O$ & "
        "\\multicolumn{2}{c}{$k_e$} & \\multicolumn{2}{c}{shift} \\\\\n & per & pool & per & pool & per & pool & & & "
        "mean & max & D & O \\\\", rows, 13)


def esm_terms(x11):
    def rows(corpus):
        return [f"{SNAME[sid]} & " + " & ".join(stk(r["terms"][k]) for k in TERMS) + " \\\\"
                for sid, r in systems(x11, corpus)]
    return esm_table(
        "tab:esm-terms", "The split of $A^P$ with 95\\% bootstrap intervals, from the same draws as the interval "
        "of $A^P$ in Table~\\ref{tab:note-results} of the note.",
        "@{}lcccccc@{}", "System & $T_1$ & $T_2$ & $T_3$ & $T_4$ & $T_2'$ & $T_3'$ \\\\", rows, 7)


def esm_thresholds(x11):
    def rows(corpus):
        out = [f"{SNAME[sid]} & {stk(r['A']['P'])} & " + " & ".join(stk(r["thresholds"][t]) for t in THRESHOLDS)
               + " \\\\" for sid, r in systems(x11, corpus)]
        th = x11["thresholds"][corpus]
        out.append("Cells (messages) removed & 0 & " + " & ".join(f"{th[t]['cells']} ({num(th[t]['messages'])})"
                                                               for t in THRESHOLDS) + " \\\\")
        return out
    return esm_table(
        "tab:esm-thresholds",
        "$A^P$ after removing every message whose event--class cell holds fewer than $t$ messages, with 95\\% "
        "bootstrap intervals.", "@{}lcccccc@{}",
        "System & all cells & $t=5$ & $t=10$ & $t=15$ & $t=20$ & $t=30$ \\\\", rows, 7)


def esm_small(x11):
    cells = {}
    for s in x11["small_cells"]["crisislext26"]:
        cells.setdefault((s["class"], s["support"], s["event"]), {})[mpa.MODEL[s["model"]]] = s["f1"]
    body = [f"{mpa.ev(e)} & {k.replace('_', ' ')} & {n} & {f2(v['D'])} & {f2(v['R'])} \\\\"
            for (k, n, e), v in sorted(cells.items())]
    return "\n".join([
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{The CrisisLexT26 event--class cells with fewer than 15 messages and their seed-mean per-class F1 "
        "for the two-epoch DistilBERT (D) and RoBERTa-base (R) systems. HumAID has no such cell.}",
        "\\label{tab:esm-small}", "\\scriptsize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
        "Event & Class & Messages & D & R \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def esm_withinclass(x11, x12):
    body = []
    rho = lambda v: "--" if v is None else f2(v)  # noqa: E731
    for corpus in CORPORA:
        by = {mpa.MODEL[w["model"]]: w for w in x11["within_class"] if w["corpus"] == corpus}
        pr = {mpa.MODEL[w["model"]]: w for w in x12["precision_recall"] if w["corpus"] == corpus}
        body.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        for k in sorted(by["D"]["per_class"]):
            cells = [rho(by[m]["per_class"][k]["rho"]) for m in "DR"]
            cells += [rho(pr[m]["per_class"][k][lab]) for lab in ("precision", "recall") for m in "DR"]
            body.append(f"{k.replace('_', ' ')} & {by['D']['per_class'][k]['n_cells']} & " + " & ".join(cells) + " \\\\")
        med = [f2(by[m]["median_rho"]) for m in "DR"]
        med += [f2(pr[m][f"median_rho_{lab}"]) for lab in ("precision", "recall") for m in "DR"]
        body.append("median over classes & & " + " & ".join(med) + " \\\\")
        if corpus == CORPORA[0]:
            body.append("\\midrule")
    return "\n".join([
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Within-class Spearman correlation between a class's support in an event and its seed-mean "
        "per-class F1, precision and recall there, over the events that contain the class, for the two-epoch "
        "DistilBERT (D) and RoBERTa-base (R) systems. Precision is averaged over the seeds that predict the class "
        "in the event. --: undefined because every cell has the same value or fewer than three cells are defined.}",
        "\\label{tab:esm-withinclass}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
        "& & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{precision} & \\multicolumn{2}{c}{recall} \\\\",
        "Class & Events & D & R & D & R & D & R \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
        "\\end{table}"])


def esm_pairs(x12):
    body = []
    for corpus in CORPORA:
        body.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        by = {cv: {(sid_of(r["i"]), sid_of(r["j"])): r for r in x12["pairs"][corpus][cv]["pairs"]} for cv in "PD"}
        for (a, b) in by["P"]:
            cells = []
            for cv in "PD":
                r = by[cv][a, b]
                cells.append(f"{f3(r['gap_pooled'])} & {f3(r['gap_per_event'])} & "
                             f"{f3(r['dA'])} [{f3(r['dA_ci95'][0])}, {f3(r['dA_ci95'][1])}] & "
                             f"{pct(r['share_draws_orders_disagree'])}")
            body.append(f"{a}--{b} & " + " & ".join(cells) + " \\\\")
        if corpus == CORPORA[0]:
            body.append("\\midrule")
    return "\n".join([
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Paired comparison of every two reference systems under the present-class (P) and scikit-learn "
        "default (D) label lists: difference in pooled and in per-event macro-F1 (first minus second system), "
        "$\\Delta A$ = difference in $A$ with a 95\\% paired bootstrap interval (events drawn once per draw for "
        "all systems, seeds drawn for each system), and the percentage of draws in which the two scores order the "
        "pair differently. D2, R2: DistilBERT and RoBERTa-base, 2 epochs; D4, R4: 4 epochs; T: TF-IDF + LR; "
        "L: DistilBERT, large sample; Z: BART-MNLI zero-shot.}",
        "\\label{tab:esm-pairs}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lrrcrrrcr@{}}", "\\toprule",
        "& \\multicolumn{4}{c}{P} & \\multicolumn{4}{c}{D} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
        "Pair & pooled & per event & $\\Delta A$ [95\\%] & \\% & pooled & per event & $\\Delta A$ [95\\%] & \\% \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def esm_bias(x12):
    def rows(corpus):
        out = []
        for r in x12["bias"]:
            if r["corpus"] == corpus:
                p, q = r["percentile_ci95"], r["bca"]["ci95"]
                out.append(f"{SNAME[sid_of(r['system'])]} & {f3(r['A_P'])} & {f3(r['boot_mean'])} & "
                           f"{f3(r['bias'])} & [{f3(p[0])}, {f3(p[1])}] & [{f3(q[0])}, {f3(q[1])}] & "
                           f"{f2(r['bca']['z0'])} & {f3(r['bca']['acceleration'])} \\\\")
        return out
    return esm_table(
        "tab:esm-bias",
        "Bootstrap bias of $A^P$ (mean of the 2{,}000 draws minus the estimate), the percentile interval of "
        "Table~\\ref{tab:note-results} of the note and the bias-corrected and accelerated (BCa) interval, with "
        "the bias correction $z_0$ and the acceleration $a$ (from the event jackknife).",
        "@{}lrrrccrr@{}",
        "System & $A^P$ & draw mean & bias & percentile & BCa & $z_0$ & $a$ \\\\", rows, 8)


def esm_withindup(x12):
    def rows(corpus):
        return [f"{SNAME[sid_of(s['system'])]} & {f3(s['A_P_all'])} & {f3(s['A_P'])} [{f3(s['ci95'][0])}, "
                f"{f3(s['ci95'][1])}] & {f3(s['A_P'] - s['A_P_all'])} \\\\"
                for s in x12["within_event_duplicates"][corpus]["systems"]]
    n = {CORP[c]: num(v["dropped"]) for c, v in x12["within_event_duplicates"].items()}
    return esm_table(
        "tab:esm-withindup",
        f"$A^P$ of all messages and with within-event duplicate texts collapsed to the message with the smallest "
        f"identifier ({n['C']} CrisisLexT26 and {n['H']} HumAID messages left out), with a 95\\% bootstrap "
        f"interval, and the change.",
        "@{}lrcr@{}", "System & all messages & collapsed [95\\%] & change \\\\", rows, 4)


def esm_zero(x12):
    body = []
    for z in sorted(x12["zero_cells"], key=lambda z: (z["corpus"], z["model"], z["class"], z["event"])):
        body.append(f"{CNAME[CORP[z['corpus']]]} & {mpa.MODEL[z['model']]} & {mpa.ev(z['event'])} & "
                    f"{z['class'].replace('_', ' ')} & {num(z['support'])} & {f2(z['f1'])} & "
                    f"{z['top_prediction'].replace('_', ' ')} ({pct(z['top_prediction_share'])}\\%) & "
                    f"{z['train_sample_n_mean']:.1f} & {num(z['train_events_n'])} \\\\")
    return "\n".join([
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Event--class cells with at least 50 messages and seed-mean per-class F1 below 0.05 for the "
        "two-epoch DistilBERT (D) and RoBERTa-base (R) systems: the most frequent predicted class of the cell's "
        "messages with its share, the mean over seeds of the class's messages in the capped training sample, and "
        "the class's messages in all training events.}",
        "\\label{tab:esm-zero}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lllp{2.6cm}rrp{3.0cm}rr@{}}", "\\toprule",
        "Corpus & & Event & Class & Messages & F1 & Most frequent prediction & Sample & Training \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def esm_jackknife(x11):
    def rows(corpus):
        out = []
        for sid, r in systems(x11, corpus):
            j, q = r["jackknife"], r["A_P_boot_quantiles"]
            out.append(f"{SNAME[sid]} & {f3(r['A']['P']['mean'])} & {f3(j['min'])} & {f3(j['max'])} & "
                       f"{mpa.ev(j['most_influential_event'])} & " + " & ".join(f3(x) for x in q) +
                       f" & {100 * r['share_draws_missing_a_class']:.1f} \\\\")
        return out
    return esm_table(
        "tab:esm-jackknife",
        "Event jackknife and bootstrap distribution of $A^P$: range when each event is left out, the event whose "
        "removal moves $A^P$ most, the 2.5, 25, 50, 75 and 97.5 percentiles of the 2{,}000 bootstrap draws, and "
        "the percentage of draws in which some class is absent from every drawn event.",
        "@{}lrrrlrrrrrr@{}",
        "System & $A^P$ & min & max & most influential event & 2.5 & 25 & 50 & 75 & 97.5 & \\% \\\\", rows, 11)


def esm_seeds(x9):
    rows = []
    for corpus in CORPORA:
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        for a in x9["arms"]:
            if a["corpus"] == corpus and len(a["seeds"]) > 1:
                ps = a["per_seed_A"]
                rows.append(f"{SNAME[SID[a['arm'], a['model']]]} & {', '.join(sorted(ps, key=int))} & "
                            f"{', '.join(f3(ps[s]) for s in sorted(ps, key=int))} \\\\")
        if corpus == CORPORA[0]:
            rows.append("\\midrule")
    return "\n".join([
        "\\begin{table}[htbp]", "\\centering", "\\caption{$A^P$ of each seed for the systems with several seeds.}",
        "\\label{tab:esm-seeds}", "\\scriptsize", "\\begin{tabular}{@{}lll@{}}", "\\toprule",
        "System & Seeds & $A^P$ per seed \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def tables(x9, x10, x11, x12):
    return {"note_results": table_results(x11), "note_terms": table_terms(x11),
            "tab_note_events": mpa.table_note_events(x9), "note_usage": usage_example(),
            "esm_conventions": esm_conventions(x11), "esm_terms": esm_terms(x11),
            "esm_thresholds": esm_thresholds(x11), "esm_small": esm_small(x11),
            "esm_withinclass": esm_withinclass(x11, x12), "esm_jackknife": esm_jackknife(x11),
            "esm_seeds": esm_seeds(x9), "esm_pairs": esm_pairs(x12), "esm_bias": esm_bias(x12),
            "esm_withindup": esm_withindup(x12), "esm_zero": esm_zero(x12)}


def current():
    x9, x10, x11, ids, x12 = load()
    out = keys(x9, x10, x11, ids, x12)
    out.update({f"table:{k}": v for k, v in tables(x9, x10, x11, x12).items()})
    return out


def main(out="paper"):
    x9, x10, x11, ids, x12 = load()
    gen = os.path.join(out, "generated")
    for name, body in tables(x9, x10, x11, x12).items():
        open(os.path.join(gen, name + ".tex"), "w").write(body + "\n")
    fig_cells(x9, x11, os.path.join(out, "figures", "fig_note_cells.pdf"))
    return keys(x9, x10, x11, ids, x12)


if __name__ == "__main__":
    main()
