"""Generate every number, table and figure the manuscript prints from the amended analysis (A2).

    python code/make_paper_assets.py
Inputs (defaults): experiments/derived/summary_v2_confirm.json, summary_v2_confirm_ep4.json,
diagnostics_confirm.json, data_facts.json, summary_confirm.json, run_stats.json, experiments/derived/convergence/.
Writes paper/generated/results.tex (\\res{key}), tab_*.tex, and paper/figures/fig_*.pdf.
Key scheme: <C>-<M>-<subset>-<conv>-<estimand>[suffix]; C in {H, C}; M in {D, R}; subset in {all, prot, dev};
conv in {P, D, O}; suffix -lo/-hi (crossed bootstrap 95%), -tlo/-thi (t 95%), -tlo90/-thi90 (t 90%).
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

CORP = {"humaid19": "H", "crisislext26": "C"}
CNAME = {"H": "HumAID-19", "C": "CrisisLexT26"}
MODEL = {"distilbert-base-uncased": "D", "roberta-base": "R"}
MNAME = {"D": "DistilBERT", "R": "RoBERTa-base"}
SUB = {"all": "all", "protected": "prot", "development": "dev"}
MARGIN = 0.02


def f3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.3f}".replace("-0.000", "0.000").replace("-", "\\textminus ")


def f3z(x):
    return f3(x).replace("0.", ".", 1)


def f4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.4f}".replace("-0.0000", "0.0000").replace("-", "\\textminus ")


def f2(x):
    return f"{x:.2f}".replace("-", "\\textminus ")


def ev(name):
    s = name.replace("_", " ").replace("aug ", "").title()
    for a, b in (("Us ", "US "), (" La ", " LA "), (" Ny ", " NY "), ("Philipinnes", "Philippines"), ("Srilanka", "Sri Lanka")):
        s = s.replace(a, b)
    return s


def pval(p):
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def tci(x, level):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    h = tdist.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def keyvals(summary, diags, ep4, facts=None, conv_dir=None, registered=None):
    kv = {}
    for r in summary:
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        kv[f"{c}-{m}-nseeds"] = str(len(r["seeds"]))
        for sub, blk in r["subsets"].items():
            s = SUB[sub]
            for conv in ("P", "D", "O"):
                base = f"{c}-{m}-{s}-{conv}"
                for est, v in blk[conv].items():
                    if isinstance(v, dict) and "mean" in v and not est.startswith("t_"):
                        kv[f"{base}-{est}"] = f3(v["mean"])
                        if "ci95" in v:
                            kv[f"{base}-{est}-lo"], kv[f"{base}-{est}-hi"] = f3(v["ci95"][0]), f3(v["ci95"][1])
                for est, v in blk[conv].get("t_rules", {}).items():
                    if v.get("mean") is None:
                        continue
                    kv[f"{base}-{est}-t"] = f3(v["mean"])
                    kv[f"{base}-{est}-tlo"], kv[f"{base}-{est}-thi"] = f3(v["t95"][0]), f3(v["t95"][1])
                    if "t90" in v:
                        kv[f"{base}-{est}-tlo90"], kv[f"{base}-{est}-thi90"] = f3(v["t90"][0]), f3(v["t90"][1])
                    kv[f"{base}-{est}-n"] = str(v.get("n_events", ""))
                if "E2_events_positive" in blk[conv]:
                    kv[f"{base}-E2-pos"] = str(blk[conv]["E2_events_positive"])
                    kv[f"{base}-E2-nev"] = str(blk[conv]["E2_n_events"])
                for k in ("E2_tweet_weighted", "E4_tweet_weighted"):
                    if k in blk[conv]:
                        kv[f"{base}-{k}"] = f3(blk[conv][k])
                loo = blk[conv].get("E2_leave_one_event_out")
                if loo:
                    kv[f"{base}-E2-loo-min"], kv[f"{base}-E2-loo-max"] = f3(min(loo.values())), f3(max(loo.values()))
            kv[f"{c}-{m}-{s}-MC-affected"] = str(blk["MC_events_affected"])
            kv[f"{c}-{m}-{s}-MC-share"] = f"{100 * blk['MC_share_of_E0D']:.0f}" if blk["MC_share_of_E0D"] is not None else "--"
            kv[f"{c}-{m}-{s}-four-resid"] = f"{abs(blk['four_term_residual']):.0e}"
            rk = blk["ranking_D_vs_P"]
            kv[f"{c}-{m}-{s}-tau"] = f2(rk['kendall_tau'])
            kv[f"{c}-{m}-{s}-moved3"] = str(rk["events_moving_ge3_ranks"])
            kv[f"{c}-{m}-{s}-hardestD"], kv[f"{c}-{m}-{s}-hardestP"] = ev(rk["hardest_D"]), ev(rk["hardest_P"])
            hz = blk["hazard_ranking_D_vs_P"]
            kv[f"{c}-{m}-{s}-hazD"], kv[f"{c}-{m}-{s}-hazP"] = hz["hardest_D"].replace("_", " "), hz["hardest_P"].replace("_", " ")
            kv[f"{c}-{m}-{s}-dose-rho"] = f2(blk['dose_vs_E2_spearman']['rho'])
            kv[f"{c}-{m}-{s}-stray"] = f"{blk['stray_preds_per_event_mean']:.1f}"
            e0d = blk["D"]["E0"]["mean"]
            for est in ("E1", "E2", "E3", "E2p", "E2_sym", "E1_loeo"):
                kv[f"{c}-{m}-{s}-share{est}"] = f"{100 * blk['P'][est]['mean'] / e0d:.0f}"
            kv[f"{c}-{m}-{s}-nstrayev"] = str(blk["MC_events_affected"])
            for lab, v in blk.get("headline", {}).items():
                pct = lab.startswith("share")
                fmt = (lambda x: f"{100 * x:.0f}".replace("-", "\\textminus ")) if pct else f3
                kv[f"{c}-{m}-{s}-hl-{lab}"], kv[f"{c}-{m}-{s}-hl-{lab}-lo"], kv[f"{c}-{m}-{s}-hl-{lab}-hi"] = \
                    fmt(v["mean"]), fmt(v["ci95"][0]), fmt(v["ci95"][1])
            if "E5_hazard_tvd_mean" in blk:
                kv[f"{c}-{m}-{s}-tvd"] = f"{blk['E5_hazard_tvd_mean']:.2f}"
                kv[f"{c}-{m}-{s}-tvd-rho"] = f2(blk['E5_vs_tvd_spearman']['rho'])
                kv[f"{c}-{m}-{s}-tvd-p"] = pval(blk['E5_vs_tvd_spearman']['p'])
                kv[f"{c}-{m}-{s}-tvd-p3"] = f"{blk['E5_vs_tvd_spearman']['p']:.3f}"
        for fp in r.get("full_pool", []):
            for conv in ("P", "D"):
                for est in ("E0", "E1", "E2", "E3", "pooled_id", "mean_loeo_whole"):
                    kv[f"{c}-{m}-full-{conv}-{est}"] = f3(fp[conv][est])
                kv[f"{c}-{m}-full-{conv}-E2-tlo"], kv[f"{c}-{m}-full-{conv}-E2-thi"] = f3(fp[conv]["E2_t95"][0]), f3(fp[conv]["E2_t95"][1])
            kv[f"{c}-{m}-full-MC"] = f3(fp["MC"])
            kv[f"{c}-{m}-full-n"] = f"{fp['n_train']:,}".replace(",", "{,}")
    # encoder comparison under D vs P and conventional vs matched (per corpus, all events)
    for corpus in CORP:
        rs = {MODEL[r["model"]]: r for r in summary if r["corpus"] == corpus}
        if set(rs) >= {"D", "R"}:
            c = CORP[corpus]
            A = {m: rs[m]["subsets"]["all"] for m in rs}
            kv[f"{c}-diff-loeoD"] = f3(A["R"]["D"]["mean_c"]["mean"] - A["D"]["D"]["mean_c"]["mean"])
            kv[f"{c}-diff-loeoP"] = f3(A["R"]["P"]["mean_c"]["mean"] - A["D"]["P"]["mean_c"]["mean"])
            kv[f"{c}-diff-E0D"] = f3(A["R"]["D"]["E0"]["mean"] - A["D"]["D"]["E0"]["mean"])
            kv[f"{c}-diff-E2P"] = f3(A["R"]["P"]["E2"]["mean"] - A["D"]["P"]["E2"]["mean"])
            # paired intervals: E0^D over seeds shared by both encoders, E2^P over events (seed-averaged contributions)
            sv = {m: {x["seed"]: x["E0"] for x in A[m]["D"]["seed_values"]} for m in A}
            common = sorted(set(sv["D"]) & set(sv["R"]), key=int)
            _, lo, hi = tci([sv["R"][s] - sv["D"][s] for s in common], 0.95)
            kv[f"{c}-diff-E0D-tlo"], kv[f"{c}-diff-E0D-thi"] = f3(lo), f3(hi)
            ec = {m: A[m]["P"]["event_contrib"]["E2"] for m in A}
            d2 = [ec["R"][e] - ec["D"][e] for e in ec["D"]]
            _, lo, hi = tci(d2, 0.95)
            _, lo9, hi9 = tci(d2, 0.90)
            kv[f"{c}-diff-E2P-tlo"], kv[f"{c}-diff-E2P-thi"] = f3(lo), f3(hi)
            kv[f"{c}-diff-E2P-tlo90"], kv[f"{c}-diff-E2P-thi90"] = f3(lo9), f3(hi9)
            kv[f"{c}-diff-E2P-equiv"] = "yes" if (lo9 > -MARGIN and hi9 < MARGIN) else "no"
            kv[f"{c}-diff-E2P-nev"] = str(len(d2))
            if all("protected" in rs[m]["subsets"] for m in rs):
                ecp = {m: rs[m]["subsets"]["protected"]["P"]["event_contrib"]["E2"] for m in rs}
                d2p = [ecp["R"][e] - ecp["D"][e] for e in ecp["D"]]
                mp, lo9, hi9 = tci(d2p, 0.90)
                kv[f"{c}-diff-E2P-prot"], kv[f"{c}-diff-E2P-prot-nev"] = f3(mp), str(len(d2p))
                kv[f"{c}-diff-E2P-prot-tlo90"], kv[f"{c}-diff-E2P-prot-thi90"] = f3(lo9), f3(hi9)
                kv[f"{c}-diff-E2P-prot-equiv"] = "yes" if (lo9 > -MARGIN and hi9 < MARGIN) else "no"
    # masking: E0 uses the pooled ID score, so its change is d(pooled ID) - d(mean LOEO) exactly
    for r in summary:
        A, c, m = r["subsets"]["all"], CORP[r["corpus"]], MODEL[r["model"]]
        if "mask_dE0" in A["D"]:
            g = lambda conv, k: A[conv][k]["mean"]
            kv[f"{c}-{m}-all-D-mask_dIDpool"] = f3(g("D", "mask_dE0") + g("D", "mask_LOEO_benefit_whole"))
            kv[f"{c}-{m}-all-P-mask_dE1"] = f3(g("P", "mask_dE0") + g("P", "mask_LOEO_benefit_whole")
                                               - g("P", "mask_ID_benefit_R"))
            for conv in ("P", "D"):
                for sv in A[conv].get("mask_seed_values", []):
                    for k in ("LOEO_benefit_whole", "dE0", "dE2"):
                        kv[f"{c}-{m}-all-{conv}-mask_{k}-s{sv['seed']}"] = f3(sv[k])
    drops = [sv["LOEO_benefit_whole"] for r in summary for sv in r["subsets"]["all"]["P"].get("mask_seed_values", [])]
    if drops:
        kv["mask-loeoP-npos"] = str(sum(x >= 0 for x in drops))
        kv["mask-loeoP-drop-min"], kv["mask-loeoP-drop-max"] = f3(-max(drops)), f3(-min(drops))
    for r in summary:
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        if "protected" in r["subsets"]:
            b = r["subsets"]["protected"]
            kv[f"{c}-{m}-prot-ratioE0E2"] = f"{b['D']['E0']['mean'] / b['P']['E2']['mean']:.1f}"
            kv[f"{c}-{m}-prot-ratioE0E2p"] = f"{b['D']['E0']['mean'] / b['P']['E2p']['mean']:.1f}"
    for est in ("E2", "E2p", "E2_sym"):
        sh = [r["subsets"]["all"]["P"][est]["mean"] / r["subsets"]["all"]["D"]["E0"]["mean"] for r in summary]
        kv[f"share{est}-min"], kv[f"share{est}-max"] = f"{100 * min(sh):.0f}", f"{100 * max(sh):.0f}"
    ratio = [r["subsets"]["all"]["D"]["E0"]["mean"] / r["subsets"]["all"]["P"]["E2"]["mean"] for r in summary]
    kv["ratioE0E2-min"], kv["ratioE0E2-max"] = f"{min(ratio):.1f}", f"{max(ratio):.1f}"
    # H3 pooled over protected HumAID and CrisisLexT26 events with future share >= 0.25
    for m in ("D", "R"):
        vals, plac, n_from = [], [], {}
        for r in summary:
            if MODEL[r["model"]] != m:
                continue
            sub = "protected" if r["corpus"] == "humaid19" else "all"
            if sub not in r["subsets"]:
                continue
            ec = r["subsets"][sub]["P"]["event_contrib"]
            for e, v in ec["E5"].items():
                fs = ec["future_share"][e]
                (vals if fs >= 0.25 else plac if fs < 0.05 else []).append(v)
                n_from[r["corpus"]] = n_from.get(r["corpus"], 0) + int(fs >= 0.25)
        kv[f"H3-{m}-nprot"], kv[f"H3-{m}-nclx"] = str(n_from.get("humaid19", 0)), str(n_from.get("crisislext26", 0))
        mu, lo, hi = tci(vals, 0.90)
        kv[f"H3-{m}-mean"], kv[f"H3-{m}-tlo90"], kv[f"H3-{m}-thi90"] = f3(mu), f3(lo), f3(hi)
        kv[f"H3-{m}-pos"] = str(int(np.sum(np.asarray(vals) > 0)))
        kv[f"H3-{m}-placebo-min"] = f3(float(np.min(plac))) if plac else "--"
        kv[f"H3-{m}-placebo-max"] = f3(float(np.max(plac))) if plac else "--"
        kv[f"H3-{m}-n"] = str(len(vals))
        kv[f"H3-{m}-pass"] = "yes" if (not np.isnan(lo) and lo > -MARGIN and hi < MARGIN) else "no"
        kv[f"H3-{m}-placebo"] = f3(float(np.mean(plac))) if plac else "--"
        kv[f"H3-{m}-nplacebo"] = str(len(plac))
    for d in diags or []:
        if d["model"] not in MODEL:
            continue
        c, m = CORP[d["corpus"]], MODEL[d["model"]]
        pm = d["prior_match_summary"]
        kv[f"{c}-{m}-pm-E2"], kv[f"{c}-{m}-pm-E2after"] = f3(pm["E2"]), f3(pm["E2_after_prior_match"])
        kv[f"{c}-{m}-pm-share"] = f"{100 * pm['share_closed']:.0f}" if pm["share_closed"] is not None else "--"
        kv[f"{c}-{m}-rare-E2"] = f3(d["rare_summary"]["E2_classes_ge5"])
        for k, v in d["stray_summary"].items():
            kv[f"{c}-{m}-stray-{k}"] = f"{v:.2f}"
        nseeds = next(len(r["seeds"]) for r in summary if r["corpus"] == d["corpus"] and r["model"] == d["model"])
        for rec in d["mc_by_absent"]:
            kv[f"{c}-{m}-mcabs{rec['n_absent_classes']}"] = f3(rec["mean"])
            kv[f"{c}-{m}-mcabs{rec['n_absent_classes']}-n"] = str(rec["count"])
            kv[f"{c}-{m}-mcabs{rec['n_absent_classes']}-nev"] = str(rec["count"] // nseeds)
        for st in d["strata"]:
            tag = {0.0: "low", 0.5: "mid", 0.8: "high"}[st["stratum"][0]]
            kv[f"{c}-{m}-sim-{tag}"] = f3(st["pooled_E2_P_mean_over_seeds"])
            kv[f"{c}-{m}-sim-{tag}-n"] = str(st["n_tweets_per_seed"])
    conv_files = sorted(glob.glob(os.path.join(conv_dir or "", "*", "*", "*.json")))
    curves = [json.load(open(f))["curve"] for f in conv_files]
    for model, m in MODEL.items():
        # X6c: epoch with the highest mean source-validation macro-F1 over this encoder's folds (earliest on ties)
        mc = [c for f, c in zip(conv_files, curves) if os.path.basename(os.path.dirname(f)) == model]
        if mc:
            means = [round(float(np.mean([c[e]["val_macro_f1_present"] for c in mc])), 3) for e in range(len(mc[0]))]
            kv[f"conv-rule-{m}"] = str(1 + int(np.argmax(means)))
            for e, v in enumerate(means):
                kv[f"conv-meanf1-{m}-e{e + 1}"] = f3(v)
    if curves:
        loss_min = [1 + int(np.argmin([x["val_loss"] for x in c])) for c in curves]
        gaps = [max(x["val_macro_f1_present"] for x in c) - c[1]["val_macro_f1_present"] for c in curves]
        kv["conv-nfolds"] = str(len(curves))
        kv["conv-lossmin-ep23"] = str(sum(e in (2, 3) for e in loss_min))
        kv["conv-f1-within002"] = str(sum(g <= 0.02 for g in gaps))
        kv["conv-f1-maxgap"] = f3(max(gaps))
    for corpus, fct in (facts or {}).items():
        c = CORP[corpus]
        for k in ("n_classes", "n_events", "classes_per_event_min", "classes_per_event_max", "events_lacking_a_class",
                  "R_e_size_median", "R_e_cells_present", "pooled_R_min_class_size", "within_event_duplicate_rows"):
            kv[f"{c}-data-{k}"] = f"{fct[k]:,}".replace(",", "{,}") if k in fct else "--"
        kv[f"{c}-data-mean_absent_classes"] = f"{fct['mean_absent_classes']:.1f}"
        kv[f"{c}-data-R_e_cells_lt5_pct"] = f"{100 * fct['R_e_cells_lt5_share']:.0f}"
    for r in registered or []:
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        for sub, blk in r["conventions"]["P"].items():
            for est in ("E2", "MC", "E5"):
                v = blk.get(est)
                if not v or v.get("mean") is None:
                    continue
                base = f"reg-{c}-{m}-{SUB[sub]}-{est}"
                kv[base] = f3(v["mean"])
                kv[base + "-lo"], kv[base + "-hi"] = f3(v["ci95"][0]), f3(v["ci95"][1])
                kv[base + "-lo90"], kv[base + "-hi90"] = f3(v["ci90"][0]), f3(v["ci90"][1])
                kv[base + "-n"] = str(v.get("n_events", "--"))
    dups = [abs(r["subsets"]["all"]["P"][k]["mean"]) for r in summary for k in ("DUP", "DUP_F") if k in r["subsets"]["all"]["P"]]
    kv["dup-absmax"] = f3(max(dups))
    ratio_p = [r["subsets"]["all"]["D"]["E0"]["mean"] / r["subsets"]["all"]["P"]["E2p"]["mean"] for r in summary]
    kv["ratioE0E2p-min"], kv["ratioE0E2p-max"] = f"{min(ratio_p):.1f}", f"{max(ratio_p):.1f}"
    # four-epoch arms. ep4-*/s42-*: seed 42 at four and two epochs (the registered sensitivity criterion);
    # ep4m-*/b2m-*: mean over every four-epoch seed and the two-epoch mean over the same seeds.
    base = {(r["corpus"], r["model"]): r for r in summary}
    ep4_d, ep4m_d = [], []
    SEED_EST = ("E0", "E1", "E2", "E3", "MC", "E5", "E2p", "E4", "mean_c", "mean_a", "pooled_id")
    for r in ep4 or []:
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        seeds = r["seeds"]
        kv[f"ep4-{c}-{m}-nseeds"] = str(len(seeds))
        kv[f"ep4-{c}-{m}-seeds"] = ", ".join(sorted(seeds, key=lambda s: (s != "42", int(s))))
        for sub, blk in r["subsets"].items():
            s = SUB[sub]
            b2 = base[(r["corpus"], r["model"])]["subsets"][sub]
            for conv in ("P", "D"):
                e42 = next(x for x in blk[conv]["seed_values"] if x["seed"] == "42")
                s42 = next(x for x in b2[conv]["seed_values"] if x["seed"] == "42")
                same = [x for x in b2[conv]["seed_values"] if x["seed"] in seeds]
                for est in SEED_EST:
                    if est in e42:
                        kv[f"ep4-{c}-{m}-{s}-{conv}-{est}"] = f3(e42[est])
                    if est in s42:
                        kv[f"s42-{c}-{m}-{s}-{conv}-{est}"] = f3(s42[est])
                    if est in blk[conv]:
                        kv[f"ep4m-{c}-{m}-{s}-{conv}-{est}"] = f3(blk[conv][est]["mean"])
                    if all(est in x for x in same):
                        kv[f"b2m-{c}-{m}-{s}-{conv}-{est}"] = f3(float(np.mean([x[est] for x in same])))
            t = blk["P"]["t_rules"]["E2"]
            kv[f"ep4m-{c}-{m}-{s}-P-E2-tlo"], kv[f"ep4m-{c}-{m}-{s}-P-E2-thi"] = f3(t["t95"][0]), f3(t["t95"][1])
            kv[f"ep4m-{c}-{m}-{s}-P-E2-pos"] = str(blk["P"]["E2_events_positive"])
            e0d = blk["D"]["E0"]["mean"]
            for est, lab in (("E2", "shareE2"), ("E1", "shareE1"), ("MC", "MC-share")):
                kv[f"ep4m-{c}-{m}-{s}-{lab}"] = f"{100 * blk['P'][est]['mean'] / e0d:.0f}"
            kv[f"ep4m-{c}-{m}-{s}-ratioE0E2"] = f"{e0d / blk['P']['E2']['mean']:.1f}"
            hl = blk.get("headline", {})
            if "remainder" in hl:
                kv[f"ep4m-{c}-{m}-{s}-hl-remainder-lo"], kv[f"ep4m-{c}-{m}-{s}-hl-remainder-hi"] = \
                    f3(hl["remainder"]["ci95"][0]), f3(hl["remainder"]["ci95"][1])
            e42p = next(x for x in blk["P"]["seed_values"] if x["seed"] == "42")
            e42d = next(x for x in blk["D"]["seed_values"] if x["seed"] == "42")
            s42p = next(x for x in b2["P"]["seed_values"] if x["seed"] == "42")
            s42d = next(x for x in b2["D"]["seed_values"] if x["seed"] == "42")
            kv[f"ep4-{c}-{m}-{s}-shareE2"] = f"{100 * e42p['E2'] / e42d['E0']:.0f}"
            for est in ("E2", "MC", "E5"):
                if est in s42p and est in e42p:
                    kv[f"ep4-{c}-{m}-{s}-d{est}"] = f3(e42p[est] - s42p[est])
            kv[f"ep4-{c}-{m}-{s}-dE0D"] = f3(e42d["E0"] - s42d["E0"])
            same_p = [x for x in b2["P"]["seed_values"] if x["seed"] in seeds]
            same_d = [x for x in b2["D"]["seed_values"] if x["seed"] in seeds]
            dE0 = e0d - np.mean([x["E0"] for x in same_d])
            dE2 = blk["P"]["E2"]["mean"] - np.mean([x["E2"] for x in same_p])
            kv[f"ep4m-{c}-{m}-{s}-dE0D"], kv[f"ep4m-{c}-{m}-{s}-dE2"] = f3(dE0), f3(dE2)
            if sub == "all":
                ep4_d.append((c, e42d["E0"] - s42d["E0"], e42p["E2"] - s42p["E2"]))
                ep4m_d.append((c, dE0, dE2))
    for c in ("H", "C"):
        d = [x for x in ep4m_d if x[0] == c]
        if d:
            kv[f"ep4m-{c}-dE0D-min"], kv[f"ep4m-{c}-dE0D-max"] = f3(min(x[1] for x in d)), f3(max(x[1] for x in d))
            kv[f"ep4m-{c}-dE2-absmax"] = f3(max(abs(x[2]) for x in d))
    d = [x for x in ep4_d if x[0] == "H"]
    if d:
        kv["ep4-H-dE0D-min"], kv["ep4-H-dE0D-max"] = f3(min(x[1] for x in d)), f3(max(x[1] for x in d))
        kv["ep4-H-dE2-absmax"] = f3(max(abs(x[2]) for x in d))
    return kv


PLAN_TAG = {"confirm": "b", "confirm_ep4": "ep4", "x3_tfidf": "x3"}


def aggregation_keys(agg, summary):
    """Keys from experiments/derived/aggregation_mechanism.json (code/aggregation_mechanism.py)."""
    kv = {}
    for rec in agg or []:
        c = CORP[rec["corpus"]]
        m = MODEL.get(rec["model"], "T")
        p = f"agg-{PLAN_TAG[rec['plan']]}-{c}-{m}"
        sm = rec["summary"]
        for k in ("E1", "scoring", "reweighting", "loeo_aggregation_R", "loeo_aggregation_whole"):
            short = {"loeo_aggregation_R": "loeoR", "loeo_aggregation_whole": "loeoW"}.get(k, k)
            kv[f"{p}-{short}"] = f3(sm[k]["mean"])
            kv[f"{p}-{short}-min"], kv[f"{p}-{short}-max"] = f3(sm[k]["min"]), f3(sm[k]["max"])
        kv[f"{p}-rho"] = f2(sm["spearman_prev_f1"]["mean"])
        cells = [s["cells"] for s in rec["seeds"]]
        kv[f"{p}-lowshare"] = f"{100 * np.mean([x['share_prev_lt_5pct'] for x in cells]):.0f}"
        kv[f"{p}-lowmed"] = f2(float(np.mean([x["median_f1_prev_lt_5pct"] for x in cells])))
        kv[f"{p}-highmed"] = f2(float(np.mean([x["median_f1_prev_ge_5pct"] for x in cells])))
        if "id_spearman_prev_f1_R" in sm:
            kv[f"{p}-idrho"] = f2(sm["id_spearman_prev_f1_R"]["mean"])
            kv[f"{p}-idrho-within"] = f2(sm["id_spearman_prev_f1_R"]["within_class_mean"])
        if rec["plan"] == "confirm":
            ep4 = next((x for x in agg if x["plan"] == "confirm_ep4" and x["corpus"] == rec["corpus"]
                        and x["model"] == rec["model"]), None)
            if ep4:
                same = {s["seed"] for s in ep4["seeds"]}
                sc = [s["scoring"] for s in rec["seeds"] if s["seed"] in same]
                kv[f"agg-b2m-{c}-{m}-scoring"] = f3(float(np.mean(sc)))
        kv[f"{p}-nseeds"] = str(len(rec["seeds"]))
        kv[f"{p}-nlearned"] = str(len(sm["rarest_class_learned_seeds"]))
        for g in ("learned", "not_learned"):
            if sm["reweighting_by_rare_class"][g]:
                kv[f"{p}-rw-{g.replace('_', '')}"] = f3(sm["reweighting_by_rare_class"][g][0])
        rc = rec["seeds"][0]["rarest_class"]
        kv[f"rare-{c}-class"] = "\\texttt{" + rc["class"].replace("_", "\\_") + "}"
        kv[f"rare-{c}-nev"], kv[f"rare-{c}-nevtotal"], kv[f"rare-{c}-ntrue"] = \
            str(rc["n_events"]), str(rc["n_events_total"]), str(rc["n_true"])
        f1s = [s["rarest_class"]["pooled_id_f1"] for s in rec["seeds"] if s["rarest_class"]["n_pred"] > 0]
        if f1s:
            kv[f"{p}-rareF1-min"], kv[f"{p}-rareF1-max"] = f3(min(f1s)), f3(max(f1s))
        if rec["plan"] == "confirm" and c == "H":
            learned = set(sm["rarest_class_learned_seeds"])
            r = next(x for x in summary if x["corpus"] == rec["corpus"] and x["model"] == rec["model"])
            sv_d = r["subsets"]["all"]["D"]["seed_values"]
            sv_p = r["subsets"]["all"]["P"]["seed_values"]
            for g, sel in (("learned", lambda s: int(s) in learned), ("notlearned", lambda s: int(s) not in learned)):
                e0 = [x["E0"] for x in sv_d if sel(x["seed"])]
                e2 = [x["E2"] for x in sv_p if sel(x["seed"])]
                if e0:
                    kv[f"{p}-E0D-{g}-min"], kv[f"{p}-E0D-{g}-max"] = f3(min(e0)), f3(max(e0))
                    kv[f"{p}-E2-{g}-mean"] = f3(float(np.mean(e2)))
                    kv[f"{p}-seeds-{g}"] = ", ".join(sorted((x["seed"] for x in sv_d if sel(x["seed"])), key=int))
    return kv


def x3_keys(x3):
    """Keys for the exploratory TF-IDF + logistic-regression arm (summary_v2_x3_tfidf.json)."""
    kv = {}
    for r in x3 or []:
        c = CORP[r["corpus"]]
        kv[f"x3-{c}-nseeds"] = str(len(r["seeds"]))
        for sub, blk in r["subsets"].items():
            s = SUB[sub]
            for conv in ("P", "D"):
                for est in ("E0", "E1", "E2", "E3", "MC", "E2p", "E5", "E4", "DUP", "mean_c", "pooled_id"):
                    if est in blk[conv]:
                        kv[f"x3-{c}-{s}-{conv}-{est}"] = f3(blk[conv][est]["mean"])
            t = blk["P"]["t_rules"]["E2"]
            kv[f"x3-{c}-{s}-P-E2-tlo"], kv[f"x3-{c}-{s}-P-E2-thi"] = f3(t["t95"][0]), f3(t["t95"][1])
            e0d = blk["D"]["E0"]["mean"]
            for est, lab in (("E2", "shareE2"), ("E1", "shareE1"), ("MC", "MC-share")):
                kv[f"x3-{c}-{s}-{lab}"] = f"{100 * blk['P'][est]['mean'] / e0d:.0f}"
            kv[f"x3-{c}-{s}-ratioE0E2"] = f"{e0d / blk['P']['E2']['mean']:.1f}"
            hl = blk["headline"]["remainder"]
            kv[f"x3-{c}-{s}-hl-remainder-lo"], kv[f"x3-{c}-{s}-hl-remainder-hi"] = f3(hl["ci95"][0]), f3(hl["ci95"][1])
    return kv


PRACTICES = (("matched_event", "me"), ("matched_pooled", "mp"), ("pooled_vs_pooled_whole", "pp"),
             ("event_vs_event_whole", "ee"), ("different_events_pooled", "de"), ("composite", "co"))


def revision_keys(rev, summary, ep4, x3):
    """Keys from experiments/derived/revision_tmlr.json (code/revision_tmlr.py, preregistration X5): the gap each
    single practice yields (pr-*), seed-aware crossed intervals (re-*, reu-* unbiased), the encoder difference in
    E2 (encdiff-*), HumAID rescored with the merged class (mg-*), and the gap-to-E2 ratio across arms (arms-*)."""
    kv = {}
    if not rev:
        return kv
    for key, subs in rev["practices"].items():
        corpus, model = key.split("/")
        for sub, convs in subs.items():
            for conv, pr in convs.items():
                for name, tag in PRACTICES:
                    v, base = pr[name], f"pr-{CORP[corpus]}-{MODEL[model]}-{SUB[sub]}-{conv}-{tag}"
                    kv[base], kv[base + "-lo"], kv[base + "-hi"] = f3(v["mean"]), f3(v["ci95"][0]), f3(v["ci95"][1])
    # comparisons that change only the test tweets, against the matched contrast with the same averaging
    diffs = [abs(s["all"]["P"][x]["mean"] - s["all"]["P"][y]["mean"]) for s in rev["practices"].values()
             for x, y in (("pooled_vs_pooled_whole", "matched_pooled"), ("event_vs_event_whole", "matched_event"))]
    kv["pr-maxdiff"] = f3(max(diffs))
    # per-event comparison under the default label set, as a multiple of the per-event matched contrast (HumAID)
    ee_d = [s[sub]["D"]["event_vs_event_whole"]["mean"] / s[sub]["P"]["matched_event"]["mean"]
            for key, s in rev["practices"].items() if key.startswith("humaid19/") for sub in ("all", "protected")]
    kv["pr-eeD-ratio-min"], kv["pr-eeD-ratio-max"] = f"{min(ee_d):.1f}", f"{max(ee_d):.1f}"
    # X6a: different-events comparison against the composite; on all events each target's pool shares all but one
    # event with the composite's, on the protected events it comes from the other 18 HumAID events
    de_co = lambda sub: [s[sub][c]["different_events_pooled"]["mean"] - s[sub][c]["composite"]["mean"]
                         for s in rev["practices"].values() if sub in s for c in ("P", "D")]
    kv["pr-de-co-maxdiff"] = f3(max(abs(x) for x in de_co("all")))
    kv["pr-de-prot-excess-min"], kv["pr-de-prot-excess-max"] = f3(min(de_co("protected"))), f3(max(de_co("protected")))
    for key, subs in rev.get("mc_bootstrap", {}).items():
        corpus, model = key.split("/")
        for sub, b in subs.items():
            base = f"mcb-{CORP[corpus]}-{MODEL[model]}-{SUB[sub]}"
            kv[base + "-lo"], kv[base + "-hi"] = f3(b["pct95"][0]), f3(b["pct95"][1])
            kv[base + "-npos"], kv[base + "-n"] = str(b["n_positive"]), str(b["n_events"])
    for key, subs in rev["random_effects"].items():
        corpus, model = key.split("/")
        for sub, ests in subs.items():
            for est, r in ests.items():
                base = f"re-{CORP[corpus]}-{MODEL[model]}-{SUB[sub]}-{est}"
                t95, t90 = r["t95"], r["t90"]
                kv[base + "-lo"], kv[base + "-hi"], kv[base + "-df"] = f3(t95["lo"]), f3(t95["hi"]), f"{t95['df']:.1f}"
                kv[base + "-lo90"], kv[base + "-hi90"] = f3(t90["lo"]), f3(t90["hi"])
                u = t95["unbiased"]
                kv[f"reu-{base[3:]}-lo"], kv[f"reu-{base[3:]}-hi"], kv[f"reu-{base[3:]}-df"] = \
                    f3(u["lo"]), f3(u["hi"]), f"{u['df']:.1f}"
                by_seed = dict(zip(r["seeds"], r["per_seed_mean"]))
                others = [v for s, v in by_seed.items() if s != "42"]
                for s, v in by_seed.items():
                    kv[f"{base}-s{s}"] = f3(v)
                kv[base + "-seed42"] = f3(by_seed["42"])
                kv[base + "-seedmin"], kv[base + "-seedmax"] = f3(min(by_seed.values())), f3(max(by_seed.values()))
                kv[base + "-othermin"], kv[base + "-othermax"] = f3(min(others)), f3(max(others))
                low = min(by_seed, key=by_seed.get)
                kv[base + "-lowseed"], kv[base + "-lowval"] = low, f3(by_seed[low])
                kv[base + "-nonlowmin"] = f3(min(v for s, v in by_seed.items() if s != low))
                kv[base + "-nonlowmax"] = f3(max(v for s, v in by_seed.items() if s != low))
    for key, d in rev["encoder_difference"].items():
        corpus, sub = key.split("/")
        base = f"encdiff-{CORP[corpus]}-{SUB[sub]}"
        kv[base], kv[base + "-lo90"], kv[base + "-hi90"] = f3(d["R_minus_D"]), f3(d["re_t90"][0]), f3(d["re_t90"][1])
    for model, subs in rev["merged_humaid"].items():
        m = MODEL[model]
        for sub, b in subs.items():
            base, e0d = f"mg-{m}-{SUB[sub]}", b["D"]["E0"]["mean"]
            for conv in ("P", "D"):
                for est, v in b[conv].items():
                    if isinstance(v, dict) and "mean" in v:
                        kv[f"{base}-{conv}-{est}"] = f3(v["mean"])
            t = b["t_rules_E2"]
            kv[f"{base}-P-E2-tlo"], kv[f"{base}-P-E2-thi"] = f3(t["t95"][0]), f3(t["t95"][1])
            kv[f"{base}-shareE2"] = f"{100 * b['P']['E2']['mean'] / e0d:.0f}"
            kv[f"{base}-MC-share"] = f"{100 * b['MC_share_of_E0D']:.0f}"
            kv[f"{base}-ratioE0E2"] = f"{e0d / b['P']['E2']['mean']:.1f}"
            rem = b["headline"]["remainder"]
            kv[f"{base}-remainder-lo"], kv[f"{base}-remainder-hi"] = f3(rem["ci95"][0]), f3(rem["ci95"][1])
    # ms: arms with at least two seeds; single: one-seed arms (full pool, CrisisLexT26 four epochs)
    ms = [r["subsets"]["all"]["D"]["E0"]["mean"] / r["subsets"]["all"]["P"]["E2"]["mean"] for r in summary]
    single = [fp["D"]["E0"] / fp["P"]["E2"] for r in summary for fp in r.get("full_pool", [])]
    for arm in (ep4 or []) + (x3 or []):
        a = arm["subsets"]["all"]
        (ms if len(arm["seeds"]) > 1 else single).append(a["D"]["E0"]["mean"] / a["P"]["E2"]["mean"])
    for subs in rev["merged_humaid"].values():
        ms.append(subs["all"]["D"]["E0"]["mean"] / subs["all"]["P"]["E2"]["mean"])
    ratios = ms + single
    kv["arms-ratio-min"], kv["arms-ratio-max"] = f"{min(ratios):.1f}", f"{max(ratios):.1f}"
    kv["arms-n"] = str(len(ratios))
    kv["arms-ms-ratio-min"], kv["arms-ms-ratio-max"] = f"{min(ms):.1f}", f"{max(ms):.1f}"
    kv["arms-ms-n"], kv["arms-single-n"] = str(len(ms)), str(len(single))
    kv["arms-single-ratio-min"], kv["arms-single-ratio-max"] = f"{min(single):.1f}", f"{max(single):.1f}"
    return kv


def table_practices(rev, summary, x7):
    """Gap that each single practice yields on the same predictions (X5a), with LOEO aggregation (X7b)."""
    iv = lambda v: f"[{f3z(v['ci95'][0])},{f3z(v['ci95'][1])}]"
    rows = []
    for key in sorted(rev["practices"], key=lambda k: (not k.startswith("humaid19"), k)):
        corpus, model = key.split("/")
        for sub in ("all", "protected"):
            if sub not in rev["practices"][key]:
                continue
            P, D = rev["practices"][key][sub]["P"], rev["practices"][key][sub]["D"]
            la = x7["evaluation_terms"][key][sub]["loeo_aggregation"]
            cells = [P["matched_event"], P["matched_pooled"], P["pooled_vs_pooled_whole"], P["event_vs_event_whole"],
                     D["event_vs_event_whole"], la["P"], la["D"], P["composite"], D["composite"]]
            lab = "\\quad protected" if sub == "protected" else f"{CNAME[CORP[corpus]]}, {MODEL[model]}"
            rows.append(lab + " & " + " & ".join(f3(v["mean"]) for v in cells) + " \\\\")
            rows.append(" & " + " & ".join(iv(v) for v in cells) + " \\\\[1pt]")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{What each scoring choice yields on the same predictions (seed means over events; 95\\% crossed-bootstrap "
        "intervals below; for $E_2^P$ the registered interval is the $t$ interval of Table~\\ref{tab:hyp}). "
        "Matched: in-distribution minus LOEO model on identical tweets $R_e$, averaged per event ($E_2^P$) or pooled "
        "($E_2^{\\mathrm{pool}}$). Pooled: pooled in-distribution score on $R$ minus pooled LOEO predictions on whole "
        "held-out events. Per event: mean in-distribution score on $R_e$ minus mean whole-event LOEO score. LOEO "
        "aggregation: pooled minus per-event macro-F1 of the same whole-event LOEO predictions (no in-distribution "
        "model). Composite: pooled in-distribution score minus mean whole-event LOEO score. P, D: label-set convention "
        "(pooled scores do not depend on it). Matched, pooled, per-event and composite columns: registered estimands "
        "and X5; LOEO aggregation: X7 (all exploratory except $E_2^P$). D: DistilBERT; R: RoBERTa.}",
        "\\label{tab:practices}", "\\scriptsize", "\\setlength{\\tabcolsep}{1.4pt}",
        "\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
        " & \\multicolumn{2}{c}{Matched} & Pooled & \\multicolumn{2}{c}{Per event} & "
        "\\multicolumn{2}{c}{LOEO aggregation} & \\multicolumn{2}{c}{Composite} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-10}",
        "Corpus, encoder & $E_2^P$ & $E_2^{\\mathrm{pool}}$ & & P & D & P & D & $E_0^P$ & $E_0^D$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


DEC = {"H1": "H1", "H1b": "H1b", "H2 protected": "H2p", "H2 CrisisLexT26": "H2c", "H3": "H3"}
RECALL_TAG = {"missing_or_found_people": "mfp", "injured_or_dead_people": "idp", "requests_or_urgent_needs": "run"}


def x7_keys(x7):
    """Keys from experiments/derived/revision_x7.json (code/revision_x7.py, preregistration X7b)."""
    kv = {}
    if not x7:
        return kv
    yn = lambda b: "yes" if b else "no"
    for key, subs in x7["evaluation_terms"].items():
        corpus, model = key.split("/")
        for sub, b in subs.items():
            base = f"{CORP[corpus]}-{MODEL[model]}-{SUB[sub]}"
            for est, tag in (("DUP", "dup"), ("DUP_F", "dupf"), ("E3", "e3")):
                if est not in b:
                    continue
                r = b[est]
                kv[f"x7-{tag}-{base}"] = f3(r["t95"]["mean"])
                kv[f"x7-{tag}-{base}-lo"], kv[f"x7-{tag}-{base}-hi"] = f3(r["t95"]["lo"]), f3(r["t95"]["hi"])
                kv[f"x7-{tag}-{base}-lo90"], kv[f"x7-{tag}-{base}-hi90"] = f3(r["t90"]["lo"]), f3(r["t90"]["hi"])
                kv[f"x7-{tag}-{base}-seedmin"] = f3(min(r["per_seed_mean"]))
                kv[f"x7-{tag}-{base}-seedmax"] = f3(max(r["per_seed_mean"]))
                for m in ("0.01", "0.02"):
                    if f"tost_{m}" in r:
                        kv[f"x7-{tag}-{base}-tost{m[2:]}"] = yn(r[f"tost_{m}"]["conservative"])
            for conv, r in b["loeo_aggregation"].items():
                kv[f"x7-lagg-{base}-{conv}"] = f3(r["mean"])
                kv[f"x7-lagg-{base}-{conv}-lo"], kv[f"x7-lagg-{base}-{conv}-hi"] = f3(r["ci95"][0]), f3(r["ci95"][1])
    for row in x7["holm"]["rows"]:
        base = f"x7-holm-{DEC[row['decision']]}-{row['encoder'][0]}"
        kv[base + "-p"] = f"{row['p']:.3f}" if row["p"] >= 0.001 else "$<$0.001"
        kv[base + "-padj"] = f"{row['p_holm']:.3f}" if row["p_holm"] >= 0.001 else "$<$0.001"
        kv[base + "-holm"] = yn(row["supported_holm"])
        if "material_on_lower_bound" in row:
            kv[base + "-lbmat"] = yn(row["material_on_lower_bound"])
    kv["x7-holm-nsupported"] = str(sum(r["supported_holm"] for r in x7["holm"]["rows"]))
    for model, er in x7["epoch_rules"].items():
        m = MODEL[model]
        kv[f"x7-epoch-{m}-loss"], kv[f"x7-epoch-{m}-f1"] = str(er["loss_rule"]), str(er["macro_f1_rule"])
        for e, (l, f) in enumerate(zip(er["mean_val_loss"], er["mean_val_macro_f1"])):
            kv[f"x7-epoch-{m}-loss-e{e + 1}"], kv[f"x7-epoch-{m}-f1-e{e + 1}"] = f3(l), f3(f)
    for model, fe in x7["four_epoch"].items():
        m = MODEL[model]
        kv[f"x7-ep4-{m}-nseeds"] = str(len(fe["seeds"]))
        kv[f"x7-ep4-{m}-seeds"] = ", ".join(sorted(fe["seeds"], key=lambda s: (s != "42", int(s))))
        for sub, b in fe["subsets"].items():
            base = f"x7-ep4-{m}-{SUB[sub]}"
            for s, v in b["per_seed_E2_ep4"].items():
                kv[f"{base}-s{s}"] = f3(v)
            for s, v in b["per_seed_E2_ep2"].items():
                kv[f"x7-ep2-{m}-{SUB[sub]}-s{s}"] = f3(v)
            kv[f"{base}-seedmin"] = f3(min(b["per_seed_E2_ep4"].values()))
            kv[f"{base}-seedmax"] = f3(max(b["per_seed_E2_ep4"].values()))
            for lab in ("old", "complete"):
                e4, e2 = b[lab]["ep4"], b[lab]["ep2"]
                kv[f"{base}-{lab}-E2"], kv[f"{base}-{lab}-E2-b2"] = f3(e4["E2"]), f3(e2["E2"])
                kv[f"{base}-{lab}-tlo"], kv[f"{base}-{lab}-thi"] = f3(e4["E2_t95"][0]), f3(e4["E2_t95"][1])
                kv[f"{base}-{lab}-b2-tlo"], kv[f"{base}-{lab}-b2-thi"] = f3(e2["E2_t95"][0]), f3(e2["E2_t95"][1])
                kv[f"{base}-{lab}-pos"] = str(e4["E2_events_positive"])
                for k in ("E0D", "E1", "MC", "E3", "E2p", "mean_c"):
                    kv[f"{base}-{lab}-{k}"], kv[f"{base}-{lab}-{k}-b2"] = f3(e4[k]), f3(e2[k])
                if e4.get("E2_crossed"):
                    c = e4["E2_crossed"]
                    kv[f"{base}-{lab}-re-lo"], kv[f"{base}-{lab}-re-hi"] = f3(c["lo"]), f3(c["hi"])
                    kv[f"{base}-{lab}-reu-lo"], kv[f"{base}-{lab}-reu-hi"] = f3(c["unbiased"]["lo"]), f3(c["unbiased"]["hi"])
    for model, classes in x7["class_recall"].items():
        m = MODEL[model]
        for cls, r in classes.items():
            base = f"x7-rec-{m}-{RECALL_TAG[cls]}"
            kv[base + "-npres"], kv[base + "-supp"], kv[base + "-suppR"] = \
                str(r["events_present"]), f"{r['support_whole']:,}".replace(",", "{,}"), str(r["support_R"])
            for lab, tag in (("loeo_whole", "loeo"), ("id_R", "id")):
                x = r[lab]
                kv[f"{base}-{tag}-recP"], kv[f"{base}-{tag}-recE"] = f2(x["recall_pooled"]), f2(x["recall_event_mean"])
                kv[f"{base}-{tag}-f1P"], kv[f"{base}-{tag}-f1E"] = f2(x["f1_pooled"]), f2(x["f1_event_mean"])
                kv[f"{base}-{tag}-fa"], kv[f"{base}-{tag}-nabs"] = f"{x['false_alarms_absent_events']:.1f}", str(x["events_absent"])
    return kv


ARM_ORDER = [("2 epochs", "distilbert-base-uncased"), ("2 epochs", "roberta-base"), ("4 epochs", "distilbert-base-uncased"),
             ("4 epochs", "roberta-base"), ("uncapped", "distilbert-base-uncased"), ("TF-IDF", "tfidf-lr")]
ARM_LABEL = {"distilbert-base-uncased": "DistilBERT", "roberta-base": "RoBERTa", "tfidf-lr": "TF-IDF + LR"}


def x8_arms(x8, corpus):
    recs = {(a["arm"], a["model"]): a for a in x8["arms"] if a["corpus"] == corpus}
    return [recs[k] for k in ARM_ORDER if k in recs]


def x8_runs(x8):
    """HumAID per-run rows (two and four epochs of both encoders, TF-IDF) with labels."""
    out = []
    for model, fe in x8["four_epoch_seeds"].items():
        for s, d in fe["per_seed"].items():
            for lab, r in d.items():
                out.append({"model": model, "arm": lab, "seed": s, **r})
    return out


def x8_keys(x8):
    """Keys from experiments/derived/revision_x8.json (code/revision_x8.py, preregistration X8)."""
    kv = {}
    if not x8:
        return kv
    for model, r in x8["dedup_E2_crisislext26"].items():
        b = f"x8-dedup-C-{MODEL[model]}"
        kv[b], kv[b + "-tlo"], kv[b + "-thi"] = f3(r["E2_dedup"]), f3(r["t95"][0]), f3(r["t95"][1])
        kv[b + "-pos"], kv[b + "-E2"] = str(r["events_positive"]), f3(r["E2"])
    tag = {"CrisisLexT26": "C", "HumAID protected": "Hp", "HumAID all": "Ha"}
    for model, blk in x8["h3_per_corpus"].items():
        for lab, r in blk.items():
            b = f"x8-h3-{MODEL[model]}-{tag[lab]}"
            kv[b], kv[b + "-n"] = f3(r["mean"]), str(r["n_events"])
            kv[b + "-lo90"], kv[b + "-hi90"] = f3(r["t90"][0]), f3(r["t90"][1])
            kv[b + "-eq"] = "yes" if r["equivalent"] else "no"
    for corpus in ("crisislext26", "humaid19"):
        arms = x8_arms(x8, corpus)
        c = CORP[corpus]
        ag = [a["subsets"]["all"]["loeo_aggregation"] for a in arms]
        e2 = [a["subsets"]["all"]["E2"] for a in arms]
        kv[f"x8-arms-{c}-n"] = str(len(arms))
        kv[f"x8-arms-{c}-agg-min"], kv[f"x8-arms-{c}-agg-max"] = f3(min(x["mean"] for x in ag)), f3(max(x["mean"] for x in ag))
        kv[f"x8-arms-{c}-agglo-min"] = f3(min(x["ci95"][0] for x in ag))
        kv[f"x8-arms-{c}-agghi-max"] = f3(max(x["ci95"][1] for x in ag))
        kv[f"x8-arms-{c}-E2-min"], kv[f"x8-arms-{c}-E2-max"] = f3(min(x["mean"] for x in e2)), f3(max(x["mean"] for x in e2))
        for a in arms:
            b = f"x8-arm-{c}-{a['arm'].replace(' ', '').replace('-', '')}-{MODEL.get(a['model'], 'T')}"
            s = a["subsets"]["all"]
            kv[b + "-agg"], kv[b + "-E2"], kv[b + "-nseeds"] = f3(s["loeo_aggregation"]["mean"]), f3(s["E2"]["mean"]), str(len(a["seeds"]))
            kv[b + "-agglo"], kv[b + "-agghi"] = f3(s["loeo_aggregation"]["ci95"][0]), f3(s["loeo_aggregation"]["ci95"][1])
            kv[b + "-E2lo"], kv[b + "-E2hi"] = f3(s["E2"]["t95"][0]), f3(s["E2"]["t95"][1])
            if "protected" in a["subsets"]:
                p = a["subsets"]["protected"]["E2"]
                kv[b + "-pE2"], kv[b + "-pE2lo"], kv[b + "-pE2hi"] = f3(p["mean"]), f3(p["t95"][0]), f3(p["t95"][1])
                if p["crossed95"]:
                    cu = p["crossed95"]["unbiased"]
                    kv[b + "-pE2clo"], kv[b + "-pE2chi"] = f3(cu["lo"]), f3(cu["hi"])
    for model, fe in x8["four_epoch_seeds"].items():
        if model not in MODEL:
            continue
        m = MODEL[model]
        kv[f"x8-ep4-{m}-nseeds"], kv[f"x8-ep4-{m}-nevery"] = str(len(fe["seeds"])), str(fe["n_seeds_every_class_ep4"])
        kv[f"x8-ep2-{m}-nevery"] = str(fe["n_seeds_every_class_ep2"])
        kv[f"x8-ep2-{m}-nseeds"] = str(sum("ep2" in d for d in fe["per_seed"].values()))
        for k, (lo, hi) in fe["range_ep4"].items():
            kv[f"x8-ep4-{m}-{k}-min"], kv[f"x8-ep4-{m}-{k}-max"] = f3(lo), f3(hi)
    runs = [r for r in x8_runs(x8) if r["model"] in MODEL]
    learned = [r for r in runs if r["rare_pooled_id_f1"] >= 0.3]
    weak = [r for r in runs if 0 < r["rare_pooled_id_f1"] < 0.3]
    never = [r for r in runs if r["rare_pooled_id_f1"] == 0]
    kv["x8-runs-n"] = str(len(runs))
    for lab, grp in (("learned", learned), ("weak", weak), ("never", never)):
        kv[f"x8-runs-{lab}-n"] = str(len(grp))
        if grp:
            for k in ("E0D", "E2", "E1"):
                kv[f"x8-runs-{lab}-{k}-min"] = f3(min(r[k] for r in grp))
                kv[f"x8-runs-{lab}-{k}-max"] = f3(max(r[k] for r in grp))
    rp = x8["repeats"]
    if rp.get("labels") and len(rp["labels"]) >= 2:
        kv["x8-rep-n"] = str(len(rp["labels"]))
        kv["x8-rep-same"] = "yes" if rp["same_training_sample"] else "no"
        kv["x8-rep-E2-min"], kv["x8-rep-E2-max"] = f4(rp["range_E2_repeats"][0]), f4(rp["range_E2_repeats"][1])
        kv["x8-rep-sdE2"], kv["x8-rep-sdE2-seeds"] = f4(rp["sd_E2_repeats"]), f4(rp["sd_E2_seeds"])
        kv["x8-rep-evsdE2"], kv["x8-rep-evsdE2-seeds"] = f3(rp["mean_sd_event_E2_repeats"]), f3(rp["mean_sd_event_E2_seeds"])
        kv["x8-rep-evsdF"], kv["x8-rep-evsdF-seeds"] = f3(rp["mean_sd_event_F_loeo_whole_repeats"]), f3(rp["mean_sd_event_F_loeo_whole_seeds"])
        kv["x8-rep-maxdiff"] = f3(rp["max_abs_event_F_diff_repeats"])
    return kv


def ls_concentration_keys(summary):
    """Share of the all-event HumAID label-set component carried by the two events with the largest mean LS over
    both encoders (seed means)."""
    kv, per = {}, {}
    for r in summary:
        if r["corpus"] == "humaid19":
            per[MODEL[r["model"]]] = {e["event"]: e["c_P"] - e["c_D"] for e in r["per_event_seed_mean"]}
    if len(per) < 2:
        return kv
    top = sorted(per["D"], key=lambda e: -(per["D"][e] + per["R"][e]))[:2]
    kv["H-LS-top2-events"] = " and ".join(ev(e) for e in top)
    for m, d in per.items():
        kv[f"H-{m}-LS-top2share"] = f"{100 * sum(d[e] for e in top) / sum(d.values()):.0f}"
    return kv


def table_arms(x8):
    """X8: whole-event LOEO aggregation and the matched contrast in every configuration."""
    iv = lambda lo, hi: f"[{f3z(lo)}, {f3z(hi)}]"
    rows = []
    for corpus in ("crisislext26", "humaid19"):
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]} ({'26' if corpus == 'crisislext26' else '19'} events)}}}} \\\\")
        for a in x8_arms(x8, corpus):
            s = a["subsets"]["all"]
            ag, e2 = s["loeo_aggregation"], s["E2"]
            ep = {"2 epochs": "2", "4 epochs": "4", "uncapped": "2", "TF-IDF": "--"}[a["arm"]]
            n = "all" if a["arm"] == "uncapped" else "6{,}000"
            cells = [f"\\quad {ARM_LABEL[a['model']]}", ep, n, str(len(a["seeds"])),
                     f"\\stk{{{f3(ag['mean'])}}}{{{iv(*ag['ci95'])}}}", f"\\stk{{{f3(e2['mean'])}}}{{{iv(*e2['t95'])}}}"]
            p = a["subsets"].get("protected", {}).get("E2")
            if p:
                cu = p["crossed95"]["unbiased"] if p["crossed95"] else None
                cells += [f"\\stk{{{f3(p['mean'])}}}{{{iv(*p['t95'])}}}", iv(cu["lo"], cu["hi"]) if cu else "--"]
            else:
                cells += ["--", "--"]
            rows.append(" & ".join(cells) + " \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Pooling held-out events against the matched contrast, in every configuration (present-class "
        "macro-F1, means over events of seed means). $A$: pooled minus per-event macro-F1 of the same whole-event LOEO "
        "predictions, with a 95\\% bootstrap interval over events and seeds (over events for one seed). $E_2$: matched "
        "contrast, with the 95\\% $t$ interval over events; on the six protected HumAID events also the 95\\% interval "
        "that treats events and seeds as crossed random effects. The two-epoch rows are the pre-specified recipe; the "
        "others are exploratory. Uncapped: the whole training pool (seed 42).}",
        "\\label{tab:arms}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
        " & & Training & & \\multicolumn{2}{c}{All events} & \\multicolumn{2}{c}{Protected HumAID: $E_2$} \\\\",
        "\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
        "Classifier & Epochs & tweets & Seeds & $A$ [95\\%] & $E_2$ [95\\%] & mean [95\\% $t$] & 95\\% crossed \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def fig_instability(x8, path):
    """X8: HumAID composite gap and matched contrast per run against the pooled F1 of the rarest class."""
    runs = x8_runs(x8)
    sty = {("distilbert-base-uncased", "ep2"): ("o", "white", "DistilBERT, 2 epochs"),
           ("distilbert-base-uncased", "ep4"): ("o", "#1b6ca8", "DistilBERT, 4 epochs"),
           ("roberta-base", "ep2"): ("^", "white", "RoBERTa, 2 epochs"),
           ("roberta-base", "ep4"): ("^", "#d95f02", "RoBERTa, 4 epochs"),
           ("tfidf-lr", "tfidf"): ("s", "#bdbdbd", "TF-IDF + LR")}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharex=True)
    for ax, k, lab in zip(axes, ("E0D", "E2"), ("composite gap $E_0^D$", "matched contrast $E_2$")):
        for (model, arm), (mk, fc, name) in sty.items():
            pts = [r for r in runs if r["model"] == model and r["arm"] == arm]
            ec = {"distilbert-base-uncased": "#1b6ca8", "roberta-base": "#d95f02"}.get(model, "#636363")
            ax.scatter([r["rare_pooled_id_f1"] for r in pts], [r[k] for r in pts], marker=mk, s=22, facecolor=fc,
                       edgecolor=ec, linewidth=0.9, label=name, zorder=3)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_xlabel("pooled in-distribution F1 of rare class", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


NOTE_ORDER = [(c, m) for c in ("crisislext26", "humaid19") for m in ("distilbert-base-uncased", "roberta-base")]


NOTE_CNAME = {"H": "HumAID", "C": "CrisisLexT26"}
NOTE_FIXTURE = {("2 epochs", "distilbert-base-uncased"): "DistilBERT, 2 ep.",
                ("2 epochs", "roberta-base"): "RoBERTa-base, 2 ep.",
                ("4 epochs", "distilbert-base-uncased"): "DistilBERT, 4 ep.",
                ("4 epochs", "roberta-base"): "RoBERTa-base, 4 ep.",
                ("TF-IDF", "tfidf-lr"): "TF-IDF + LR",
                ("uncapped", "distilbert-base-uncased"): "DistilBERT, uncapped"}


def note_arm(x9, corpus, model, arm="2 epochs"):
    return next(a for a in x9["arms"] if a["corpus"] == corpus and a["model"] == model and a["arm"] == arm)


def note_scores(x9):
    """Project Note: per-event mean, pooled score and A of the same two-epoch whole-event LOEO predictions."""
    out = []
    for corpus, model in NOTE_ORDER:
        a = note_arm(x9, corpus, model)
        per = float(np.mean([v[model] for v in x9["per_event"][corpus].values()]))
        out.append({"corpus": corpus, "model": model, "per": per, "pool": per + a["A"]["P"]["mean"], "arm": a})
    return out


def ci_keys(kv, b, blk):
    kv[b], kv[b + "lo"], kv[b + "hi"] = f3(blk["mean"]), f3(blk["ci95"][0]), f3(blk["ci95"][1])


def note_keys(x9, x10, ids_check=None):
    kv = {}
    for s in note_scores(x9):
        b = f"note-{CORP[s['corpus']]}-{MODEL[s['model']]}"
        kv[b + "-per"], kv[b + "-pool"] = f3(s["per"]), f3(s["pool"])
        for conv in ("P", "D", "O"):
            ci_keys(kv, f"{b}-A{conv}", s["arm"]["A"][conv])
        ci_keys(kv, f"{b}-W", s["arm"]["W"])
        ci_keys(kv, f"{b}-N", s["arm"]["N"])
        ci_keys(kv, f"{b}-A15", s["arm"]["A_min_support"])
        ps = list(s["arm"]["per_seed_A"].values())
        kv[b + "-seedmin"], kv[b + "-seedmax"] = f3(min(ps)), f3(max(ps))
    for corpus, c in CORP.items():
        arms = [a for a in x9["arms"] if a["corpus"] == corpus]
        for key, get in (("AP", lambda a: a["A"]["P"]["mean"]), ("A15", lambda a: a["A_min_support"]["mean"]),
                         ("AD", lambda a: a["A"]["D"]["mean"]), ("AO", lambda a: a["A"]["O"]["mean"])):
            v = [get(a) for a in arms]
            kv[f"note-{c}-{key}-min"], kv[f"note-{c}-{key}-max"] = f3(min(v)), f3(max(v))
        kv[f"note-{c}-narms"] = str(len(arms))
        kv[f"note-{c}-Wabsmax"] = f3(max(abs(a["W"]["mean"]) for a in arms))
        cb = x9["cells_below_min_support"][corpus]
        kv[f"note-{c}-cells15"], kv[f"note-{c}-msgs15"] = str(cb["cells"]), str(cb["messages"])
        kv[f"note-{c}-events15"], kv[f"note-{c}-cellstotal"] = str(cb["events"]), str(cb["cells_total"])
        kv[f"note-{c}-msgs15-pct"] = f"{100 * cb['messages'] / cb['messages_total']:.1f}"
        pe = x9["per_event"][corpus].values()
        n = [v["n"] for v in pe]
        kv[f"note-{c}-nmin"], kv[f"note-{c}-nmax"] = f"{min(n):,}".replace(",", "{,}"), f"{max(n):,}".replace(",", "{,}")
        kv[f"note-{c}-presmin"] = str(min(v["n_present"] for v in pe))
        kv[f"note-{c}-presmax"] = str(max(v["n_present"] for v in pe))
    for corpus, by_model in x9["cells"].items():
        for model, cells in by_model.items():
            b = f"note-{CORP[corpus]}-{MODEL[model]}"
            small = [c["f1"] for c in cells if c["support"] < MIN_CELL]
            if small:
                kv[b + "-smallf1"] = f2(np.mean(small))
                kv[b + "-otherf1"] = f2(np.mean([c["f1"] for c in cells if c["support"] >= MIN_CELL]))
        tot = {}
        for c in next(iter(by_model.values())):
            tot[c["class"]] = tot.get(c["class"], 0) + c["support"]
        kv[f"note-{CORP[corpus]}-minclass"] = f"{min(tot.values()):,}".replace(",", "{,}")
    for s in x9["spearman"]:
        b = f"note-{CORP[s['corpus']]}-{MODEL[s['model']]}"
        kv[b + "-rho"], kv[f"note-{CORP[s['corpus']]}-ncells"] = f2(s["rho_mean"]), str(s["n_cells"])
    chk = x9["implementation_check"]
    kv["note-check-n"] = f"{chk['n_scores']:,}".replace(",", "{,}")
    kv["note-check-exp"] = str(int(np.ceil(np.log10(max(chk["max_abs_diff_sklearn"], 1e-300)))))
    kv["note-check-companion"] = "0" if chk["max_abs_diff_companion"] == 0 else f"{chk['max_abs_diff_companion']:.1e}"
    if ids_check:
        kv["note-ids-runs"] = f"{ids_check['runs_checked']:,}".replace(",", "{,}")
        kv["note-ids-mismatches"] = str(ids_check["mismatches"])
        for corpus, n in ids_check["n_full"].items():
            kv[f"note-{CORP[corpus]}-nfull"] = f"{n:,}".replace(",", "{,}")
    if x10:
        for corpus, rec in x10["corpora"].items():
            b = f"note-Z-{CORP[corpus]}"
            kv[b + "-per"], kv[b + "-pool"] = f3(rec["P"]["per_event_mean"]), f3(rec["P"]["pooled"])
            for conv in ("P", "D", "O"):
                ci_keys(kv, f"{b}-A{conv}", {"mean": rec[conv]["A"], "ci95": rec[conv]["A_ci95"]})
    return kv


def civ(blk):
    return f"\\stk{{{f3(blk['mean'])}}}{{[{f3(blk['ci95'][0])}, {f3(blk['ci95'][1])}]}}"


def ci1(blk):
    return f"{f3(blk['mean'])} [{f3z(blk['ci95'][0])}, {f3z(blk['ci95'][1])}]"


def table_note_results(x9, x10):
    rows = []
    for corpus in ("crisislext26", "humaid19"):
        c = CORP[corpus]
        for s in [s for s in note_scores(x9) if s["corpus"] == corpus]:
            a = s["arm"]["A"]
            lab = NOTE_CNAME[c] if MODEL[s["model"]] == "D" else ""
            rows.append(f"{lab} & {MNAME[MODEL[s['model']]]} & {f3(s['per'])} & {f3(s['pool'])} & "
                        f"{civ(a['P'])} & {civ(a['D'])} & {civ(a['O'])} \\\\")
        if x10:
            z = x10["corpora"][corpus]
            rows.append(f" & BART-MNLI zero-shot & {f3(z['P']['per_event_mean'])} & {f3(z['P']['pooled'])} & " +
                        " & ".join(civ({"mean": z[cv]["A"], "ci95": z[cv]["A_ci95"]}) for cv in ("P", "D", "O")) +
                        " \\\\")
        if corpus == "crisislext26":
            rows.append("\\midrule")
    return "\n".join([
        "\\begin{table}[tbp]", "\\centering",
        "\\caption{Whole-event LOEO scores of the reference systems (seed means). Per event and pooled are "
        "present-class (P) macro-F1. $A^P$, $A^D$ and $A^O$ are pooled minus per-event macro-F1 of the same "
        "predictions when both scores use present classes, the scikit-learn default list or the fixed ontology; "
        "95\\% bootstrap intervals over events and seeds (events only for the zero-shot system) are shown below "
        "each value.}",
        "\\label{tab:note-results}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llrrccc@{}}", "\\toprule",
        "Resource & System & Per event & Pooled & $A^P$ & $A^D$ & $A^O$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_note_fixtures(x9):
    rows = []
    for corpus in ("crisislext26", "humaid19"):
        arms = [a for a in x9["arms"] if a["corpus"] == corpus]
        for i, a in enumerate(arms):
            lab = NOTE_CNAME[CORP[corpus]] if i == 0 else ""
            rows.append(f"{lab} & {NOTE_FIXTURE[a['arm'], a['model']]} & {len(a['seeds'])} & {ci1(a['A']['P'])} & "
                        f"{f3(a['W']['mean'])} & {f3(a['N']['mean'])} & {ci1(a['A_min_support'])} \\\\")
        if corpus == "crisislext26":
            rows.append("\\midrule")
    return "\n".join([
        "\\begin{table}[tbp]", "\\centering",
        f"\\caption{{Present-class aggregation difference $A^P$ for every reference system, its split into "
        f"event-size weighting $W$ and non-additivity $N$ ($A^P = W + N$), and $A^P$ after removing messages in "
        f"event--class cells with fewer than {MIN_CELL} messages (``cells $\\geq${MIN_CELL}''). Seed means with "
        f"95\\% bootstrap intervals; ep.: training epochs; LR: logistic regression. The two-epoch systems are "
        f"those of the analysis plan; the others are exploratory.}}",
        "\\label{tab:note-fixtures}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcrrrr@{}}", "\\toprule",
        f"Resource & System & Seeds & $A^P$ & $W$ & $N$ & $A^P$, cells $\\geq${MIN_CELL} \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


MIN_CELL = 15
EVENT_NOUNS = {"earthquake", "earthquakes", "floods", "wildfires", "bushfire", "bombings", "fire", "nightclub", "train",
               "crash", "helicopter", "shootings", "airport", "refinery", "meteor", "haze", "explosion", "building",
               "collapse"}


def table_note_events(x9):
    def cells(corpus):
        out = []
        for e, v in x9["per_event"][corpus].items():
            words = [w for w in ev(e).split() if not w.isdigit()]
            name = " ".join([words[0]] + [w.lower() if w.lower() in EVENT_NOUNS else w for w in words[1:]])
            out.append((name, f"{name} & {v['n']:,} & {v['n_present']} & {f2(v['distilbert-base-uncased'])} & "
                              f"{f2(v['roberta-base'])}".replace(",", "{,}")))
        return [row for _, row in sorted(out)]
    h, c = cells("humaid19"), cells("crisislext26")
    h += [" & & & & "] * (len(c) - len(h))
    col = lambda w: f">{{\\raggedright\\arraybackslash}}p{{{w}}}rrrr"  # noqa: E731
    body = [f"{a} & {b} \\\\" for a, b in zip(h, c)]
    return "\n".join([
        "\\begin{table}[tbp]", "\\centering",
        "\\caption{Events of the two resources: messages $n_e$, classes present $|C_e|$, and seed-mean "
        "present-class macro-F1 $F_e$ of the two-epoch DistilBERT (D) and RoBERTa-base (R) systems. Event names "
        "are shortened forms of the provider identifiers, in alphabetical order.}",
        "\\label{tab:note-events}", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
        f"\\begin{{tabular}}{{@{{}}{col('3.9cm')}@{{\\hspace{{10pt}}}}{col('3.5cm')}@{{}}}}", "\\toprule",
        "\\multicolumn{5}{@{}l}{HumAID} & \\multicolumn{5}{l@{}}{CrisisLexT26} \\\\",
        "\\cmidrule(r{6pt}){1-5}\\cmidrule{6-10}",
        "Event & $n_e$ & $|C_e|$ & D & R & Event & $n_e$ & $|C_e|$ & D & R \\\\", "\\midrule",
        *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def usage_example():
    """Verbatim usage example: score the seed-42 TF-IDF predictions of CrisisLexT26 through the companion."""
    import subprocess
    import sys
    import tempfile
    meta_path = "experiments/meta/corpus_crisislext26_meta.tsv"
    run = "experiments/runs/x3_tfidf/crisislext26/tfidf-lr/seed42"
    with tempfile.TemporaryDirectory() as tmp:
        stdout = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "companion_eval.py"),
                                 "--convention", "P", "--out", os.path.join(tmp, "out"), "--meta", meta_path,
                                 "--preds", run + "/loeo__*/preds.tsv.gz"],
                                capture_output=True, text=True, check=True).stdout
    stdout = stdout.replace(os.path.join(tmp, "out"), "out")
    return "\n".join([
        "\\begin{verbatim}",
        f"$ RUN={run}",
        "$ python code/companion_eval.py --convention P --out out \\",
        f"    --meta {meta_path} \\",
        "    --preds \"$RUN/loeo__*/preds.tsv.gz\"",
        stdout.rstrip("\n"),
        "\\end{verbatim}"])


def fig_note_cells(x9, path):
    """Project Note: per-class F1 against event--class support in whole-event LOEO predictions (RoBERTa-base)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.9), sharey=True)
    for ax, corpus in zip(axes, ("crisislext26", "humaid19")):
        cells = x9["cells"][corpus]["roberta-base"]
        s = np.array([c["support"] for c in cells])
        f = np.array([c["f1"] for c in cells])
        small = s < MIN_CELL
        ax.scatter(s[~small], f[~small], s=9, color="#1b6ca8", alpha=0.7, linewidths=0)
        ax.scatter(s[small], f[small], s=14, color="#c0392b", marker="^", linewidths=0)
        ax.axvline(MIN_CELL, color="0.4", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"{NOTE_CNAME[CORP[corpus]]}: {int(small.sum())} of {len(s)} cells below {MIN_CELL}", fontsize=8)
        ax.set_xlabel("messages of the class in the event", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("per-class F1 in the event", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table_evalterms(x7):
    """X7b: crossed events x seeds intervals for DUP, DUP_F and E3, and LOEO aggregation."""
    yn = lambda r, m: "yes" if r[f"tost_{m}"]["conservative"] else "no"
    iv = lambda r: f"[{f3z(r['lo'])}, {f3z(r['hi'])}]"
    rows = []
    for key in sorted(x7["evaluation_terms"], key=lambda k: (not k.startswith("humaid19"), k)):
        corpus, model = key.split("/")
        for sub in ("all", "protected"):
            b = x7["evaluation_terms"][key].get(sub)
            if b is None:
                continue
            d, e3 = b["DUP"], b["E3"]
            df_ = b.get("DUP_F")
            lab = "\\quad protected" if sub == "protected" else f"{CNAME[CORP[corpus]]}, {MODEL[model]}"
            rows.append(f"{lab} & {f3(d['t95']['mean'])} & {iv(d['t90'])} & {f3z(min(d['per_seed_mean']))} to "
                        f"{f3z(max(d['per_seed_mean']))} & {yn(d, '0.01')} / {yn(d, '0.02')} & "
                        + (f"{f3(df_['t95']['mean'])} & {iv(df_['t90'])} & {yn(df_, '0.01')} / {yn(df_, '0.02')} & "
                           if df_ else "-- & -- & -- & ")
                        + f"{f3(e3['t95']['mean'])} & {iv(e3['t95'])} \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Evaluation-time terms with seed-aware intervals (X7, exploratory; present-class macro-F1). DUP and "
        "DUP$_F$: in-distribution score on $R_e$ minus the score after removing exact (DUP) or near (DUP$_F$, seeds 42, 1, "
        "2) duplicates of test tweets from training; 90\\% crossed events$\\times$seeds interval (two-component "
        "variance, Table~\\ref{tab:seedaware}), the range of per-seed means, and whether the 90\\% interval lies within "
        "$\\pm0.01$ / $\\pm0.02$ (two one-sided tests). HumAID has two within-event duplicate rows, so its DUP measures "
        "retraining noise. $E_3^P$: the LOEO model on $R_e$ minus on the whole event, 95\\% crossed interval. LOEO "
        "aggregation with intervals is in Table~\\ref{tab:practices}.}",
        "\\label{tab:evalterms}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrcccrccrc@{}}", "\\toprule",
        " & \\multicolumn{4}{c}{DUP} & \\multicolumn{3}{c}{DUP$_F$} & \\multicolumn{2}{c}{$E_3^P$} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(l){9-10}",
        "Corpus, encoder & mean & 90\\% & per seed & $\\pm$.01/.02 & mean & 90\\% & $\\pm$.01/.02 & mean & 95\\% \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_ep4_seeds(x7):
    """X7: per-seed protected and all-event matched contrast of the four-epoch arms beside two epochs."""
    rows = []
    order = ["42", "1", "2", "3", "4"]
    for model, fe in sorted(x7["four_epoch"].items(), key=lambda x: MODEL[x[0]]):
        for sub in ("protected", "all"):
            b = fe["subsets"][sub]
            for ep, per, blk in (("4", b["per_seed_E2_ep4"], "ep4"), ("2", b["per_seed_E2_ep2"], "ep2")):
                lab = f"{MNAME[MODEL[model]].replace('-base', '')}, {'protected' if sub == 'protected' else 'all events'}" \
                    if ep == "4" else ""
                c = b["complete"][blk]
                rows.append(f"{lab} & {ep} & " + " & ".join(f3(per[s]) if s in per else "--" for s in order)
                            + f" & {f3(c['E2'])} [{f3(c['E2_t95'][0])}, {f3(c['E2_t95'][1])}] \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Matched contrast $E_2^P$ per seed at four and two epochs (four epochs exploratory except seed 42). "
        "Mean: over seeds, with the 95\\% $t$ interval over events of seed-averaged contributions.}",
        "\\label{tab:ep4seeds}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lcrrrrrc@{}}", "\\toprule",
        " & & \\multicolumn{5}{c}{Seed} & \\\\ \\cmidrule(lr){3-7}",
        "Encoder, events & Epochs & 42 & 1 & 2 & 3 & 4 & Mean [95\\%] \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_recall(x7):
    """X7: per-class recall and F1 of whole-event LOEO predictions for three HumAID classes."""
    rows = []
    for cls, tag in RECALL_TAG.items():
        for model in sorted(x7["class_recall"], key=lambda k: MODEL[k]):
            r = x7["class_recall"][model][cls]
            lo, idr = r["loeo_whole"], r["id_R"]
            lab = f"{cls.replace('_', ' ')} ({r['events_present']}; {r['support_whole']:,})".replace(",", "{,}") \
                if MODEL[model] == "D" else ""
            rows.append(f"{lab} & {MNAME[MODEL[model]].replace('-base', '')} & {f2(lo['recall_pooled'])} & "
                        f"{f2(lo['recall_event_mean'])} & {f2(lo['f1_pooled'])} & {f2(lo['f1_event_mean'])} & "
                        f"{f2(idr['recall_pooled'])} & {f2(idr['recall_event_mean'])} & "
                        f"{lo['false_alarms_absent_events']:.1f} ({lo['events_absent']}) \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Three actionable HumAID classes (X7, exploratory; two epochs, means over five seeds). In parentheses: "
        "events containing the class and its tweets. LOEO: whole-event predictions of the LOEO models; pooled over "
        "events, or the mean over the events that contain the class. ID: in-distribution predictions on $R_e$. Absent: "
        "LOEO predictions of the class in the events that lack it (number of such events in parentheses), which "
        "convention~D charges as a zero-F1 class and convention~P ignores.}",
        "\\label{tab:recall}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
        " & & \\multicolumn{2}{c}{LOEO recall} & \\multicolumn{2}{c}{LOEO F1} & \\multicolumn{2}{c}{ID recall} & Absent \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "Class (events; tweets) & Encoder & pooled & per event & pooled & per event & pooled & per event & predictions \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_merged(rev, summary):
    """HumAID rescored with Missing or found people merged into Injured or dead people (X5b)."""
    rows = []
    base = {MODEL[r["model"]]: r for r in summary if r["corpus"] == "humaid19"}
    for model, subs in sorted(rev["merged_humaid"].items(), key=lambda x: MODEL[x[0]]):
        m = MODEL[model]
        for sub in ("all", "protected"):
            for scheme, b in (("10", base[m]["subsets"][sub]), ("9", subs[sub])):
                P, D = b["P"], b["D"]
                rem = b["headline"]["remainder"]
                lab = (f"{MNAME[m].replace('-base', '')}, {'all events' if sub == 'all' else 'protected'}"
                       if scheme == "10" else "")
                rows.append(f"{lab} & {scheme} & {f3(D['E0']['mean'])} & {f3(P['E1']['mean'])} & "
                            f"{f3(P['E2']['mean'])} & {f3(P['E3']['mean'])} & {f3(P['MC']['mean'])} & "
                            f"{100 * P['E2']['mean'] / D['E0']['mean']:.0f} & "
                            f"[{f3(rem['ci95'][0])}, {f3(rem['ci95'][1])}] \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{HumAID under its distributed ten-class scheme and with \\emph{Missing or found people} merged into "
        "\\emph{Injured or dead people} (nine classes), rescoring the same predictions (seed means; last column: 95\\% "
        "crossed-bootstrap interval of $E_0^D-E_2^P$).}",
        "\\label{tab:merged}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lcrrrrrrc@{}}", "\\toprule",
        "Encoder, events & Classes & $E_0^D$ & $E_1^P$ & $E_2^P$ & $E_3^P$ & LS & $E_2^P/E_0^D$ (\\%) & $E_0^D-E_2^P$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_seedaware(rev, summary):
    """Registered event-level t intervals against crossed (events x seeds) intervals (X5c)."""
    reg = {(r["corpus"], r["model"]): r for r in summary}
    rows = []
    for est in ("E2", "MC"):
        for corpus, sub in (("humaid19", "protected"), ("humaid19", "all"), ("crisislext26", "all")):
            for model in MODEL:
                r = rev["random_effects"].get(f"{corpus}/{model}", {}).get(sub)
                if not r:
                    continue
                t = reg[(corpus, model)]["subsets"][sub]["P"]["t_rules"][est]
                c, u = r[est]["t95"], r[est]["t95"]["unbiased"]
                by_seed = dict(zip(r[est]["seeds"], r[est]["per_seed_mean"]))
                order = ["42"] + sorted((s for s in by_seed if s != "42"), key=int)
                lab = f"{CNAME[CORP[corpus]]}{' protected' if sub == 'protected' else ''}, {MODEL[model]}"
                rows.append(f"{'$E_2^P$' if est == 'E2' else 'LS'} & {lab} & {f3(t['mean'])} & "
                            f"[{f3(t['t95'][0])}, {f3(t['t95'][1])}] & [{f3(c['lo'])}, {f3(c['hi'])}] ({c['df']:.1f}) & "
                            f"[{f3(u['lo'])}, {f3(u['hi'])}] ({u['df']:.1f}) & "
                            + " & ".join(f3(by_seed[s]) for s in order) + " & " * (5 - len(order)) + " \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Seed-aware intervals (95\\%; X5, exploratory). Registered: $t$ interval over events of "
        "seed-averaged contributions. Crossed: two-way events $\\times$ seeds variance. The two-term version uses "
        "$(\\mathrm{MS}_e+\\mathrm{MS}_s)/(ES)$, whose expectation exceeds the variance of the mean by the residual "
        "component; the unbiased version subtracts $\\mathrm{MS}_r$. Satterthwaite degrees of freedom in parentheses; "
        "when they collapse, the unbiased interval is the wider one. Last five columns: mean over events for each seed "
        "(CrisisLexT26 has seeds 42, 1 and 2). D: DistilBERT; R: RoBERTa.}",
        "\\label{tab:seedaware}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{@{}llrcccrrrrr@{}}", "\\toprule",
        " & & & & & & \\multicolumn{5}{c}{Seed} \\\\ \\cmidrule(l){7-11}",
        " & Events, encoder & Mean & Registered & Crossed, two-term & Crossed, unbiased & 42 & 1 & 2 & 3 & 4 \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_decomposition(summary):
    rows = []
    for r in sorted(summary, key=lambda r: (r["corpus"] != "humaid19", r["model"])):
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        for sub in ("all", "protected"):
            if sub not in r["subsets"]:
                continue
            b = r["subsets"][sub]
            P, D = b["P"], b["D"]
            lab = f"{CNAME[c]}, {MNAME[m].replace('-base', '')}" + (" (protected)" if sub == "protected" else "")
            rows.append(f"{lab} & {f3(D['E0']['mean'])} & {f3(P['E1']['mean'])} & {f3(P['E2']['mean'])} & "
                        f"{f3(P['E3']['mean'])} & {f3(P['MC']['mean'])} & {f3(P['E0']['mean'])} & "
                        f"{f3(P['E2p']['mean'])} & {f3(O_E0(b))} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{What the composite gap contains (seed means over events). $E_0^D$, pooled random-split macro-F1 "
        "minus mean per-event LOEO macro-F1 under the scikit-learn default label set, equals "
        "$E_1^P+E_2^P+E_3^P+\\mathrm{LS}$ exactly before display rounding: aggregation, matched contrast on identical tweets and test-set term "
        "(present-class macro-F1), plus the label-set component. $E_0^P$: the gap under present-class macro-F1; "
        "$E_2^{\\mathrm{pool}}$: the matched contrast on pooled tweets (the other order); $E_0^O$: the gap with a fixed corpus-wide "
        "label list. 95\\% intervals: Table~\\ref{tab:intervals} (\\appref{app:seeds}).}",
        "\\label{tab:decomp}", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
        "Corpus, encoder & $E_0^D$ & $E_1^P$ & $E_2^P$ & $E_3^P$ & LS & $E_0^P$ & $E_2^{\\mathrm{pool}}$ & $E_0^O$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def O_E0(b):
    return b["O"]["E0"]["mean"] if "O" in b else None


def table_hypotheses(summary, kv):
    rows = []
    for m in ("D", "R"):
        rH = next((r for r in summary if r["corpus"] == "humaid19" and MODEL[r["model"]] == m), None)
        rC = next((r for r in summary if r["corpus"] == "crisislext26" and MODEL[r["model"]] == m), None)
        cells = [MNAME[m]]
        for r, sub in ((rH, "protected"), (rC, "all")):
            if r is None or sub not in r["subsets"]:
                cells += ["--", "--"]
                continue
            t = r["subsets"][sub]["P"]["t_rules"]["E2"]
            pos = r["subsets"][sub]["P"]["E2_events_positive"]
            cells.append(f"\\stk{{{f3(t['mean'])} ({pos}/{t['n_events']})}}{{[{f3(t['t95'][0])}, {f3(t['t95'][1])}]}}")
            mc = r["subsets"][sub]["P"]["t_rules"]["MC"]
            cells.append(f"\\stk{{{f3(mc['mean'])}}}{{[{f3(mc['t95'][0])}, {f3(mc['t95'][1])}]}}")
        cells.append(f"\\stk{{{kv[f'H3-{m}-mean']}}}{{[{kv[f'H3-{m}-tlo90']}, {kv[f'H3-{m}-thi90']}]}}")
        rows.append(" & ".join(cells) + " \\\\")
        if f"x7-holm-H1-{m}-padj" in kv:
            rows.append("\\quad Holm $p$ & " + " & ".join(kv[f"x7-holm-{d}-{m}-padj"] for d in ("H1", "H2p", "H1b", "H2c", "H3"))
                        + " \\\\[1pt]")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Registered hypotheses, as defined in Section~\\ref{sec:setup} (present-class macro-F1; mean over "
        "events of seed-averaged contributions with two-sided $t$ intervals over events). H1, H1b and H2: 95\\% "
        "intervals, positive events in parentheses. H3: 90\\% intervals on the "
        f"{kv['H3-D-n']} events per encoder whose chronological pool excludes at least 25\\% of the LOEO pool. "
        "Holm $p$: Holm-adjusted $p$-value over the ten decisions (X7; two-sided $t$ test of zero for H1, H1b and H2, "
        "two one-sided tests at $\\pm0.02$ for H3).}",
        "\\label{tab:hyp}", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        " & \\multicolumn{2}{c}{HumAID protected (6)} & \\multicolumn{2}{c}{CrisisLexT26 (26)} & "
        f"Both ({kv['H3-D-n']}) \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}",
        "Encoder & $E_2$ (H1) & LS (H2) & $E_2$ (H1b) & LS (H2) & $E_5$ (H3) \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def h3_values(summary, m):
    vals = []
    for r in summary:
        if MODEL[r["model"]] != m:
            continue
        sub = "protected" if r["corpus"] == "humaid19" else "all"
        if sub not in r["subsets"]:
            continue
        ec = r["subsets"][sub]["P"]["event_contrib"]
        vals += [v for e, v in ec["E5"].items() if ec["future_share"][e] >= 0.25]
    return vals


def table_margins(summary):
    margins = (0.01, 0.02, 0.03)

    def pos_cell(t, mg):
        if not t["t95"][0] > 0:
            return "--"
        return "M" if t["mean"] >= mg else "+"

    rows = []
    for m in ("D", "R"):
        rH = next(r for r in summary if r["corpus"] == "humaid19" and MODEL[r["model"]] == m)
        rC = next(r for r in summary if r["corpus"] == "crisislext26" and MODEL[r["model"]] == m)
        _, lo90, hi90 = tci(h3_values(summary, m), 0.90)
        tests = [("H1", rH["subsets"]["protected"]["P"]["t_rules"]["E2"]),
                 ("H1b", rC["subsets"]["all"]["P"]["t_rules"]["E2"]),
                 ("H2 prot.", rH["subsets"]["protected"]["P"]["t_rules"]["MC"]),
                 ("H2 CLX", rC["subsets"]["all"]["P"]["t_rules"]["MC"])]
        cells = [MNAME[m]]
        for _, t in tests:
            cells += [pos_cell(t, mg) for mg in margins]
        cells += ["E" if (lo90 > -mg and hi90 < mg) else "--" for mg in margins]
        rows.append(" & ".join(cells) + " \\\\")
    head = " & ".join(f"{mg:.2f}" for mg in margins)
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Registered decisions at margins of 0.01, 0.02 (registered) and 0.03 macro-F1, from the intervals of "
        "Table~\\ref{tab:hyp}. M: 95\\% $t$ interval above zero and mean at least the margin; +: interval above zero, "
        "mean below the margin; E: 90\\% interval within $\\pm$ the margin (equivalence); --: neither. CLX: "
        "CrisisLexT26.}",
        "\\label{tab:margins}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}l" + "ccc" * 5 + "@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{H1: $E_2$, protected} & \\multicolumn{3}{c}{H1b: $E_2$, CLX} & "
        "\\multicolumn{3}{c}{H2: LS, protected} & \\multicolumn{3}{c}{H2: LS, CLX} & \\multicolumn{3}{c}{H3: $E_5$} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\\cmidrule(l){14-16}",
        "Encoder & " + " & ".join([head] * 5) + " \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_consequence(summary, kv):
    rows = []
    for r in sorted(summary, key=lambda r: (r["corpus"] != "humaid19", r["model"])):
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        rows.append(f"\\stk{{{CNAME[c]}}}{{{MNAME[m].replace('-base', '')}}} & {kv[f'{c}-{m}-all-tau']} & {kv[f'{c}-{m}-all-moved3']} & "
                    f"{kv[f'{c}-{m}-all-hardestD']} / {kv[f'{c}-{m}-all-hardestP']} & "
                    f"{kv[f'{c}-{m}-all-hazD']} / {kv[f'{c}-{m}-all-hazP']} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Event-difficulty conclusions under the default (D) and present-class (P) macro-F1 label sets "
        "(whole-event LOEO scores; seed means). $\\tau$: Kendall correlation of event rankings. Moved: events whose "
        "rank changes by at least three. Hazard scores average their events; several hazards have only one or two.}",
        "\\label{tab:consequence}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}p{2.7cm}rrp{4.1cm}p{2.1cm}@{}}", "\\toprule",
        "Corpus, encoder & $\\tau$ & Moved & Hardest event (D / P) & Hardest hazard (D / P) \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_secondary(summary, kv):
    rows = []
    for r in sorted(summary, key=lambda r: (r["corpus"] != "humaid19", r["model"])):
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        P, D = r["subsets"]["all"]["P"], r["subsets"]["all"]["D"]
        g = lambda k, B=P: f3(B[k]["mean"]) if k in B else "--"
        rows.append(f"{CNAME[c]}, {MNAME[m].replace('-base', '')} & {g('E4')} & {g('DUP')} & {g('DUP_F')} & {g('E5_div')} & "
                    f"{g('E5_time')} & {g('mask_dE0', D)} & {g('mask_dE2')} & {g('mask_LOEO_benefit_whole')} & "
                    f"{kv.get(f'{c}-{m}-pm-share', '--')} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Secondary estimands (present-class macro-F1, seed means over all events). $E_4$: matched contrast "
        "when only earlier tweets of the target are eligible. DUP and DUP$_F$: effect of removing exact and near "
        "duplicates of test tweets from the in-distribution training sample (paired samples; HumAID DUP is a "
        "run-to-run placebo). $E_5$ split into a diversity part (all versus $k$ random source events) and a time part "
        "($k$ random versus $k$ earlier events). Masking: change in the composite gap $E_0^D$ and in the matched "
        "contrast $E_2^P$ when event tokens are masked, and the change in mean present-class LOEO macro-F1 under "
        "masking (means over seeds 42 and 1; per seed in Table~\\ref{tab:masking}). Prior: share of $E_2$ removed by "
        "oracle label-prior matching (\\%).}",
        "\\label{tab:secondary}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
        "Corpus, encoder & $E_4$ & DUP & DUP$_F$ & $E_5^{div}$ & $E_5^{time}$ & $\\Delta E_0^D$ & $\\Delta E_2^P$ & "
        "$\\Delta F^P_{\\textsc{loeo}}$ & Prior \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_masking(summary):
    """Event-token masking per seed (two seeds, so no seed interval)."""
    rows = []
    for r in sorted(summary, key=lambda r: (r["corpus"] != "humaid19", r["model"])):
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        P, D = r["subsets"]["all"]["P"], r["subsets"]["all"]["D"]
        sp = {s["seed"]: s for s in P.get("mask_seed_values", [])}
        sd = {s["seed"]: s for s in D.get("mask_seed_values", [])}
        for seed in sorted(sp, key=lambda s: (s != "42", int(s))):
            rows.append(f"{CNAME[c]}, {MNAME[m].replace('-base', '')} & {seed} & {f3(sd[seed]['dE0'])} & "
                        f"{f3(sp[seed]['dE2'])} & {f3(sp[seed]['LOEO_benefit_whole'])} & "
                        f"{f3(sd[seed]['LOEO_benefit_whole'])} \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Event-token masking per seed (all events): change in the composite gap $E_0^D$, in the matched "
        "contrast $E_2^P$, and in mean LOEO macro-F1 under conventions P and D.}",
        "\\label{tab:masking}", "\\footnotesize",
        "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
        "Corpus, encoder & Seed & $\\Delta E_0^D$ & $\\Delta E_2^P$ & $\\Delta F^P_{\\textsc{loeo}}$ & "
        "$\\Delta F^D_{\\textsc{loeo}}$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def grouped(summary, ncols, row_fn, subsets=("all", "protected")):
    """Rows grouped under one header line per corpus and encoder, so the first column stays narrow."""
    rows = []
    for r in sorted(summary, key=lambda r: (r["corpus"] != "humaid19", r["model"])):
        c, m = CORP[r["corpus"]], MODEL[r["model"]]
        body = [row_fn(r, sub) for sub in subsets if sub in r["subsets"]]
        body = [x for b in body for x in (b if isinstance(b, list) else [b])]
        if rows:
            rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{{CNAME[c]}, {MNAME[m].replace('-base', '')}}}}} \\\\")
        rows += body
    return rows


def table_seeds(summary):
    def row_fn(r, sub):
        D, P = r["subsets"][sub]["D"]["seed_values"], r["subsets"][sub]["P"]["seed_values"]
        return [f"\\quad seed {d['seed']} & {f3(d['E0'])} & {f3(p['E1'])} & {f3(p['E2'])} & {f3(p['E3'])} & "
                f"{f3(p['MC'])} & {f3(p['E0'])} & {f3(p.get('E5'))} & {f3(p['DUP'])} \\\\" for d, p in zip(D, P)]
    rows = grouped(summary, 9, row_fn, subsets=("all",))
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Per-seed estimands, all events. $E_0^D=E_1^P+E_2^P+E_3^P+\\mathrm{LS}$ within each row.}",
        "\\label{tab:seeds}", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
        " & $E_0^D$ & $E_1^P$ & $E_2^P$ & $E_3^P$ & LS & $E_0^P$ & $E_5^P$ & DUP$^P$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_intervals(summary):
    iv = lambda v: f"[{f3(v[0])}, {f3(v[1])}]"

    def row_fn(r, sub):
        b = r["subsets"][sub]
        P, D, t, hl = b["P"], b["D"], b["P"]["t_rules"], b["headline"]
        return (f"\\quad {'protected' if sub == 'protected' else 'all events'} & {iv(D['E0']['ci95'])} & "
                f"{iv(P['E1']['ci95'])} & {iv(t['E2']['t95'])} & {iv(t['MC']['t95'])} & {iv(P['E2p']['ci95'])} & "
                f"{iv(hl['remainder']['ci95'])} \\\\")
    rows = grouped(summary, 7, row_fn)
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{95\\% intervals for Table~\\ref{tab:decomp}. Pooled quantities ($E_0^D$, $E_1^P$, $E_2^{\\mathrm{pool}}$) and the "
        "remainder $E_0^D-E_2^P$ use the crossed bootstrap over events and seeds (5{,}000 draws, the same draws for both "
        "conventions); the per-event estimands $E_2^P$ and LS use $t$ intervals over events of seed-averaged "
        "contributions, as in the main text.}",
        "\\label{tab:intervals}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        " & $E_0^D$ & $E_1^P$ & $E_2^P$ & LS & $E_2^{\\mathrm{pool}}$ & $E_0^D-E_2^P$ \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def rare_f1_range(seed_recs):
    f = [s["rarest_class"]["pooled_id_f1"] for s in seed_recs]
    if not f:
        return "--"
    return f2(min(f)) if len(f) == 1 or f2(min(f)) == f2(max(f)) else f"{f2(min(f))}--{f2(max(f))}"


def table_aggregation(agg):
    order = [("confirm", "humaid19"), ("confirm_ep4", "humaid19"), ("x3_tfidf", "humaid19"),
             ("confirm", "crisislext26"), ("confirm_ep4", "crisislext26"), ("x3_tfidf", "crisislext26")]
    recipe = {"confirm": "2 epochs", "confirm_ep4": "4 epochs", "x3_tfidf": "TF-IDF + LR"}
    rows = []
    for plan, corpus in order:
        recs = sorted((x for x in agg or [] if x["plan"] == plan and x["corpus"] == corpus),
                      key=lambda x: MODEL.get(x["model"], "T"))
        for x in recs:
            sm, m = x["summary"], MODEL.get(x["model"])
            enc = MNAME[m].replace("-base", "") if m else "--"
            n = len(x["seeds"])
            idr = sm.get("id_spearman_prev_f1_R")
            rows.append(f"{CNAME[CORP[corpus]]} & {enc} & {recipe[plan]} ({n}) & {f3(sm['E1']['mean'])} & "
                        f"{f3(sm['scoring']['mean'])} & {f3(sm['reweighting']['mean'])} & "
                        f"{f3(sm['reweighting']['min'])} to {f3(sm['reweighting']['max'])} & "
                        f"{f3(sm['loeo_aggregation_R']['mean'])} & {f3(sm['loeo_aggregation_whole']['mean'])} & "
                        f"{f2(idr['within_class_mean']) if idr else '--'} & "
                        f"{f2(sm['spearman_prev_f1']['mean'])} & {rare_f1_range(x['seeds'])} \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Where aggregation comes from (present-class macro-F1, means over seeds; seeds in parentheses). "
        "$E_1^P$ of the in-distribution predictions equals a within-event scoring term, "
        "$\\frac{1}{K}\\sum_k(F_k^{pool}-\\bar F_k)$, plus a class-reweighting term, $\\sum_k(\\frac{1}{K}-w_k)\\bar F_k$, "
        "where $w_k$ is class $k$'s weight in the per-event mean and $\\bar F_k$ its $w$-weighted per-event F1 "
        "(Appendix~\\ref{app:aggregation}). LOEO aggregation: pooled minus per-event macro-F1 of the same LOEO "
        "predictions, on the matched fifth of each event ($R_e$) and on whole events. $\\rho$: Spearman correlation "
        "across event--class cells between per-cell F1 and the class's share of the event, for the in-distribution "
        "predictions on $R_e$ after centring F1 on each class's mean (ID, within class) and for the LOEO "
        "predictions on whole events (LOEO, uncentred). Rare-class "
        "F1: range over seeds of the pooled in-distribution F1 of the class present in the fewest events.}",
        "\\label{tab:aggregation}", "\\scriptsize", "\\setlength{\\tabcolsep}{1.9pt}",
        "\\begin{tabular}{@{}lllrrrcrrrrc@{}}", "\\toprule",
        " & & & & \\multicolumn{3}{c}{Split of $E_1^P$} & \\multicolumn{2}{c}{LOEO aggregation} & "
        "\\multicolumn{2}{c}{$\\rho$} & Rare-class \\\\",
        "\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
        "Corpus & Encoder & Recipe & $E_1^P$ & scoring & reweight. & reweight. range & on $R_e$ & whole & ID & LOEO & F1 \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def per_class_means(agg, plan, corpus, model):
    """Mean over seeds of each class's per-class record; None if the arm is absent."""
    g = next((x for x in agg or [] if (x["plan"], x["corpus"], x["model"]) == (plan, corpus, model)), None)
    if g is None:
        return None
    classes = g["seeds"][0]["per_class"].keys()
    out = {}
    for k in classes:
        pc = [s["per_class"][k] for s in g["seeds"]]
        out[k] = {f: float(np.mean([p[f] for p in pc])) for f in pc[0]}
        out[k]["n_seeds_no_pred"] = sum(p["id_n_pred"] == 0 for p in pc)
        out[k]["n_seeds"] = len(pc)
    return out


def per_class_keys(agg):
    kv = {}
    for plan, tag in (("confirm", "b"), ("confirm_ep4", "ep4")):
        for corpus in ("humaid19", "crisislext26"):
            for model, m in MODEL.items():
                pcm = per_class_means(agg, plan, corpus, model)
                for k, r in (pcm or {}).items():
                    base = f"pc-{CORP[corpus]}-{m}-{tag}-{k.replace('_', '-')}"
                    kv[f"{base}-idf1"], kv[f"{base}-loeof1"] = f3(r["id_f1"]), f3(r["loeo_f1_R"])
                    kv[f"{base}-npred"] = f"{r['id_n_pred']:.0f}"
                    kv[f"{base}-nopred"] = str(r["n_seeds_no_pred"])
                    kv[f"{base}-ntrue"] = f"{r['n_true']:.0f}"
    return kv


def table_perclass(agg):
    rows = []
    for corpus in ("humaid19", "crisislext26"):
        b = {m: per_class_means(agg, "confirm", corpus, model) for model, m in MODEL.items()}
        e4 = {m: per_class_means(agg, "confirm_ep4", corpus, model) for model, m in MODEL.items()}
        if not all(b.values()):
            continue
        rows.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\textit{{{CNAME[CORP[corpus]]}}}}} \\\\")
        for k, r in b["D"].items():
            cell = lambda d, f: f3(d[k][f]) if d else "--"
            rows.append(f"\\quad {k.replace('_', ' ')} & {r['n_events']:.0f} & {r['n_true']:.0f} & "
                        f"{b['D'][k]['id_n_pred']:.0f} & {b['R'][k]['id_n_pred']:.0f} & "
                        f"{cell(b['D'], 'id_f1')} & {cell(b['R'], 'id_f1')} & "
                        f"{cell(b['D'], 'loeo_f1_R')} & {cell(b['R'], 'loeo_f1_R')} & "
                        f"{cell(e4['D'], 'id_f1')} & {cell(e4['R'], 'id_f1')} \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{Per-class scores on the matched test tweets $R$ (means over seeds; two epochs unless stated). "
        "Ev.: events containing the class; True: its tweets in $R$; Predicted: mean number of tweets in $R$ the "
        "in-distribution model assigns to it; ID F1 and LOEO F1: pooled F1 of the class for the in-distribution "
        "predictions and for the LOEO predictions on the same tweets; 4 ep. ID: in-distribution F1 at four epochs "
        "(HumAID: the four-epoch seeds of Table~\\ref{tab:exploratory}; CrisisLexT26: DistilBERT seed 42 only, "
        "no RoBERTa arm, shown as --). D: DistilBERT; R: RoBERTa.}",
        "\\label{tab:perclass}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}", "\\toprule",
        " & & & \\multicolumn{2}{c}{Predicted} & \\multicolumn{2}{c}{ID F1} & \\multicolumn{2}{c}{LOEO F1} & "
        "\\multicolumn{2}{c}{4 ep. ID} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
        "Class & Ev. & True & D & R & D & R & D & R & D & R \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_exploratory(summary, ep4, x3, agg):
    """Two- versus four-epoch composition on the same seeds, and the TF-IDF arm (all exploratory except seed 42)."""
    seed_recs = {(x["plan"], x["corpus"], x["model"]): x["seeds"] for x in agg or []}
    rare = lambda plan, corpus, model, seeds=None: rare_f1_range(
        [s for s in seed_recs.get((plan, corpus, model), []) if seeds is None or str(s["seed"]) in seeds])
    rows = []
    base = {(r["corpus"], r["model"]): r for r in summary}
    for r in sorted(ep4 or [], key=lambda r: (r["corpus"] != "humaid19", r["model"])):
        seeds = r["seeds"]
        b2 = base[(r["corpus"], r["model"])]["subsets"]["all"]
        lab = f"{CNAME[CORP[r['corpus']]]}, {MNAME[MODEL[r['model']]].replace('-base', '')}"
        sd = ", ".join(sorted(seeds, key=lambda s: (s != "42", int(s))))
        same = lambda conv: [x for x in b2[conv]["seed_values"] if x["seed"] in seeds]
        mean = lambda conv, k: float(np.mean([x[k] for x in same(conv)]))
        l2 = rare("confirm", r["corpus"], r["model"], seeds)
        l4 = rare("confirm_ep4", r["corpus"], r["model"], seeds)
        rows.append(f"{lab} & {sd} & 2 & {f3(mean('D', 'E0'))} & {f3(mean('P', 'E1'))} & {f3(mean('P', 'E2'))} & "
                    f"{f3(mean('P', 'E3'))} & {f3(mean('P', 'MC'))} & {f3(mean('D', 'mean_c'))} & {l2} \\\\")
        P, D = r["subsets"]["all"]["P"], r["subsets"]["all"]["D"]
        rows.append(f" & & 4 & {f3(D['E0']['mean'])} & {f3(P['E1']['mean'])} & {f3(P['E2']['mean'])} & "
                    f"{f3(P['E3']['mean'])} & {f3(P['MC']['mean'])} & {f3(D['mean_c']['mean'])} & {l4} \\\\")
    if x3:
        rows.append("\\addlinespace")
    for r in sorted(x3 or [], key=lambda r: r["corpus"] != "humaid19"):
        P, D = r["subsets"]["all"]["P"], r["subsets"]["all"]["D"]
        nl = rare("x3_tfidf", r["corpus"], r["model"])
        rows.append(f"{CNAME[CORP[r['corpus']]]}, TF-IDF + LR & {', '.join(sorted(r['seeds'], key=lambda s: (s != '42', int(s))))} "
                    f"& -- & {f3(D['E0']['mean'])} & {f3(P['E1']['mean'])} & {f3(P['E2']['mean'])} & {f3(P['E3']['mean'])} & "
                    f"{f3(P['MC']['mean'])} & {f3(D['mean_c']['mean'])} & {nl} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Composition under other recipes and models (all events, seed means). Each four-epoch arm is shown "
        "next to the registered two-epoch recipe on the same seeds; the TF-IDF + logistic-regression classifier uses "
        "the transformer runs' training samples. Four epochs were registered for HumAID seed 42 of each encoder; the "
        "other four-epoch seeds, the CrisisLexT26 four-epoch arm and the TF-IDF classifier are exploratory "
        "(Section~\\ref{sec:setup}). LOEO$^D$: mean whole-event LOEO macro-F1 under the default convention. "
        "Rare F1: range over seeds of the pooled in-distribution F1 of the class present in the fewest events.}",
        "\\label{tab:exploratory}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcrrrrrrc@{}}", "\\toprule",
        "Corpus, model & Seeds & Epochs & $E_0^D$ & $E_1^P$ & $E_2^P$ & $E_3^P$ & LS & LOEO$^D$ & Rare F1 \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_registered(kv):
    rows = []
    for c, sub, lab in (("H", "prot", "HumAID protected (6)"), ("H", "all", "HumAID all (19)"), ("C", "all", "CrisisLexT26 (26)")):
        for m in ("D", "R"):
            b = f"reg-{c}-{m}-{sub}"
            if f"{b}-E2" not in kv:
                continue
            rows.append(f"{lab} & {MNAME[m].replace('-base', '')} & {kv[b + '-E2']} [{kv[b + '-E2-lo']}, {kv[b + '-E2-hi']}] & "
                        f"{kv[b + '-MC']} [{kv[b + '-MC-lo']}, {kv[b + '-MC-hi']}] & "
                        f"{kv[b + '-E5']} [{kv[b + '-E5-lo90']}, {kv[b + '-E5-hi90']}] ({kv[b + '-E5-n']}) \\\\")
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        "\\caption{As-registered analysis (hierarchical percentile bootstrap over events and seeds within events, "
        "5{,}000 draws; present-class macro-F1). Original rules: H1 supported if the 95\\% interval of $E_2$ excludes 0 "
        "on the protected events; H2 if the 95\\% interval of LS excludes 0 on CrisisLexT26 and on all HumAID events; "
        "H3 if the 90\\% interval of $E_5$ lies inside $\\pm0.02$, per corpus. Event counts in the first column apply "
        "to $E_2$ and LS; $E_5$ is defined only for events with at least 3{,}000 earlier tweets, whose number is in "
        "parentheses.}",
        "\\label{tab:registered}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llccc@{}}", "\\toprule",
        "Events & Encoder & $E_2$ [95\\%] & LS [95\\%] & $E_5$ [90\\%] \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def table_events(summary, corpus, label):
    by = {MODEL[r["model"]]: {e["event"]: e for e in r["per_event_seed_mean"]} for r in summary if r["corpus"] == corpus}
    events = sorted(next(iter(by.values())))
    rows = []
    for e in events:
        d, r = by.get("D", {}).get(e), by.get("R", {}).get(e)
        name = ev(e) + (" $^\\dagger$" if d["protected"] else "")
        cells = [name, str(d["n"]), str(d["n_true_classes"]), f"{d['dose']:.0f}"]
        for x in (d, r):
            cells += [f3(x["a_P"] - x["b_P"]), f3(x["c_P"] - x["c_D"]), f"{x['stray_preds']:.1f}"] if x else ["--"] * 3
        rows.append(" & ".join(cells) + " \\\\")
    dag = " $\\dagger$ protected confirmation event." if corpus == "humaid19" else ""
    return "\n".join([
        "\\begin{table}[h]", "\\centering",
        f"\\caption{{{CNAME[CORP[corpus]]}: per-event seed means. Dose: target-event tweets in the in-distribution "
        "training sample. $E_2$: matched contrast; LS: label-set component; stray: LOEO predictions of classes absent "
        f"from the event.{dag}}}", f"\\label{{{label}}}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
        " & & & & \\multicolumn{3}{c}{DistilBERT} & \\multicolumn{3}{c}{RoBERTa-base} \\\\",
        "\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
        "Event & Tweets & Classes & Dose & $E_2$ & LS & stray & $E_2$ & LS & stray \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def fig_decomposition(summary, path):
    order = sorted(summary, key=lambda r: (r["corpus"] != "humaid19", r["model"]))
    labels, comps, e0d, e0p = [], {k: [] for k in ("E2", "E1", "E3", "MC")}, [], []
    for r in order:
        b = r["subsets"]["all"]
        labels.append(f"{CNAME[CORP[r['corpus']]]}\n{MNAME[MODEL[r['model']]]}")
        for k in ("E2", "E1", "E3", "MC"):
            comps[k].append(b["P"][k]["mean"])
        e0d.append(b["D"]["E0"]["mean"])
        e0p.append(b["P"]["E0"]["mean"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    style = [("E2", "$E_2^P$ matched contrast", "#1b6ca8"), ("E1", "$E_1^P$ aggregation", "#9bbcd6"),
             ("E3", "$E_3^P$ test set", "#bdbdbd"), ("MC", "LS label-set component", "#d95f02")]
    pos, neg = np.zeros(len(x)), np.zeros(len(x))
    for k, lab, col in style:
        v = np.asarray(comps[k])
        ax.bar(x, v, 0.55, bottom=np.where(v >= 0, pos, neg), label=lab, color=col, edgecolor="white", linewidth=0.5)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    ax.scatter(x, e0d, marker="_", s=500, color="black", zorder=3, label="$E_0^D$ composite gap")
    ax.scatter(x, e0p, marker="D", s=22, color="white", edgecolor="black", zorder=3, label="$E_0^P$ present-class gap")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("macro-F1 difference")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_mechanism(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, corpus in zip(axes, ["humaid19", "crisislext26"]):
        for r in summary:
            if r["corpus"] != corpus:
                continue
            e = r["per_event_seed_mean"]
            m = MODEL[r["model"]]
            ax.scatter([x["stray_preds"] for x in e], [x["c_P"] - x["c_D"] for x in e], s=14, alpha=0.8,
                       marker="o" if m == "D" else "^", label=MNAME[m], color="#1b6ca8" if m == "D" else "#d95f02")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_title(CNAME[CORP[corpus]], fontsize=9)
        ax.set_xlabel("LOEO predictions of absent classes\nper event (seed mean)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("per-event LS", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(conv_dir, path):
    files = sorted(glob.glob(os.path.join(conv_dir, "*", "*", "*.json")))
    if not files:
        return
    models = sorted({json.load(open(f))["model"].rstrip("/").split("/")[-2] for f in files})
    fig, axes = plt.subplots(1, len(models), figsize=(6.4, 3.3), sharey=True, squeeze=False)
    handles = {}
    folds = sorted({json.load(open(f))["held_out_event_not_scored"] for f in files})
    for i, (ax, mtag) in enumerate(zip(axes[0], models)):
        ax2 = ax.twinx()
        model_curves = []
        for f in files:
            d = json.load(open(f))
            if d["model"].rstrip("/").split("/")[-2] != mtag:
                continue
            model_curves.append(d["curve"])
            ep = [c["epoch"] for c in d["curve"]]
            lab = f"{CNAME[CORP[d['corpus']]]}: {ev(d['held_out_event_not_scored'])}"
            col = f"C{folds.index(d['held_out_event_not_scored'])}"
            line, = ax.plot(ep, [c["val_macro_f1_present"] for c in d["curve"]], marker="o", ms=2, lw=0.6,
                            alpha=0.35, color=col)
            ax2.plot(ep, [c["val_loss"] for c in d["curve"]], ls="--", lw=0.5, alpha=0.3, color=col)
            handles.setdefault(lab, line)
        if model_curves:
            ep = [c["epoch"] for c in model_curves[0]]
            mean_f1 = [np.mean([c[j]["val_macro_f1_present"] for c in model_curves]) for j in range(len(ep))]
            mean_loss = [np.mean([c[j]["val_loss"] for c in model_curves]) for j in range(len(ep))]
            ax.plot(ep, mean_f1, marker="o", ms=3.5, lw=1.8, color="black", label="mean")
            ax2.plot(ep, mean_loss, ls="--", lw=1.4, color="black")
        ax.axvline(2, color="grey", lw=0.6, ls=":")
        ax.set_title(MNAME.get(MODEL.get(mtag, ""), mtag), fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7, labelleft=True)
        ax2.tick_params(labelsize=7)
        if i == len(models) - 1:
            ax2.set_ylabel("source-validation loss (dashed)", fontsize=8)
    axes[0][0].set_ylabel("source-validation macro-F1 (P, solid)", fontsize=8)
    labs = sorted(handles)
    fig.legend([handles[k] for k in labs], labs, fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="experiments/derived/summary_v2_confirm.json")
    ap.add_argument("--ep4", default="experiments/derived/summary_v2_confirm_ep4.json")
    ap.add_argument("--diagnostics", default="experiments/derived/diagnostics_confirm.json")
    ap.add_argument("--convergence", default="experiments/derived/convergence")
    ap.add_argument("--facts", default="experiments/derived/data_facts.json")
    ap.add_argument("--registered", default="experiments/derived/summary_confirm.json")
    ap.add_argument("--run_stats", default="experiments/derived/run_stats.json")
    ap.add_argument("--aggregation", default="experiments/derived/aggregation_mechanism.json")
    ap.add_argument("--x3", default="experiments/derived/summary_v2_x3_tfidf.json")
    ap.add_argument("--revision", default="experiments/derived/revision_tmlr.json")
    ap.add_argument("--x7", default="experiments/derived/revision_x7.json")
    ap.add_argument("--x8", default="experiments/derived/revision_x8.json")
    ap.add_argument("--x9", default="experiments/derived/revision_x9.json")
    ap.add_argument("--x10", default="experiments/derived/x10_zeroshot.json")
    ap.add_argument("--ids_check", default="experiments/derived/training_ids_check.json")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    summary = json.load(open(args.summary))
    ep4 = json.load(open(args.ep4)) if args.ep4 and os.path.exists(args.ep4) else None
    agg_raw = json.load(open(args.aggregation)) if args.aggregation and os.path.exists(args.aggregation) else {}
    agg, rare_rel = agg_raw.get("groups"), agg_raw.get("rare_class_vs_reweighting", {})
    x3 = json.load(open(args.x3)) if args.x3 and os.path.exists(args.x3) else None
    diags = json.load(open(args.diagnostics)) if args.diagnostics and os.path.exists(args.diagnostics) else None
    gen, figs = os.path.join(args.out, "generated"), os.path.join(args.out, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    facts = json.load(open(args.facts)) if args.facts and os.path.exists(args.facts) else None
    registered = json.load(open(args.registered)) if args.registered and os.path.exists(args.registered) else None
    kv = keyvals(summary, diags, ep4, facts, args.convergence, registered)
    kv.update(aggregation_keys(agg, summary))
    kv.update(per_class_keys(agg))
    for corpus, rel in rare_rel.items():
        c = CORP[corpus]
        kv[f"rarerel-{c}-n"], kv[f"rarerel-{c}-rho"] = str(rel["n_runs"]), f2(rel["spearman_f1_reweighting"])
        for band, b in rel["bands"].items():
            for k in ("n", "f1_min", "f1_max", "rw_min", "rw_max"):
                kv[f"rarerel-{c}-{band}-{k.replace('_', '')}"] = str(b[k]) if k == "n" else f3(b[k])
    kv.update(x3_keys(x3))
    rev = json.load(open(args.revision)) if args.revision and os.path.exists(args.revision) else None
    kv.update(revision_keys(rev, summary, ep4, x3))
    x7 = json.load(open(args.x7)) if args.x7 and os.path.exists(args.x7) else None
    kv.update(x7_keys(x7))
    x8 = json.load(open(args.x8)) if args.x8 and os.path.exists(args.x8) else None
    kv.update(x8_keys(x8))
    x9 = json.load(open(args.x9)) if os.path.exists(args.x9) else None
    x10 = json.load(open(args.x10)) if os.path.exists(args.x10) else None
    ids_check = json.load(open(args.ids_check)) if os.path.exists(args.ids_check) else None
    if x9:
        import note_assets
        kv.update(note_assets.main(args.out))
    kv.update(ls_concentration_keys(summary))
    rs = json.load(open(args.run_stats))
    kv["n-runs"] = f"{rs['n_runs']:,}".replace(",", "{,}")
    kv["gpu-hours"] = f"{rs['hours']:.0f}"
    if "n_tfidf_runs" in rs:
        kv["n-tfidf-runs"] = str(rs["n_tfidf_runs"])
    with open(os.path.join(gen, "results.tex"), "w") as f:
        f.write("% generated by code/make_paper_assets.py -- do not edit\n\\makeatletter\n")
        for k in sorted(kv):
            f.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{kv[k]}}}\n")
        f.write("\\newcommand{\\res}[1]{\\@ifundefined{res@#1}{\\textbf{??\\detokenize{#1}}}{\\csname res@#1\\endcsname}}\n\\makeatother\n")
        ep4_complete = all(kv.get(f"x7-ep4-{m}-nseeds") == "5" for m in ("D", "R"))
        f.write("\\newcommand{\\IfEpFourComplete}[2]{#%d}\n" % (1 if ep4_complete else 2))
    tabs = {"tab_decomposition": table_decomposition(summary), "tab_hypotheses": table_hypotheses(summary, kv),
            "tab_margins": table_margins(summary),
            "tab_consequence": table_consequence(summary, kv), "tab_secondary": table_secondary(summary, kv),
            "tab_masking": table_masking(summary),
            "tab_seeds": table_seeds(summary), "tab_registered": table_registered(kv),
            "tab_intervals": table_intervals(summary),
            "tab_aggregation": table_aggregation(agg), "tab_exploratory": table_exploratory(summary, ep4, x3, agg),
            "tab_perclass": table_perclass(agg)}
    if rev and x7:
        tabs.update({"tab_practices": table_practices(rev, summary, x7), "tab_merged": table_merged(rev, summary),
                     "tab_seedaware": table_seedaware(rev, summary), "tab_evalterms": table_evalterms(x7),
                     "tab_ep4seeds": table_ep4_seeds(x7), "tab_recall": table_recall(x7)})
    for corpus, label in (("humaid19", "tab:events-h"), ("crisislext26", "tab:events-c")):
        if any(r["corpus"] == corpus for r in summary):
            tabs[f"tab_events_{corpus}"] = table_events(summary, corpus, label)
    if x8:
        tabs["tab_arms"] = table_arms(x8)
        fig_instability(x8, os.path.join(figs, "fig_instability.pdf"))
    for name, body in tabs.items():
        open(os.path.join(gen, name + ".tex"), "w").write(body + "\n")
    fig_decomposition(summary, os.path.join(figs, "fig_decomposition.pdf"))
    fig_mechanism(summary, os.path.join(figs, "fig_mechanism.pdf"))
    fig_convergence(args.convergence, os.path.join(figs, "fig_convergence.pdf"))
    print(f"wrote {len(kv)} keys, {len(tabs)} tables, figures")


if __name__ == "__main__":
    main()
