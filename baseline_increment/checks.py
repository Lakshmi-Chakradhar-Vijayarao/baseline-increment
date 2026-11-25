"""`check`: try to break a detector's claim with cheap alternative explanations.

For each explanation the data allow (question difficulty, response length, task/source, model identity, all of
them together) it measures the detector's increment with ``audit`` and assigns a status:

* PASS           the lower bound of the validated 95% interval for the gain exceeds min_increment (default 0);
* FAIL           the upper bound is below min_increment;
* INCONCLUSIVE   anything else;
(0.13: statuses come from the held-out gain and its interval only; the permutation p is reported, not used)
* NOT CHECKABLE  the data lack the needed column; the reason is stated.

It also runs a same-question test: when several models answer the same question, does the detector (ranked within
each model) separate the model that failed from the one that did not? This is the design of the H-1 study
(decision-point, Nov 2026). Statements are fixed sentences tied to these rules; nothing else is generated.
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from .core import VERSION, audit, auc

DATA = os.path.join(os.path.dirname(__file__), "data")
BENCHMARKS = ("triviaqa", "nq_open", "popqa", "sciq", "ragtruth", "helm_nq")
NOT_CHECKABLE_FROM_SCORES = [
    "Train/test leakage or test-set tuning of the detector (needs the detector's training data and selection history).",
    "Out-of-distribution generalisation (needs a second dataset).",
    "Label quality (needs a second, independent labelling of the same items).",
    "Downstream usefulness, e.g. whether acting on the flags improves outcomes (needs an intervention experiment).",
]


def qhash(text):
    """WQA's question key: sha1 of the lower-cased, whitespace-collapsed question text."""
    return hashlib.sha1(re.sub(r"\s+", " ", str(text).strip().lower()).encode()).hexdigest()


def _match_keys(texts, table_keys):
    """qhash of each question, falling back to the text with / without a trailing '?' when only that variant is in
    the table (source datasets differ: NQ-open drops the '?', HELM keeps it). An exact match always wins."""
    table_keys = set(table_keys)
    out = []
    for t in texts:
        base = str(t).strip()
        bare = re.sub(r"\s*\?+$", "", base)
        for cand in (base, bare, bare + "?"):
            k = qhash(cand)
            if k in table_keys:
                break
        else:
            k = qhash(base)
        out.append(k)
    return pd.Series(out)


def load_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; available: {', '.join(BENCHMARKS)}")
    t = pd.read_csv(os.path.join(DATA, f"difficulty_{name}.csv"), dtype={"source_id": str})
    meta = json.load(open(os.path.join(DATA, "difficulty_meta.json")))[name]
    if name == "ragtruth":
        return t, meta
    cols = [c for c in ["share_wrong", "rasch_difficulty", "answer_dispersion", "mean_risk_rank"] if c in t.columns]
    return t.groupby("qhash")[cols].mean().reset_index(), meta          # duplicate question texts: averaged


def _status(rep, padj, min_inc):
    """0.13: from the validated interval only. PASS: lower 95% bound > min_gain; FAIL: upper bound < min_gain."""
    ci = rep["increment_ci"]
    if ci[0] > min_inc:
        return "PASS"
    if ci[1] < min_inc:
        return "FAIL"
    return "INCONCLUSIVE"


def _holm(ps):
    order = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def pair_auc(y, z, q):
    """Within-question pairwise AUROC over (wrong, right) pairs of different rows on the same question."""
    num = den = 0.0
    for k in np.unique(q):
        m = q == k; w = z[m][y[m] == 1]; r = z[m][y[m] == 0]
        if len(w) and len(r):
            d = w[:, None] - r[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den if den else float("nan")


def same_question(df, score, label, question, model, n_boot=200, seed=0, min_questions=30):
    """Pair AUROC (score ranked within model) with a question bootstrap CI; conditional logit with model fixed
    effects when statsmodels is installed."""
    y = np.asarray(df[label], int); q = np.asarray(df[question]).astype(str); m = np.asarray(df[model]).astype(str)
    z = np.empty(len(y))
    for k in np.unique(m):
        s = np.asarray(df[score], float)[m == k]
        z[m == k] = (pd.Series(s).rank().to_numpy() - 0.5) / len(s)
    uq, inv = np.unique(q, return_inverse=True)
    mixed = ((np.bincount(inv, y) > 0) & (np.bincount(inv, y) < np.bincount(inv)))[inv]
    nq = int(len(np.unique(q[mixed])))
    if nq < min_questions:
        return {"status": "NOT CHECKABLE", "reason": f"only {nq} questions have both a failing and a succeeding model "
                                                     f"(need {min_questions})", "n_mixed_questions": nq}
    y, z, q, m = y[mixed], z[mixed], q[mixed], m[mixed]
    out = {"pair_auc": float(pair_auc(y, z, q)), "n_mixed_questions": nq, "n_models": int(len(np.unique(m)))}
    rng = np.random.default_rng(seed); groups = {k: np.flatnonzero(q == k) for k in np.unique(q)}; keys = list(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        qq = np.concatenate([np.full(len(groups[keys[i]]), j) for j, i in enumerate(pick)])
        boots.append(pair_auc(y[idx], z[idx], qq))
    out["pair_auc_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit
        lv = sorted(set(m))[1:]
        X = np.column_stack([np.array([[a == b for b in lv] for a in m], float), (z - z.mean()) / z.std()])
        X = X[:, X.std(0) > 0]
        r = ConditionalLogit(y, X, groups=q).fit(disp=0, method="lbfgs", maxiter=1000)
        out["beta"] = float(r.params[-1]); out["p_wald"] = float(r.pvalues[-1])
    except ImportError:
        out["beta"] = out["p_wald"] = None
    lo, hi = out["pair_auc_ci"]
    beta_ok = out["beta"] is None or (out["beta"] > 0 and out["p_wald"] < 0.05)
    out["status"] = ("PASS" if lo > 0.5 and out["pair_auc"] >= 0.55 and beta_ok
                     else "FAIL" if hi < 0.55 else "INCONCLUSIVE")
    return out


SENTENCE = {
    "PASS": "The detector improves held-out prediction beyond {what} (gain {inc:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]).",
    "FAIL": "Beyond {what}, the detector's gain is below {mi:+.2f} AUROC (95% CI upper bound {hi:+.3f}).",
    "INCONCLUSIVE": "Whether the detector improves held-out prediction by more than {mi:+.2f} AUROC beyond {what} is not "
                    "settled by these data (gain {inc:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]).",
}


def check(df, score, label, question=None, model=None, response=None, length=None, task=None, difficulty=(),
          benchmark=None, question_text=None, min_increment=0.0, n_boot=0, n_perm=200, seeds=2,
          families=("linear", "spline"), n_jobs=1, samples=(), panel_meta=None, panel_sensitivity_check=True):
    """Run every check the columns allow. Returns a JSON-serialisable report."""
    from .core import check_binary_label
    check_binary_label(df[label], label)
    df = df.copy()
    rep = {"tool": f"baseline-increment {VERSION}", "n_items": int(len(df)), "checks": {}, "not_checkable": [],
           "warnings": [], "settings": {"min_increment": min_increment, "n_boot": n_boot, "n_perm": n_perm,
                                        "seeds": seeds, "families": list(families)}}
    explanations = {}                                       # name -> (columns, description)

    difficulty = list(difficulty)
    if benchmark == "ragtruth":
        tab, meta = load_benchmark(benchmark)
        if not question:
            raise ValueError("--benchmark ragtruth needs --question = the RAGTruth source_id column")
        sid = df[question].astype(str).to_numpy()
        gen = df[model].astype(str).to_numpy() if model else np.full(len(df), "*")
        loo = dict(zip(zip(tab["source_id"], tab["generator"]), tab["share_hallucinated"]))
        known = set(meta["models"])
        vals = [loo.get((s_, g_ if g_ in known else "*")) for s_, g_ in zip(sid, gen)]
        df["bench_share_wrong"] = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
        match = float(df["bench_share_wrong"].notna().mean())
        n_loo = int(sum(g_ in known for g_ in gen))
        rep["benchmark"] = {"name": benchmark, "match_rate": match, "reference_models": meta["models"],
                            "label_source": meta["label_source"], "leave_one_generator_out_rows": n_loo}
        rep["difficulty_panel"] = {"source": "shipped ragtruth table", "models": meta["models"], "n_models": len(meta["models"]),
                                   "label_source": meta["label_source"]}
        if match < 0.9:
            rep["warnings"].append(f"only {match:.0%} of rows matched RAGTruth source ids; checks use matched rows only")
        if not model:
            rep["warnings"].append("no --model column: difficulty uses all RAGTruth generators; if your responses are "
                                   "RAGTruth's own, their labels are included (pass --model for leave-one-generator-out)")
        df = df[df["bench_share_wrong"].notna()].reset_index(drop=True)
        explanations["benchmark difficulty (other generators' hallucination rate)"] = (["bench_share_wrong"], "source difficulty")
    elif benchmark:
        tab, meta = load_benchmark(benchmark)
        if question_text:
            key = _match_keys(df[question_text], tab["qhash"])
        elif question and df[question].astype(str).str.fullmatch(r"[0-9a-f]{40}").all():
            key = df[question].astype(str)
        else:
            raise ValueError("--benchmark needs --question-text (question strings) or a --question column of qhash keys")
        j = pd.DataFrame({"qhash": key}).merge(tab, on="qhash", how="left")
        match = float(j["share_wrong"].notna().mean())
        rep["benchmark"] = {"name": benchmark, "match_rate": match, "reference_models": meta["models"],
                            "label_source": meta["label_source"]}
        if meta.get("protocol"):
            rep["benchmark"]["protocol"] = meta["protocol"]
            rep["warnings"].append(f"the {benchmark} table was built under a fixed protocol ({meta['protocol']}); difficulty "
                                   "transfers poorly across prompts and graders, so check the 'benchmark difficulty' baseline's "
                                   "own AUROC on your labels, and prefer difficulty from your own multi-model logs (--lomo)")
        if match == 0:
            raise ValueError(f"no rows matched the {benchmark} table (questions are matched by lower-cased, whitespace-"
                             "collapsed text, with or without a trailing '?'); check --question-text")
        if match < 0.9:
            rep["warnings"].append(f"only {match:.0%} of rows matched the {benchmark} table; checks use matched rows only")
        free = [c for c in ("answer_dispersion", "mean_risk_rank") if c in tab.columns]
        for c in ["share_wrong"] + free:
            df[f"bench_{c}"] = j[c].to_numpy()
        df = df[df["bench_share_wrong"].notna()].reset_index(drop=True)
        if model is not None and set(df[model].astype(str)) & set(meta["models"]):
            rep["warnings"].append("your model is one of the table's reference models, so the difficulty columns include its "
                                   "own labels and scores; increments over them are biased toward zero")
        rep["difficulty_panel"] = {"source": f"shipped {benchmark} table", "models": meta["models"], "n_models": len(meta["models"]),
                                   "label_source": meta["label_source"]}
        if meta.get("families"):
            rep["difficulty_panel"]["n_families"] = len(meta["families"])
        explanations["benchmark difficulty (other models' error rate)"] = (["bench_share_wrong"], "question difficulty")
        if free:
            explanations["benchmark difficulty, label-free"] = ([f"bench_{c}" for c in free],
                                                            "label-free question difficulty")
    y = np.asarray(df[label], int)
    rep["n_items_used"] = int(len(df)); rep["detector_auc"] = auc(y, df[score]); rep["positive_rate"] = float(y.mean())
    if difficulty:
        explanations["your difficulty columns"] = (difficulty, "your difficulty measure")
        if panel_meta:
            rep.setdefault("difficulty_panel", {"source": "user-supplied", **panel_meta})
        else:
            rep["warnings"].append("your difficulty columns come with no panel metadata (--panel-meta): their increments are reported "
                                   "but are never the headline, because the increment depends on how many reference models built "
                                   "the difficulty (decision-point panel-size study)")
    _p = rep.get("difficulty_panel") or {}
    if isinstance(_p.get("n_models"), (int, float)) and _p["n_models"] < 5:
        rep["warnings"].append(f"the difficulty panel has only {int(_p['n_models'])} reference models: increments over panel-relative "
                               "difficulty are several-fold larger with small panels (decision-point B4: about 0.12 AUROC at k = 1 "
                               "vs about 0.02 at k = 10). Treat this increment as an upper bound and report k with it")
    if not benchmark and not difficulty:
        rep["not_checkable"].append("Question difficulty: no --benchmark table and no --difficulty column. Difficulty is "
                                    "usually the largest cheap explanation; see the README for how to supply it.")
    if response or length:
        df["_length"] = df[response].astype(str).str.split().str.len() if response else df[length].astype(float)
        explanations["response length"] = (["_length"], "response length (words)" if response else "response length")
    else:
        rep["not_checkable"].append("Response length: no --response or --length column.")
    if task:
        explanations["task / source"] = ([task], f"'{task}'")
    else:
        rep["not_checkable"].append("Task or source identity: no --task column.")
    if model:
        if df[model].nunique() > 1:
            explanations["model identity"] = ([model], "which model produced the answer")
    else:
        rep["not_checkable"].append("Model identity and the same-question test: no --model column.")
    allcols = sorted({c for cols, _ in explanations.values() for c in cols})
    if len(explanations) > 1:
        explanations["all of the above together"] = (allcols, "all of these together")

    names = list(explanations); raw = {}
    for nm in names:
        cols, _ = explanations[nm]
        raw[nm] = audit(df, score, label, question, cols, families=families, seeds=range(seeds), n_perm=n_perm,
                        n_boot=n_boot, n_jobs=n_jobs, n_boot_desc=200)
    adj = _holm(np.array([raw[nm]["increment_perm_p"] for nm in names])) if names else []
    for nm, pa in zip(names, adj):
        r = raw[nm]; st = _status(r, pa, min_increment); lo, hi = r["increment_ci"]
        rep["checks"][nm] = {"status": st, "baseline_auc": r["baselines_combined_auc"], "with_detector_auc": r["baselines_plus_detector_auc"],
                             "increment": r["increment"], "ci": r["increment_ci"], "perm_p": r["increment_perm_p"], "p_holm": float(pa),
                             "baseline_family": r["headline_family"], "columns": explanations[nm][0],
                             "sentence": SENTENCE[st].format(what=explanations[nm][1], inc=r["increment"], lo=lo, hi=hi, mi=min_increment)}
        from .validity import incremental_dprime
        rep["checks"][nm]["incremental_dprime"] = incremental_dprime(r["baselines_combined_auc"], r["baselines_plus_detector_auc"])
        rep["warnings"] += [f"[{nm}] {w}" for w in r["warnings"] if "not distinguishable from zero" not in w]
    if task:
        br = audit(df, score, label, question, [], strata=[task], families=("linear",), seeds=range(1), n_perm=1, n_boot=0, n_boot_desc=200)
        rep["by_task"] = {k: {"n": v["n"], "positive_rate": v["positive_rate"], "detector_auc": v.get("detector_auc")}
                          for k, v in br["strata"][task]["within"].items()}
        rep["task_base_rate_only_auc"] = br["strata"][task]["base_rate_only_auc"]
    if model and question and df[model].nunique() > 1:
        sq = same_question(df, score, label, question, model, n_boot=n_boot)
        if sq["status"] == "NOT CHECKABLE":
            rep["not_checkable"].append(f"Same-question test: {sq['reason']}.")
        else:
            lo, hi = sq["pair_auc_ci"]
            sq["sentence"] = {"PASS": f"On the same question, the detector ranks the failing model above the succeeding one "
                                           f"{sq['pair_auc']:.0%} of the time (95% CI {lo:.0%}–{hi:.0%}): it carries model-specific information.",
                              "FAIL": f"On the same question, the detector barely separates failing from succeeding models "
                                               f"(pair AUROC {sq['pair_auc']:.3f}, CI upper bound {hi:.3f} < 0.55).",
                              "INCONCLUSIVE": f"Same-question separation is not settled (pair AUROC {sq['pair_auc']:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])."}[sq["status"]]
            rep["checks"]["same question, across models"] = sq
    elif model and not question:
        rep["not_checkable"].append("Same-question test: no --question column.")
    from . import validity
    if model and question and df[model].nunique() >= 3 and panel_sensitivity_check:
        try:
            rep["panel_sensitivity"] = validity.panel_sensitivity(df, question, model, label, score, seeds=range(max(1, seeds)))
        except ValueError as e:
            rep["not_checkable"].append(f"Panel-size sensitivity: {e}.")
    elif model and question:
        rep["not_checkable"].append("Panel-size sensitivity: needs >= 3 models answering the same questions.")
    if samples:
        if not (model and question):
            rep["not_checkable"].append("Error-propensity decomposition: needs --model and --question as well as --samples.")
        else:
            try:
                rep["decomposition"] = validity.decomposition(df, question, model, list(samples), scores=[score], n_boot=n_boot)
            except ValueError as e:
                rep["not_checkable"].append(f"Error-propensity decomposition: {e}.")
    else:
        rep["not_checkable"].append("Error-propensity decomposition (how much model-specific error exists, how much the detector "
                                    "recovers): needs repeated sampled-answer labels (--samples).")
    rep["not_checkable"] += NOT_CHECKABLE_FROM_SCORES
    if not question:
        rep["warnings"].append("no --question column: rows from the same question may sit in both train and test folds")
    rep["ladder"] = ladder(rep)
    rep["headline"] = headline(rep)
    rep["manifest"] = manifest(rep, dict(score=score, label=label, question=question, model=model, response=response, length=length,
                                         task=task, difficulty=list(difficulty), benchmark=benchmark, samples=list(samples)))
    return rep


RUNGS = [("same question, across models", "Same-question model discrimination (no difficulty proxy needed)"),
         ("benchmark difficulty (other models' error rate)", "Beyond panel-relative difficulty"),
         ("benchmark difficulty (other generators' hallucination rate)", "Beyond panel-relative source difficulty"),
         ("benchmark difficulty, label-free", "Beyond label-free difficulty"),
         ("your difficulty columns", "Beyond your difficulty measure"),
         ("response length", "Beyond response length (surface)"),
         ("task / source", "Beyond task / source"),
         ("model identity", "Beyond model identity"),
         ("all of the above together", "Beyond all of these together")]
NEVER_FROM_SCORES = ["Out-of-distribution generalisation", "Independent labels (second rater / judge)", "Independent replication"]


def ladder(rep):
    """Evidence ladder: one row per rung, status taken from the checks; untested rungs say NOT TESTED, never pass/fail."""
    out = []
    for key, text in RUNGS:
        c = rep["checks"].get(key)
        if c is not None:
            out.append({"rung": text, "status": c["status"]})
    d = rep.get("decomposition")
    if d:
        r2 = d["estimate"].get(next(k for k in d["estimate"] if k.startswith("R2_v")))
        out.append({"rung": "Recovers model-specific error propensity (R2_v)",
                    "status": "NOT CHECKABLE" if not d["r2_v_defined"] else f"MEASURED ({r2:.3f})"})
    else:
        out.append({"rung": "Recovers model-specific error propensity (R2_v)", "status": "NOT TESTED"})
    out += [{"rung": t, "status": "NOT TESTED"} for t in NEVER_FROM_SCORES]
    return out


def headline(rep):
    sq = rep["checks"].get("same question, across models")
    if sq and sq.get("status") != "NOT CHECKABLE":
        return {"metric": "within-question pair AUROC", "value": sq["pair_auc"], "ci": sq["pair_auc_ci"], "status": sq["status"]}
    for key in ("benchmark difficulty (other models' error rate)", "benchmark difficulty (other generators' hallucination rate)",
                "your difficulty columns"):
        c = rep["checks"].get(key)
        if c and (key != "your difficulty columns" or rep.get("difficulty_panel")):
            p = rep.get("difficulty_panel", {})
            return {"metric": f"AUROC increment over difficulty from a {p.get('n_models', '?')}-model panel", "value": c["increment"],
                    "ci": c["ci"], "status": c["status"],
                    "caveat": "panel-relative: the increment shrinks as the reference panel grows; see panel_sensitivity"}
    return {"metric": None, "status": "NO HEADLINE",
            "reason": "no same-question data and no difficulty with panel metadata; individual checks are listed below"}


def manifest(rep, args):
    import platform, sys
    vers = {}
    for m in ("numpy", "scipy", "pandas", "sklearn", "statsmodels"):
        try:
            vers[m] = __import__(m).__version__
        except Exception:
            vers[m] = None
    return {"tool": rep["tool"], "python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
            "columns": args, "settings": rep["settings"], "difficulty_panel": rep.get("difficulty_panel"),
            "complete": bool(rep.get("difficulty_panel") or not any(k in rep["checks"] for k, _ in RUNGS[1:5]))}


# ---------------------------------------------------------------- rendering
def to_text(rep):
    L = [rep["tool"], f"items {rep['n_items']}   positive rate {rep['positive_rate']:.3f}   raw detector AUROC {rep['detector_auc']:.3f}"]
    h = rep.get("headline", {})
    if h.get("metric"):
        L.append(f"HEADLINE  {h['metric']}: {h['value']:.3f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]  -> {h['status']}")
        if h.get("caveat"):
            L.append(f"          caveat: {h['caveat']}")
    else:
        L.append(f"HEADLINE  none: {h.get('reason', '')}")
    L += ["", "Evidence ladder (supported against TESTED alternatives only; NOT TESTED is not a pass):"]
    L += [f"  {r['status']:<16} {r['rung']}" for r in rep.get("ladder", [])]
    L += ["", "Details:"]
    for nm, c in rep["checks"].items():
        if "increment" in c:
            d = c.get("incremental_dprime", {})
            L.append(f"  {c['status']:<14} beyond {nm}: baseline {c['baseline_auc']:.3f} -> with detector {c['with_detector_auc']:.3f}, "
                     f"increment {c['increment']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], Holm p {c['p_holm']:.3f}; "
                     f"incremental d' {d.get('d_add', float('nan')):.2f} (model-based)")
        else:
            L.append(f"  {c['status']:<14} {nm}: pair AUROC {c['pair_auc']:.3f} [{c['pair_auc_ci'][0]:.3f}, {c['pair_auc_ci'][1]:.3f}], "
                     f"{c['n_mixed_questions']} questions" + (f", beta {c['beta']:+.2f}" if c.get("beta") is not None else ""))
    if rep.get("difficulty_panel"):
        p = rep["difficulty_panel"]; L.append(f"  difficulty panel: {p.get('source')}, {p.get('n_models', '?')} models")
    ps = rep.get("panel_sensitivity")
    if ps and "delta_by_k" in ps:
        L.append("  panel-size sensitivity (increment over D1 from k of your other models): " +
                 ", ".join(f"k={k}: {v:+.3f}" for k, v in ps["delta_by_k"].items()) +
                 (f"; extrapolated {ps['delta_extrapolated']:+.3f} (assumes c/k; in decision-point B4 the c/k fit over-predicted larger k, so the true floor may be lower)" if ps.get("delta_extrapolated") is not None else ""))
    d = rep.get("decomposition")
    if d:
        e, ci = d["estimate"], d["ci"]
        L.append(f"  error-propensity decomposition ({d['n_models']} models x {d['n_items']} items, {d['samples_per_half']} samples per half): "
                 f"model-specific share {e['model_specific_share']:.3f} [{ci['model_specific_share'][0]:.3f}, {ci['model_specific_share'][1]:.3f}], "
                 f"split-half r {e['split_half_r']:.3f}")
        nl = d.get("additive_null")
        if nl:
            L.append(f"    ability differences alone (no model x item interaction) would give {nl['null_share']:.3f} "
                     f"(range {nl['null_share_range'][0]:.3f}-{nl['null_share_range'][1]:.3f} over item spreads; ability spread "
                     f"{nl['ability_spread_logit_sd']:.2f} logit sd); excess over that: {nl['excess_share']:.3f}. "
                     "The share depends on which models are in the panel: quote it per panel.")
        for k in [k for k in e if k.startswith("R2_v")]:
            L.append(f"    detector recovers {e[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}] of model-specific propensity variance"
                     + ("" if d["r2_v_defined"] else "  (NOT DEFINED: " + d.get("note", "") + ")"))
    L += ["", "What this means:"] + [f"  - {c['sentence']}" for c in rep["checks"].values() if "sentence" in c]
    L += ["", "Not tested / not checkable:"] + [f"  - {x}" for x in rep["not_checkable"]]
    if rep["warnings"]:
        L += ["", "Warnings:"] + [f"  - {w}" for w in rep["warnings"]]
    m = rep.get("manifest", {})
    if m:
        L += ["", f"Manifest: python {m['python']}, " + ", ".join(f"{k} {v}" for k, v in m["packages"].items())
              + ("" if m.get("complete") else "  [INCOMPLETE: difficulty panel metadata missing]")]
    return "\n".join(L)


def _html_head(rep, esc):
    h = rep.get("headline", {})
    head = (f"<p><b>Headline:</b> {esc(h['metric'])} = {h['value']:.3f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}] &rarr; {esc(h['status'])}"
            + (f"<br><small>{esc(h['caveat'])}</small>" if h.get("caveat") else "") + "</p>") if h.get("metric") else \
           f"<p><b>Headline:</b> none &mdash; {esc(h.get('reason', ''))}</p>"
    lad = "".join(f"<tr><td>{esc(r['rung'])}</td><td>{esc(r['status'])}</td></tr>" for r in rep.get("ladder", []))
    return head + f"<h2>Evidence ladder</h2><p><small>Supported against tested alternatives only. NOT TESTED is not a pass.</small></p><table>{lad}</table>"


def to_html(rep):
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    color = {"PASS": "#1a7f37", "FAIL": "#cf222e", "INCONCLUSIVE": "#9a6700", "NOT CHECKABLE": "#57606a", "NOT TESTED": "#57606a"}
    rows = []
    for nm, c in rep["checks"].items():
        if "increment" in c:
            detail = (f"{c['baseline_auc']:.3f} &rarr; {c['with_detector_auc']:.3f}; increment {c['increment']:+.3f} "
                      f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]; Holm p {c['p_holm']:.3f}")
        else:
            detail = f"pair AUROC {c['pair_auc']:.3f} [{c['pair_auc_ci'][0]:.3f}, {c['pair_auc_ci'][1]:.3f}]; {c['n_mixed_questions']} questions"
        rows.append(f"<tr><td>{esc(nm)}</td><td><span class=s style='background:{color[c['status']]}'>{c['status']}</span></td>"
                    f"<td>{detail}</td></tr><tr><td></td><td colspan=2 class=sent>{esc(c['sentence'])}</td></tr>")
    li = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)
    bench = (f"<p>Benchmark table <b>{esc(rep['benchmark']['name'])}</b>: matched {rep['benchmark']['match_rate']:.0%} of rows; "
             f"reference models: {esc(', '.join(rep['benchmark']['reference_models']))}.</p>") if "benchmark" in rep else ""
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Detector check</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:880px;margin:2rem auto;padding:0 16px;color:#1f2328;background:#fff}}
table{{border-collapse:collapse;width:100%}}td{{padding:.4rem .5rem;border-top:1px solid #d0d7de;vertical-align:top}}
.s{{color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;white-space:nowrap}}.sent{{color:#57606a;border-top:none;padding-top:0}}
@media (prefers-color-scheme:dark){{body{{background:#0d1117;color:#e6edf3}}td{{border-color:#30363d}}.sent{{color:#8b949e}}}}
</style></head><body><h1>Detector check</h1>
<p>{esc(rep['tool'])} &middot; {rep['n_items']} items &middot; positive rate {rep['positive_rate']:.3f} &middot; detector AUROC <b>{rep['detector_auc']:.3f}</b></p>
{bench}{_html_head(rep, esc)}<h2>Checks</h2><table>{''.join(rows)}</table>
<h2>Not checked</h2><ul>{li(rep['not_checkable'])}</ul>
{'<h2>Warnings</h2><ul>' + li(rep['warnings']) + '</ul>' if rep['warnings'] else ''}
<p style="color:#57606a;font-size:13px">Statuses: PASS = lower bound of the validated 95% interval for the gain &gt; {rep['settings']['min_increment']};
FAIL = upper bound &lt; {rep['settings']['min_increment']}; otherwise INCONCLUSIVE. The permutation p is reported, not used. Settings: {esc(json.dumps(rep['settings']))}.</p>
</body></html>"""
