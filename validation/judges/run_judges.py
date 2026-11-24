"""Run the analysis pre-registered in docs/prereg/PREREG_JUDGES.md (single run; results to validation/judges/results.json).

    python validation/judges/run_judges.py --jobs 8
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from baseline_increment.core import audit  # noqa: E402
from baseline_increment.ordinal import audit_ordinal, concordance  # noqa: E402

FAST = dict(seeds=range(2), n_perm=200, perm_stop=10, n_jobs=1, null="residual_gbm")
J2_JUDGES = ["hhem_2_1_risk", "gpt_3_5_turbo_risk", "gpt_4_turbo_risk", "gpt_4o_risk", "trueteacher_risk", "true_nli_risk"]


def _ordinal(df, judge, label, group, base):
    d = df.dropna(subset=[judge]).reset_index(drop=True)
    r = audit_ordinal(d, judge, label, group, base, n_boot_desc=200, **FAST)
    return {"increment": r["increment"], "p": r["increment_perm_p"], "adds": r["adds_information"],
            "raw_concordance": concordance(d[label].to_numpy(), d[judge].to_numpy(float)),
            "baselines_concordance": r["baselines_combined_c"], "headline": r["headline_family"], "n": len(d)}


def _binary(df, judge, label, group, base):
    d = df.dropna(subset=[judge]).reset_index(drop=True)
    r = audit(d, judge, label, group, base, n_boot=0, n_boot_desc=200, **FAST)
    return {"increment": r["increment"], "p": r["increment_perm_p"], "adds": r["adds_information"],
            "raw_concordance": r["detector_auc"], "baselines_concordance": r["baselines_combined_auc"],
            "headline": r["headline_family"], "n": len(d)}


def tasks():
    j1 = pd.read_csv(HERE / "data/j1_summeval.csv")
    j2 = pd.read_csv(HERE / "data/j2_faithbench.csv")
    j3 = pd.read_csv(HERE / "data/j3_llmjudge.csv")
    j3_judges = [c for c in j3.columns if c not in ("qid", "docid", "human", "log_words")]
    out = []
    for dim in ("consistency", "coherence", "fluency", "relevance"):
        out.append(("J1", dim, "geval_gpt4", _ordinal, j1, f"geval_{dim}", f"human_{dim}", "doc_id",
                    ["system_id", "log_words", f"d_doc_{dim}"]))
    for lab in ("hallucinated", "hallucinated_or_questionable"):
        for jd in J2_JUDGES:
            out.append(("J2", lab, jd, _binary, j2, jd, lab, "source_id", ["summarizer", "log_words", f"d_src_{lab}"]))
    for jd in j3_judges:
        out.append(("J3", "relevance", jd, _ordinal, j3, jd, "human", "docid", ["qid", "log_words"]))
    return out, j3, j3_judges


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = run
    return adj


def _control_one(j3, judge, rep):
    rng = np.random.default_rng(880_000 + rep)
    d = j3.copy()
    strata = d.qid.astype(str) + "|" + pd.qcut(d.log_words.rank(method="first"), 10, labels=False).astype(str)
    s = d[judge].to_numpy().copy()
    for ix in pd.Series(range(len(d))).groupby(strata.to_numpy()).indices.values():
        s[ix] = s[rng.permutation(ix)]
    d["perm"] = s
    r = audit_ordinal(d, "perm", "human", "docid", ["qid", "log_words"], n_boot_desc=20, **FAST)
    return bool(r["adds_information"])


def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    from joblib import Parallel, delayed
    T, j3, j3_judges = tasks()
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs)(delayed(fn)(df, judge, lab, grp, base) for (_, _, _, fn, df, judge, lab, grp, base) in T)
    out = {"prereg": "docs/prereg/PREREG_JUDGES.md", "settings": {k: (list(v) if isinstance(v, range) else v) for k, v in FAST.items()},
           "audits": {}}
    for (s, outcome, jd, *_), r in zip(T, res):
        out["audits"].setdefault(s, {}).setdefault(outcome, {})[jd] = r
    summary = {}
    for s, outs in out["audits"].items():
        for outcome, js in outs.items():
            ps = np.array([v["p"] for v in js.values()])
            adj = holm(ps)
            for (k, v), pa in zip(js.items(), adj):
                v["p_holm"] = float(pa)
                v["adds_holm"] = bool(pa < 0.05 and v["increment"] > 0)
            summary[f"{s}/{outcome}"] = {"judges": len(js), "adds_raw": sum(v["adds"] for v in js.values()),
                                         "adds_holm": sum(v["adds_holm"] for v in js.values())}
    j3r = out["audits"]["J3"]["relevance"]
    raw = [j3r[j]["raw_concordance"] for j in j3_judges]
    inc = [j3r[j]["increment"] for j in j3_judges]
    tau = kendalltau(raw, inc).correlation
    rank_raw = pd.Series(raw, index=j3_judges).rank(ascending=False)
    rank_inc = pd.Series(inc, index=j3_judges).rank(ascending=False)
    moved = int(((rank_raw - rank_inc).abs() >= 5).sum())
    out["Q2"] = {"kendall_tau": float(tau), "H2_supported": bool(tau < 0.80), "judges_moving_5_or_more": moved,
                 "top_by_raw": rank_raw.idxmin(), "top_by_increment": rank_inc.idxmin(),
                 "same_top": bool(rank_raw.idxmin() == rank_inc.idxmin())}
    out["summary"] = summary
    flags = Parallel(n_jobs=a.jobs)(delayed(_control_one)(j3, "NISTRetrieval-instruct0", r) for r in range(100))
    k = int(sum(flags))
    out["control"] = {"judge": "NISTRetrieval-instruct0", "reps": 100, "flag_rate": k / 100, "wilson95": wilson(k, 100),
                      "pass": k / 100 <= 0.10}
    out["wall_sec"] = round(time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({"summary": summary, "Q2": out["Q2"], "control": out["control"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
