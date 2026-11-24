"""Run the real-data validation (docs/prereg/PREREG_VALIDATION.md Part A + addendum 1).

    python validation/run_validation.py [--jobs N] [--part agreement|nulls|all]

Fixed before the first run: the null controls use the first-listed detector of each table (LettuceDetect for RAGTruth
and TofuEval, P(True) for WQA), and the WQA null controls use Mistral-7B-Instruct-v0.3.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from baseline_increment import audit  # noqa: E402

FAST = dict(seeds=range(2), n_perm=200, perm_stop=10, n_boot=0, n_boot_desc=200)
PRIOR = {("ragtruth", "lettucedetect"): (0.131, 0.111, 0.161), ("ragtruth", "hhem"): (0.091, 0.071, 0.115),
         ("tofueval", "lettucedetect"): (0.089, 0.066, 0.122), ("tofueval", "hhem"): (0.095, 0.074, 0.126),
         ("wqa_triviaqa", "p_true"): (0.042, 0.038, 0.047), ("wqa_triviaqa", "logprob"): (0.038, 0.034, 0.042),
         ("wqa_triviaqa", "sem_ent"): (0.036, 0.033, 0.040)}
OUT = HERE / "results.json"


def load():
    return {n: pd.read_csv(HERE / "data" / f"{n}.csv") for n in ("ragtruth", "tofueval", "wqa_triviaqa")}


def agreement(T, jobs):
    res = {}
    rt = T["ragtruth"]
    for det in ("lettucedetect", "hhem"):
        cells = {}
        for gen, d in rt.groupby("generator"):
            r = audit(d.reset_index(drop=True), det, "hallucinated", "response_id", ["d_src"], n_jobs=jobs, **FAST)
            cells[gen] = {"increment": r["increment"], "p": r["increment_perm_p"], "adds": r["adds_information"],
                          "headline": r["headline_family"], "raw_auc": r["detector_auc"]}
        pooled = audit(rt, det, "hallucinated", "source_id", ["d_src"], n_jobs=jobs, **FAST)
        res[("ragtruth", det)] = {"cells": cells, "mean_increment": float(np.mean([c["increment"] for c in cells.values()])),
                                  "n_adds": sum(c["adds"] for c in cells.values()), "n_cells": len(cells),
                                  "secondary_pooled": {"increment": pooled["increment"], "p": pooled["increment_perm_p"],
                                                       "adds": pooled["adds_information"], "headline": pooled["headline_family"]}}
    tf = T["tofueval"]
    for det in ("lettucedetect", "hhem"):
        r = audit(tf, det, "label", "source", ["d_src"], n_jobs=jobs, **FAST)
        res[("tofueval", det)] = {"mean_increment": r["increment"], "p": r["increment_perm_p"], "n_adds": int(r["adds_information"]),
                                  "n_cells": 1, "headline": r["headline_family"], "raw_auc": r["detector_auc"]}
    wq = T["wqa_triviaqa"]
    for det in ("p_true", "logprob", "sem_ent"):
        cells = {}
        for m, d in wq.groupby("model"):
            r = audit(d.reset_index(drop=True), det, "wrong", "qid", ["d1"], n_jobs=jobs, **FAST)
            cells[m] = {"increment": r["increment"], "p": r["increment_perm_p"], "adds": r["adds_information"],
                        "headline": r["headline_family"]}
        res[("wqa_triviaqa", det)] = {"cells": cells, "mean_increment": float(np.mean([c["increment"] for c in cells.values()])),
                                      "n_adds": sum(c["adds"] for c in cells.values()), "n_cells": len(cells)}
    checks = {}
    for k, v in res.items():
        est, lo, hi = PRIOR[k]
        need = 5 if k[0] == "ragtruth" else 6 if k[0] == "wqa_triviaqa" else 1
        v["prior"] = {"increment": est, "ci": [lo, hi]}
        v["A1_verdict"] = v["n_adds"] >= need
        v["A2_magnitude"] = lo - 0.02 <= v["mean_increment"] <= hi + 0.02
        checks[f"{k[0]}/{k[1]}"] = {"A1": v["A1_verdict"], "A2": v["A2_magnitude"]}
    grounding = [v["mean_increment"] for k, v in res.items() if k[0] != "wqa_triviaqa"]
    qa = [v["mean_increment"] for k, v in res.items() if k[0] == "wqa_triviaqa"]
    a3 = min(grounding) > max(qa)
    verdict = "AGREES" if all(c["A1"] and c["A2"] for c in checks.values()) and a3 else "DISAGREES"
    return {"per_detector": {f"{k[0]}/{k[1]}": v for k, v in res.items()}, "checks": checks,
            "A3_ordering": {"min_grounding": min(grounding), "max_qa": max(qa), "pass": bool(a3)}, "verdict": verdict}


def _rank01(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 1) / (len(x) - 1)


def null_tables(T):
    rt = T["ragtruth"].copy()
    rt["stratum"] = pd.qcut(rt.d_src.rank(method="first"), 10, labels=False).astype(str) + "|" + rt.task_type
    tf = T["tofueval"].copy()
    tf["stratum"] = pd.qcut(tf.d_src.rank(method="first"), 10, labels=False).astype(str) + "|" + tf.dataset
    wq = T["wqa_triviaqa"]
    wq = wq[wq.model == "Mistral-7B-Instruct-v0.3"].reset_index(drop=True).copy()
    wq["stratum"] = pd.qcut(wq.d1.rank(method="first"), 10, labels=False).astype(str)
    return {"ragtruth": (rt, "lettucedetect", "hallucinated", "source_id", ["d_src", "task_type"]),
            "tofueval": (tf, "lettucedetect", "label", "source", ["d_src", "dataset"]),
            "wqa_triviaqa_mistral": (wq, "p_true", "wrong", "qid", ["d1"])}


def one_null(name, kind, rep, table, null="residual"):
    df, det, lab, grp, base = table
    names = ["ragtruth", "tofueval", "wqa_triviaqa_mistral", "judge_popqa", "judge_triviaqa"]
    rng = np.random.default_rng(70_000 + 1000 * names.index(name)
                                + 100_000 * ["N1", "N2", "P1"].index(kind) + rep)
    d = df.copy()
    perm = d[det].to_numpy().copy()
    for _, ix in d.groupby("stratum").indices.items():
        perm[ix] = perm[rng.permutation(ix)]
    dcol = base[0]
    if kind == "N1":
        d["s"] = perm
    elif kind == "N2":
        d["s"] = _rank01(d[dcol].to_numpy()) + rng.normal(0, 0.3, len(d))
    else:
        r = _rank01(perm)
        d["s"] = r + 0.6 * r.std() * d[lab].to_numpy()
    out = audit(d, "s", lab, grp, base, **{**FAST, "null": null})
    return bool(out["adds_information"])


def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def nulls(T, jobs, reps=100, tabs=None, key="nulls"):
    from joblib import Parallel, delayed
    tabs = null_tables(T) if tabs is None else tabs
    out = json.load(open(OUT)).get(key, {}) if OUT.exists() else {}
    for name, tab in tabs.items():
        for kind in ("N1", "N2", "P1"):
            key = f"{name}/{kind}"
            if key in out:
                continue
            t0 = time.time()
            flags = Parallel(n_jobs=jobs)(delayed(one_null)(name, kind, r, tab, FAST.get("null", "residual")) for r in range(reps))
            k = sum(flags)
            out[key] = {"reps": reps, "flag_rate": k / reps, "wilson95": wilson(k, reps),
                        "rule": "<= 0.10" if kind != "P1" else "power, reported",
                        "verdict": ("pass" if k / reps <= 0.10 else "FAIL") if kind != "P1" else "reported",
                        "wall_sec": round(time.time() - t0)}
            print(key, out[key], flush=True)
            save({key: out})
    return out


JUDGE_NULL_MODEL = "Qwen2.5-7B-Instruct"     # in both datasets; fixed before any judge run


def judge_tables():
    j = pd.read_csv(HERE / "data" / "wqa_judge.csv")
    tabs = {}
    for ds in ("popqa", "triviaqa"):
        d = j[(j.dataset == ds) & (j.model == JUDGE_NULL_MODEL)].reset_index(drop=True).copy()
        d["stratum"] = pd.qcut(d.d1.rank(method="first"), 10, labels=False).astype(str)
        tabs[f"judge_{ds}"] = (d, "falcon_risk", "wrong", "qid", ["d1", "log_answer_words"])
    return tabs


def judge_increments(jobs):
    j = pd.read_csv(HERE / "data" / "wqa_judge.csv")
    res = {}
    for (ds, m), d in j.groupby(["dataset", "model"]):
        d = d.reset_index(drop=True)
        for det in ("falcon_risk", "glm_risk"):
            r = audit(d, det, "wrong", "qid", ["d1", "log_answer_words"], n_jobs=jobs, **FAST)
            res[f"{ds}/{det}/{m}"] = {"increment": r["increment"], "p": r["increment_perm_p"], "adds": r["adds_information"],
                                      "headline": r["headline_family"], "raw_auc": r["detector_auc"],
                                      "baselines_auc": r["baselines_combined_auc"]}
            print(ds, det, m, res[f"{ds}/{det}/{m}"], flush=True)
    return res


def save(part):
    cur = json.load(open(OUT)) if OUT.exists() else {}
    cur.update(part)
    json.dump(cur, open(OUT, "w"), indent=1, default=float)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--part", default="all")
    ap.add_argument("--null", default="residual")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    FAST["null"] = a.null
    T = load()
    if a.part in ("agreement", "all"):
        ag = agreement(T, a.jobs)
        save({"agreement": ag})
        print(json.dumps({"checks": ag["checks"], "A3": ag["A3_ordering"], "verdict": ag["verdict"]}, indent=1), flush=True)
    if a.part in ("nulls", "all"):
        nulls(T, a.jobs)
    if a.part in ("judge", "all"):
        save({"judge_increments": judge_increments(a.jobs)})
        nulls(T, a.jobs, tabs=judge_tables(), key="judge_nulls")


if __name__ == "__main__":
    main()
