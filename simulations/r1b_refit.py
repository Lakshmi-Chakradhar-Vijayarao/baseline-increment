"""R1b: coverage of a refit-aware interval for the held-out gain (docs/prereg/PREREG_R1b.md).

Per coverage replicate (the same 200 datasets as R1, seed salt 31):
  - the production audit at the frozen R1 settings: gain and clustered DeLong interval;
  - B = 100 refit group-bootstrap draws with the production ``_boot_once`` (groups resampled with replacement, copies keep
    their group id so they share a fold, folds re-drawn, every family refitted for both models, headline re-chosen);
  - for D-series and control cells, the conditional gain of the fitted fold models on a fresh sample (secondary).
Checkpointed every chunk of replicates, so a timed-out session loses at most one chunk.

    python simulations/r1b_refit.py --cells D4@3000 --reps 0-100 --jobs 4 --out simulations/results/r1b_part.json
    python simulations/r1b_refit.py --merge a.json b.json --r1 simulations/results/r1_coverage_full.json --out simulations/results/r1b.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from baseline_increment import audit  # noqa: E402
from baseline_increment.core import (_boot_once, _features, _fit_predict, auc, group_folds, split_design,  # noqa: E402
                                     transform_score, FAMILIES)
import diag_suite  # noqa: E402
from r1_coverage import FROZEN, SALT_COV, make, wilson  # noqa: E402

B = 100
N_FRESH = 20_000
SALT_FRESH = 33
CHUNK = 8


def conditional_gain(df, fresh, head, seeds=range(2)):
    """Mean over seeds x folds of AUROC(model with s) - AUROC(model without s), each fold model fitted on its training
    folds of ``df`` and scored on ``fresh`` (an independent sample from the same generator)."""
    both = pd.concat([df, fresh], ignore_index=True)
    n = len(df)
    num, lin = split_design(both, ["task", "L1", "L2"])
    s = np.concatenate([transform_score(df, "s"), transform_score(fresh, "s")])
    y = both["y"].to_numpy(int)
    groups = df["g"].astype(str).to_numpy()
    te = np.zeros(len(both), bool)
    te[n:] = True
    d = []
    for sd in seeds:
        for fold in group_folds(groups, sd):
            tr = np.zeros(len(both), bool)
            tr[:n] = ~fold
            pb = _fit_predict(head, _features(head, num, lin, None, tr), y, tr, te, sd)
            pf = _fit_predict(head, _features(head, num, lin, s, tr), y, tr, te, sd)
            d.append(auc(y[te], pf) - auc(y[te], pb))
    return float(np.mean(d))


def one(cell, rep):
    df, base, kw = make(cell, rep, SALT_COV)
    r = audit(df, "s", "y", "g", base, (), **FROZEN, **kw)
    y = np.asarray(df["y"], int)
    s = transform_score(df, "s", "rank", None)
    groups = np.asarray(df["g"]).astype(str)
    num, lin = split_design(df, base)
    t0 = time.time()
    draws = [_boot_once(num, lin, s, y, groups, tuple(FAMILIES), [0, 1], 1000 + b)[0] for b in range(B)]
    out = {"rep": rep, "inc": r["increment"], "delong_ci": r["increment_ci"], "head": r["headline_family"],
           "boot_sd": float(np.std(draws, ddof=1)), "boot_q": [float(np.percentile(draws, 2.5)),
                                                               float(np.percentile(draws, 97.5))],
           "boot_mean": float(np.mean(draws)), "boot_sec": round(time.time() - t0, 1)}
    if "@" in cell:
        c, n = cell.split("@")
        rng = np.random.default_rng(10_000_019 * SALT_FRESH + 1_000 * rep + int(n))
        out["cond_gain"] = conditional_gain(df, diag_suite.gen(c, N_FRESH, rng), r["headline_family"])
    return out


def run(a):
    from joblib import Parallel, delayed
    lo, hi = map(int, a.reps.split("-"))
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"prereg": "docs/prereg/PREREG_R1b.md", "B": B, "cells": {}}
    for cell in a.cells.split(","):
        rows = out["cells"].setdefault(cell, {})
        todo = [r for r in range(lo, hi) if str(r) not in rows]
        for i in range(0, len(todo), CHUNK):
            t0 = time.time()
            res = Parallel(n_jobs=a.jobs)(delayed(_safe)(cell, r) for r in todo[i:i + CHUNK])
            for x in res:
                rows[str(x["rep"])] = x
            json.dump(out, open(a.out, "w"))
            print(cell, f"{len(rows)} reps", f"{time.time() - t0:.0f}s/chunk", flush=True)


def _safe(cell, rep):
    for _ in (0, 1):
        try:
            return one(cell, rep)
        except Exception as e:  # technical failure: one retry with the same seed, then report
            err = repr(e)[:200]
    return {"rep": rep, "error": err}


def summarise(rows, ref):
    ok = [x for x in rows if "error" not in x]
    def cov(f):
        hit = [f(x)[0] <= ref <= f(x)[1] for x in ok]
        k, n = int(sum(hit)), len(hit)
        rate = k / n
        st = "pass" if 0.85 <= rate <= 0.99 else "under-covers" if rate < 0.85 else "over-conservative"
        return {"coverage": rate, "wilson95": wilson(k, n), "status": st,
                "miss_low": int(sum(f(x)[1] < ref for x in ok)), "miss_high": int(sum(f(x)[0] > ref for x in ok))}
    refit = lambda x: (x["inc"] - 1.959964 * x["boot_sd"], x["inc"] + 1.959964 * x["boot_sd"])
    s = {"n": len(ok), "technical_failures": len(rows) - len(ok), "reference_gain": ref,
         "refit": cov(refit), "delong": cov(lambda x: x["delong_ci"]),
         "refit_percentile_descriptive": cov(lambda x: x["boot_q"]),
         "mean_width_refit": float(np.mean([2 * 1.959964 * x["boot_sd"] for x in ok])),
         "mean_width_delong": float(np.mean([x["delong_ci"][1] - x["delong_ci"][0] for x in ok])),
         "sd_of_gain_across_reps": float(np.std([x["inc"] for x in ok], ddof=1)),
         "mean_boot_sd": float(np.mean([x["boot_sd"] for x in ok]))}
    if all("cond_gain" in x for x in ok):
        hit = [x["delong_ci"][0] <= x["cond_gain"] <= x["delong_ci"][1] for x in ok]
        s["delong_conditional_coverage"] = {"coverage": float(np.mean(hit)), "wilson95": wilson(int(sum(hit)), len(hit))}
    return s


def merge(a):
    r1 = json.load(open(a.r1))["cells"]
    rows = {}
    for f in a.merge:
        for cell, d in json.load(open(f))["cells"].items():
            rows.setdefault(cell, {}).update(d)
    cells = {c: summarise(list(d.values()), r1[c]["reference_gain"]) for c, d in rows.items()}
    json.dump({"prereg": "docs/prereg/PREREG_R1b.md", "B": B, "cells": cells}, open(a.out, "w"), indent=1)
    for c, s in cells.items():
        print(c, s["n"], "refit", round(s["refit"]["coverage"], 3), s["refit"]["status"],
              "| delong", round(s["delong"]["coverage"], 3), "| cond", s.get("delong_conditional_coverage", {}).get("coverage"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells")
    ap.add_argument("--reps", default="0-200")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--r1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a) if a.merge else run(a)
