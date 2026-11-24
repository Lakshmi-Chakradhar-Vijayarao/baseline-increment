"""Validation of the omitted-baseline sensitivity analysis (docs/prereg/PREREG_0.13.md, Part 1 + addendum 1).

Each data-generating process has a real omitted variable Z. The audit runs without Z; the erasure index phi_Z is
computed for Z; then Z is added as a baseline and the actual increment is measured.

    python3 simulations/sensitivity_suite.py --jobs 8 --null residual_sl --seed-salt 13 --out simulations/results/sensitivity_v0.13.0.json
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment import audit  # noqa: E402
from baseline_increment.core import resolve_null, split_design, transform_score  # noqa: E402
from baseline_increment.sensitivity import phi_of, residuals  # noqa: E402

FAMILIES_DGP = ["linear", "nonlinear", "categorical", "interaction"]
REGIMES = ["none", "half", "all"]
SIZES = [600, 2000]
TASK = {"A": 0.8, "B": -0.4, "C": -0.4}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def gen(fam, regime, n, rng):
    G = n // 2
    g = np.repeat(np.arange(G), 2)
    L = np.repeat(rng.normal(size=G), 2)
    task = np.array(list("ABC"))[g % 3]
    if fam == "categorical":
        zc = rng.choice(list("PQRS"), n)
        fz = np.vectorize({"P": 1.2, "Q": 0.4, "R": -0.4, "S": -1.2}.get)(zc)
        Z = zc
    else:
        zn = rng.normal(size=n)
        fz = {"linear": 1.0 * zn, "nonlinear": 1.5 * np.tanh(2 * zn),
              "interaction": 1.0 * zn * np.where(task == "A", 1.0, -1.0)}[fam]
        Z = zn
    y = (rng.random(n) < _sig(0.8 * L + np.vectorize(TASK.get)(task) + fz)).astype(int)
    e = rng.normal(size=n)
    s = {"none": e + 0.9 * y, "half": 0.5 * fz + 0.6 * y + e, "all": fz + e}[regime]
    return pd.DataFrame({"g": g, "L": L, "task": task, "Z": Z, "y": y, "s": s})


def one(fam, regime, n, rep, null, salt):
    rng = np.random.default_rng(40_000 * (1 + zlib.crc32(f"{fam}|{regime}|{n}".encode()) % 1000) + rep + 7_919_000 * salt)
    df = gen(fam, regime, n, rng)
    kw = dict(seeds=range(2), n_perm=200, perm_stop=10, n_boot=0, n_boot_desc=20, null=null)
    a0 = audit(df, "s", "y", "g", ["L", "task"], **kw)
    out = {"adds_without": a0["adds_information"], "inc_without": a0["increment"]}
    if not a0["adds_information"]:
        return out
    groups = df.g.astype(str).to_numpy()
    y = df.y.to_numpy()
    s = transform_score(df, "s", "rank")
    num, lin = split_design(df, ["L", "task"])
    model = "super_learner" if resolve_null(null) == "residual_sl" else "gbm"
    r_s, r_y = residuals(a0["headline_family"], num, lin, s, y, groups, model)
    zs, zy = (r_s - r_s.mean()) / r_s.std(), (r_y - r_y.mean()) / r_y.std()
    r = float(np.mean(zs * zy))
    nz, lz = split_design(df, ["Z"])
    na, la = split_design(df, ["L", "task", "Z"])
    ph = phi_of(r_s, r_y, r, num, lin, np.hstack([nz, lz]), na, la, groups, model)
    a1 = audit(df, "s", "y", "g", ["L", "task", "Z"], **kw)
    out.update({"r": r, "alpha_Z": ph["alpha"], "beta_Z": ph["beta"], "phi_Z": ph["phi"], "phi_linear": ph["phi_linear"],
                "phi_flexible": ph["phi_flexible"], "inc_with": a1["increment"], "adds_with": a1["adds_information"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--null", default="residual")
    ap.add_argument("--seed-salt", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    import baseline_increment
    out = {"version": baseline_increment.__version__, "null": a.null, "seed_salt": a.seed_salt, "reps": a.reps, "cells": {}}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    for fam in FAMILIES_DGP:
        for regime in REGIMES:
            for n in SIZES:
                key = f"{fam}|{regime}|{n}"
                if key in out["cells"]:
                    continue
                t0 = time.time()
                res = Parallel(n_jobs=a.jobs)(delayed(one)(fam, regime, n, r, a.null, a.seed_salt) for r in range(a.reps))
                out["cells"][key] = {"runs": res, "sec": round(time.time() - t0)}
                used = [x for x in res if x["adds_without"]]
                print(key, f"used {len(used)}/{a.reps}", "mean phi", round(float(np.mean([x['phi_Z'] for x in used])), 3) if used else None,
                      "erased", sum(x["inc_with"] <= 0 for x in used), flush=True)
                json.dump(out, open(a.out, "w"), indent=1, default=float)
    # verdicts (PREREG_0.13 addendum 1)
    allr = [(k.split("|")[0], x) for k, c in out["cells"].items() for x in c["runs"] if x["adds_without"]]
    weak = [(f, x) for f, x in allr if x["phi_Z"] <= 0.5]
    v1_pooled = float(np.mean([x["inc_with"] > 0 for _, x in weak])) if weak else None
    v1_fam = {f: float(np.mean([x["inc_with"] > 0 for g, x in weak if g == f])) for f in FAMILIES_DGP
              if any(g == f for g, _ in weak)}
    strong = [x for _, x in allr if x["phi_Z"] >= 1]
    v2 = float(np.mean([not x["adds_with"] for x in strong])) if strong else None
    drop = [1 - x["inc_with"] / x["inc_without"] for _, x in allr]
    v3 = float(spearmanr([x["phi_Z"] for _, x in allr], drop).correlation) if len(allr) > 2 else None
    out["verdict"] = {"V1_weak_n": len(weak), "V1_pooled": v1_pooled, "V1_by_family": v1_fam,
                      "V1_pass": bool(v1_pooled is not None and v1_pooled >= 0.95 and all(v >= 0.90 for v in v1_fam.values())),
                      "V2_strong_n": len(strong), "V2_share_not_detected_after_adding_Z": v2, "V3_spearman_phi_vs_drop": v3}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
