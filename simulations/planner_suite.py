"""Validation of the sample-size planner (docs/prereg/PREREG_0.13.md, Part 2).

For each scenario: the planner recommends N; at N/2, N and 2N the planner's predicted power (its own generator,
R = 200) is compared with (P1) a fresh simulation of the same generator and (P2) an independent, misspecified
generator matched to the same prevalence, baselines-only AUROC, group size and population increment (R = 400).

    python3 simulations/planner_suite.py --jobs 8 --null residual_sl --seed-salt 13 --out simulations/results/planner_v0.13.0.json
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import baseline_increment.planner as P  # noqa: E402
from baseline_increment import audit  # noqa: E402
from baseline_increment.core import _rank01, auc, cv_predict, split_design  # noqa: E402

SCEN = {"S1": (0.3, 0.70, 2, 0.03), "S2": (0.1, 0.80, 1, 0.05), "S3": (0.5, 0.65, 4, 0.02)}
TASK = np.array([1.0, -0.5, -0.5])


def _sig(x):
    return 1 / (1 + np.exp(-x))


def make_alt(n_groups, m, a, b, theta, rng):
    t = rng.integers(0, 3, n_groups)
    D = rng.normal(size=n_groups)
    t, D = np.repeat(t, m), np.repeat(D, m)
    y = (rng.random(len(D)) < _sig(a + b * (0.6 * TASK[t] + 0.8 * D))).astype(int)
    s = rng.normal(size=len(D)) + theta * y
    return pd.DataFrame({"g": np.repeat(np.arange(n_groups), m), "task": np.array(list("ABC"))[t], "D": D, "y": y, "s": s})


def calibrate_alt(pi, a0, m, delta, seed=7):
    rng = np.random.default_rng(seed)
    G = P.POP_N // m
    t, D = np.repeat(rng.integers(0, 3, G), m), np.repeat(rng.normal(size=G), m)
    lin = 0.6 * TASK[t] + 0.8 * D
    u = rng.random(len(D))

    def a_for(b):
        return P._bisect(lambda a: float(_sig(a + b * lin).mean()), -12, 12, pi)

    b = P._bisect(lambda b: auc((u < _sig(a_for(b) + b * lin)).astype(int), lin), 0.0, 15.0, a0, iters=30)
    a = a_for(b)

    def inc(theta):
        df = make_alt(G, m, a, b, theta, np.random.default_rng(seed + 1))
        num, li = split_design(df, ["task", "D"])
        y, g, s = df.y.to_numpy(), df.g.astype(str).to_numpy(), _rank01(df.s.to_numpy())
        return auc(y, cv_predict("linear", num, li, s, y, g, 0)) - auc(y, cv_predict("linear", num, li, None, y, g, 0))

    theta = P._bisect(inc, 0.0, 5.0, delta, iters=18)
    return {"a": a, "b": b, "theta": theta}


def one_alt(n_groups, m, c, seed, null):
    df = make_alt(n_groups, m, c["a"], c["b"], c["theta"], np.random.default_rng(seed))
    if df.y.nunique() < 2:
        return False
    r = audit(df, "s", "y", "g", ["task", "D"], seeds=range(2), n_perm=200, perm_stop=10, n_boot=0, n_boot_desc=10, null=null)
    return bool(r["adds_information"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--null", default="residual")
    ap.add_argument("--seed-salt", type=int, default=13)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated scenario names (S1,S2,S3)")
    a = ap.parse_args()
    from joblib import Parallel, delayed
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"null": a.null, "seed_salt": a.seed_salt, "scenarios": {}}
    for name, (pi, a0, m, delta) in SCEN.items():
        if a.only and name not in a.only.split(","):
            continue
        if name in out["scenarios"]:
            continue
        t0 = time.time()
        c = P.calibrate(pi, a0, m, delta)
        pl = P.plan(pi, a0, m, delta, reps=200, n_jobs=a.jobs, null=a.null, seed=1, calib=c)
        rec = pl["recommended"]["groups"] if pl["recommended"] else P.default_grid(pi, m)[-1]
        Ns = sorted({max(2, rec // 2), rec, rec * 2})
        pred = P.plan(pi, a0, m, delta, grid=Ns, reps=200, n_jobs=a.jobs, null=a.null, seed=2, calib=c)["grid"]
        fresh = P.plan(pi, a0, m, delta, grid=Ns, reps=400, n_jobs=a.jobs, null=a.null, seed=1000 + a.seed_salt, calib=c)["grid"]
        ca = calibrate_alt(pi, a0, m, delta)
        alt = []
        for N in Ns:
            flags = Parallel(n_jobs=a.jobs)(delayed(one_alt)(N, m, ca, 5_000_000 + 7919 * a.seed_salt + N * 1009 + r, a.null)
                                            for r in range(400))
            k = int(sum(flags))
            alt.append({"groups": N, "power": k / 400, "wilson95": P._wilson(k, 400)})
        cells = []
        for pr, fr, al in zip(pred, fresh, alt):
            cells.append({"groups": pr["groups"], "predicted": pr["power"], "fresh": fr["power"], "fresh_wilson95": fr["wilson95"],
                          "independent": al["power"], "P1_inside": fr["wilson95"][0] <= pr["power"] <= fr["wilson95"][1],
                          "P2_close": abs(pr["power"] - al["power"]) <= 0.10})
        out["scenarios"][name] = {"inputs": [pi, a0, m, delta], "plan": pl, "recommended_groups": rec, "cells": cells,
                                  "calibration_alt": ca, "independent_power_at_recommended": alt[Ns.index(rec)]["power"],
                                  "sec": round(time.time() - t0)}
        print(name, json.dumps(cells), flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
    if a.only:
        return
    cells = [c for s in out["scenarios"].values() for c in s["cells"]]
    out["verdict"] = {"P1_inside": sum(c["P1_inside"] for c in cells), "P2_close": sum(c["P2_close"] for c in cells),
                      "n_cells": len(cells),
                      "P3_min_independent_at_recommended": min(s["independent_power_at_recommended"] for s in out["scenarios"].values())}
    v = out["verdict"]
    v["pass"] = bool(v["P1_inside"] >= 8 and v["P2_close"] >= 8 and v["P3_min_independent_at_recommended"] >= 0.70)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
