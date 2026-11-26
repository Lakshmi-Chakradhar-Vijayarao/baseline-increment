"""Sample-size planner (docs/prereg/PREREG_0.13.md, Part 2).

How many items (groups) must be labelled for the audit to detect a *population* increment delta with the target
power? Power is estimated by simulating the actual --fast audit, not from a formula.

Model (fixed): each group shares a latent L ~ N(0, 1); Y ~ Bernoulli(sigmoid(a + b L)); the observed baseline is L;
the detector is S = noise + theta * Y. a and b are solved so that prevalence = pi and the population AUROC of L is A0;
theta is solved so that the population increment (200,000 items, linear family, one seed) equals delta.
"""
import math

import numpy as np
import pandas as pd

from .core import auc, cv_predict, split_design

POP_N = 200_000


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _bisect(f, lo, hi, target, iters=40):
    flo = f(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm < target) == (flo < target):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def _population(pi, a0, m, seed=0):
    rng = np.random.default_rng(seed)
    G = POP_N // m
    L = np.repeat(rng.normal(size=G), m)
    u = rng.random(len(L))

    def prev_for(a, b):
        return float(_sig(a + b * L).mean())

    def a_for(b):
        return _bisect(lambda a: prev_for(a, b), -12, 12, pi)

    def auc_for(b):
        a = a_for(b)
        return auc((u < _sig(a + b * L)).astype(int), L)

    if not 0.5 < a0 < 0.995:
        raise ValueError("baselines-only AUROC must be in (0.5, 0.995)")
    b = _bisect(auc_for, 0.0, 15.0, a0, iters=30)
    a = a_for(b)
    return a, b


def _make(n_groups, m, a, b, theta, rng):
    L = np.repeat(rng.normal(size=n_groups), m)
    y = (rng.random(len(L)) < _sig(a + b * L)).astype(int)
    s = rng.normal(size=len(L)) + theta * y
    return pd.DataFrame({"g": np.repeat(np.arange(n_groups), m), "L": L, "y": y, "s": s})


def _pop_increment(m, a, b, theta, seed=1):
    df = _make(POP_N // m, m, a, b, theta, np.random.default_rng(seed))
    num, lin = split_design(df, ["L"])
    y = df.y.to_numpy()
    g = df.g.astype(str).to_numpy()
    from .core import _rank01
    s = _rank01(df.s.to_numpy())
    return auc(y, cv_predict("linear", num, lin, s, y, g, 0)) - auc(y, cv_predict("linear", num, lin, None, y, g, 0))


def calibrate(pi, a0, m, delta):
    a, b = _population(pi, a0, m)
    theta = _bisect(lambda t: _pop_increment(m, a, b, t), 0.0, 5.0, delta, iters=18)
    return {"a": a, "b": b, "theta": theta}


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


CRITERIA = ("pass", "adds_information")


def detected(r, criterion):
    """'pass' (default from 0.14): the primary verdict, lower 95% bound of the gain > 0. 'adds_information': the
    experimental permutation test, kept only to reproduce the 0.13 planner runs."""
    return r["incremental_prediction"]["verdict"] == "PASS" if criterion == "pass" else bool(r["adds_information"])


def _one(n_groups, m, a, b, theta, seed, null, criterion="pass"):
    from .core import audit
    df = _make(n_groups, m, a, b, theta, np.random.default_rng(seed))
    if df.y.nunique() < 2:
        return False
    perms = dict(n_perm=1, perm_stop=None) if criterion == "pass" else dict(n_perm=200, perm_stop=10)
    r = audit(df, "s", "y", "g", ["L"], seeds=range(2), n_boot=0, n_boot_desc=10, null=null, interval="delong", **perms)
    return detected(r, criterion)


def default_grid(pi, m, max_items=20_000, k=8):
    lo = max(2, math.ceil(20 / max(pi, 1e-6) / m), math.ceil(20 / max(1 - pi, 1e-6) / m))
    hi = max(lo + 1, max_items // m)
    return sorted({int(round(lo * (hi / lo) ** (i / (k - 1)))) for i in range(k)})


def plan(pi, a0, m, delta, power=0.80, grid=None, reps=200, n_jobs=-1, null="residual", seed=0, calib=None,
         stop_at_target=None, criterion="pass"):
    """Power of the audit's primary verdict (PASS: lower 95% bound of the gain > 0) at each group count in ``grid``;
    returns the smallest N reaching ``power``. ``criterion="adds_information"`` reproduces the 0.13 planner.
    With the default grid the search stops at the first size reaching the target (larger sizes cannot change the
    recommendation); an explicit grid is evaluated in full unless stop_at_target=True."""
    if stop_at_target is None:
        stop_at_target = grid is None
    from joblib import Parallel, delayed
    c = calib or calibrate(pi, a0, m, delta)
    grid = grid or default_grid(pi, m)
    rows = []
    for N in grid:
        seeds = [seed * 1_000_003 + N * 1009 + r for r in range(reps)]
        flags = Parallel(n_jobs=n_jobs)(delayed(_one)(N, m, c["a"], c["b"], c["theta"], sd, null, criterion) for sd in seeds)
        k = int(sum(flags))
        rows.append({"groups": N, "items": N * m, "expected_positives": round(N * m * pi, 1), "power": k / reps,
                     "wilson95": _wilson(k, reps)})
        if stop_at_target and k / reps >= power:
            break
    rec = next((r for r in rows if r["power"] >= power), None)
    return {"inputs": {"prevalence": pi, "baselines_auroc": a0, "group_size": m, "target_population_increment": delta,
                       "target_power": power, "reps_per_size": reps, "criterion": criterion}, "calibration": c, "grid": rows,
            "recommended": rec,
            "note": ("power is for a population increment of the stated size; the out-of-fold estimate at small N is "
                     "smaller because of fitting. Simulated under a one-baseline model matched to your prevalence and "
                     "baselines-only AUROC (docs/prereg/PREREG_0.13.md)")}


def from_pilot(df, label, group, baselines, families=("linear", "spline", "gbm")):
    """Prevalence, baselines-only AUROC (headline family, out of fold) and mean group size from pilot data."""
    from .core import estimate
    y = np.asarray(df[label], int)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, list(baselines))
    est, head = estimate(num, lin, np.zeros(len(y)), y, groups, tuple(families), range(2))
    m = len(y) / len(np.unique(groups))
    return {"prevalence": float(y.mean()), "baselines_auroc": est[head]["baseline_auc"], "group_size": max(1, int(round(m)))}


def fmt_plan(p):
    i = p["inputs"]
    lines = [f"sample-size plan: detect a population increment of {i['target_population_increment']:+.3f} with power "
             f"{i['target_power']:.2f}",
             f"prevalence {i['prevalence']:.3f}  baselines-only AUROC {i['baselines_auroc']:.3f}  group size {i['group_size']}"
             f"  ({i['reps_per_size']} simulated audits per size)"]
    for r in p["grid"]:
        lines.append(f"  {r['groups']:>7} groups ({r['items']:>7} items, ~{r['expected_positives']:.0f} positives)  power "
                     f"{r['power']:.2f} [{r['wilson95'][0]:.2f}, {r['wilson95'][1]:.2f}]")
    rec = p["recommended"]
    lines.append(f"recommended: {rec['groups']} groups ({rec['items']} items)" if rec else
                 "no size in the grid reached the target power; widen the grid (--max-items)")
    lines.append("note: " + p["note"])
    return "\n".join(lines)
