"""Omitted-baseline sensitivity analysis (docs/prereg/PREREG_0.13.md, Part 1).

How strong would a cheap variable missing from the baselines have to be to erase the detector's increment?

An adaptation of Cinelli & Hazlett's (2020) robustness value to the out-of-fold AUROC increment. A hypothetical
omitted baseline U is a unit-variance mix of the detector's residual (given the baselines) and the label's residual
(given the baselines), with prescribed correlations alpha and beta with them. U is added to the baselines and the
increment is re-estimated. The robustness value (RV) is the smallest alpha = beta that brings it to zero. Declared
baselines are benchmarked by their own (alpha, beta) when left out.

This is a sensitivity analysis, not a bound: no worst-case guarantee is proved for AUROC with flexible models. Its
validation (no false comfort against simulated omitted baselines) is in CALIBRATION.md.
"""
import numpy as np

from .core import (auc, cv_predict, estimate, resolve_null, score_given_baselines, split_design, transform_score)

GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0.0


def _mix(zs, zy, a_t, b_t, rng):
    """U with corr(U, zs) = a_t and corr(U, zy) = b_t (population sense), or None if infeasible."""
    r = float(np.mean(zs * zy))
    M = np.array([[1.0, r], [r, 1.0]])
    try:
        w = np.linalg.solve(M, np.array([a_t, b_t]))
    except np.linalg.LinAlgError:
        return None
    q = w[0] ** 2 + w[1] ** 2 + 2 * w[0] * w[1] * r
    if q > 1 + 1e-9:
        return None
    return w[0] * zs + w[1] * zy + np.sqrt(max(0.0, 1 - q)) * rng.standard_normal(len(zs))


def _increment_with(fam, num, lin, s, y, groups, seeds, extra):
    num2 = np.hstack([num, extra[:, None]])
    b = [auc(y, cv_predict(fam, num2, lin, None, y, groups, sd)) for sd in seeds]
    f = [auc(y, cv_predict(fam, num2, lin, s, y, groups, sd)) for sd in seeds]
    return float(np.mean(f) - np.mean(b))


def _multiple_corr(target, R):
    """|corr| for one column; the multiple correlation (sqrt of R^2 of an OLS fit) for several."""
    t = _z(target)
    R = np.asarray(R, float).reshape(len(t), -1)
    R = R[:, R.std(0) > 0]
    if R.shape[1] == 0:
        return 0.0
    X = np.column_stack([np.ones(len(t)), R])
    beta, *_ = np.linalg.lstsq(X, t, rcond=None)
    fit = X @ beta
    return float(np.sqrt(max(0.0, 1 - np.mean((t - fit) ** 2) / max(np.var(t), 1e-12))))


def _residuals(fam, num, lin, s, y, groups, model):
    r_s = s - score_given_baselines(num, lin, s, groups, 0, model)
    r_y = y - cv_predict(fam, num, lin, None, y, groups, 0)
    return r_s, r_y


def sensitivity(df, score, label, group, baselines, families=("linear", "spline", "gbm"), seeds=range(2),
                null="residual", score_transform="rank", rank_within=None, draws=3, grid=GRID, seed=0, headline=None):
    baselines = list(baselines)
    y = np.asarray(df[label], int)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, baselines)
    seeds = list(seeds)
    if headline is None:
        _, headline = estimate(num, lin, s, y, groups, tuple(families), seeds)
    null = resolve_null(null)
    model = "super_learner" if null == "residual_sl" else "gbm"
    r_s, r_y = _residuals(headline, num, lin, s, y, groups, model)
    zs, zy = _z(r_s), _z(r_y)
    rng = np.random.default_rng(seed)
    eps_seeds = [int(x) for x in rng.integers(0, 2**31 - 1, draws)]

    def remaining(a_t, b_t):
        vals = []
        for es in eps_seeds:
            u = _mix(zs, zy, a_t, b_t, np.random.default_rng(es))
            if u is None:
                return None
            vals.append(_increment_with(headline, num, lin, s, y, groups, seeds, u))
        return float(np.mean(vals))

    # robustness value along alpha = beta
    r = float(np.mean(zs * zy))
    t_max = min(0.99, float(np.sqrt((1 + r) / 2)) - 1e-6)
    top = remaining(t_max, t_max)
    if top is not None and top > 0:
        rv, rv_note = None, f"increment still above zero at the largest feasible strength ({t_max:.2f})"
    else:
        lo, hi = 0.0, t_max
        while hi - lo > 0.01:
            mid = (lo + hi) / 2
            v = remaining(mid, mid)
            if v is not None and v > 0:
                lo = mid
            else:
                hi = mid
        rv, rv_note = round(hi, 3), None
    contour = {f"{a:.1f},{b:.1f}": remaining(a, b) for a in grid for b in grid}

    bench = {}
    for j, c in enumerate(baselines):
        rest = [x for x in baselines if x != c]
        n2, l2 = split_design(df, rest)
        rs_j, ry_j = _residuals(headline, n2, l2, s, y, groups, model)
        nj, lj = split_design(df, [c])
        cols = np.hstack([nj, lj])
        res = np.column_stack([cols[:, k] - score_given_baselines(n2, l2, cols[:, k], groups, 0, model)
                               for k in range(cols.shape[1])])
        bench[c] = {"alpha": _multiple_corr(rs_j, res), "beta": _multiple_corr(ry_j, res)}
    strongest = max(bench, key=lambda k: min(bench[k]["alpha"], bench[k]["beta"])) if bench else None
    return {"headline_family": headline, "residual_correlation": r, "robustness_value": rv, "robustness_note": rv_note,
            "largest_feasible_strength": t_max, "contour": contour, "benchmarks": bench, "strongest_benchmark": strongest,
            "draws": draws, "label": "sensitivity analysis (validated against simulated omitted baselines; "
                                     "no worst-case guarantee)"}


def sentence(sens):
    rv = sens["robustness_value"]
    b = sens["strongest_benchmark"]
    head = (f"The increment would fall to zero only if an omitted baseline were correlated at least {rv:.2f} with both "
            "the detector's and the label's residuals" if rv is not None else
            f"No omitted baseline up to correlation {sens['largest_feasible_strength']:.2f} with both residuals brings "
            "the increment to zero")
    if b:
        v = sens["benchmarks"][b]
        head += f"; the strongest declared baseline, {b}, has ({v['alpha']:.2f}, {v['beta']:.2f})."
    else:
        head += "."
    return head
