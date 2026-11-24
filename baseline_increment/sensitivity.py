"""Omitted-baseline sensitivity analysis (docs/prereg/PREREG_0.13.md, Part 1 as amended by addendum 1).

How strong would a cheap variable missing from the baselines have to be to make the detector redundant?

Out of fold, under the audit's grouped folds (seed 0, headline family):
  r_S = rank(S) - S_hat(B)     the detector's residual given the baselines (the null's score model)
  r_Y = Y - p_hat(B)           the label's residual given the baselines
  r   = corr(r_S, r_Y)         how much the detector tracks the label beyond the baselines

An omitted variable U with multiple correlations alpha (with r_S) and beta (with r_Y) leaves a partial covariance of
r - alpha * beta between detector and label in the linear-Gaussian approximation (the omitted-variable logic of
Cinelli & Hazlett 2020, applied here to the residuals that drive the increment). So U makes the detector redundant
when alpha * beta reaches r:

  erasure index  phi(U) = alpha * beta / r        (phi >= 1: would erase the increment)
  robustness value  RV = sqrt(r)                   (the equal-strength point alpha = beta)

Each declared baseline is benchmarked by its own phi, computed as if it had been left out. This is a sensitivity
analysis in a linear approximation, not a bound; its validation against simulated omitted baselines is in
CALIBRATION.md.
"""
import numpy as np

from .core import cv_predict, estimate, resolve_null, score_given_baselines, split_design, transform_score


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0.0


def multiple_corr(target, R):
    """|corr| for one column; the multiple correlation (square root of the OLS R^2) for several."""
    t = _z(target)
    R = np.asarray(R, float).reshape(len(t), -1)
    R = R[:, R.std(0) > 0]
    if R.shape[1] == 0:
        return 0.0
    X = np.column_stack([np.ones(len(t)), R])
    coef, *_ = np.linalg.lstsq(X, t, rcond=None)
    return float(np.sqrt(max(0.0, 1 - np.mean((t - X @ coef) ** 2) / max(np.var(t), 1e-12))))


def residuals(fam, num, lin, s, y, groups, model):
    r_s = s - score_given_baselines(num, lin, s, groups, 0, model)
    r_y = y - cv_predict(fam, num, lin, None, y, groups, 0)
    return r_s, r_y


def strengths(r_s, r_y, num_rest, lin_rest, cols, groups, model):
    """(alpha, beta) of a variable (columns ``cols``) residualised on the other baselines."""
    res = np.column_stack([cols[:, k] - score_given_baselines(num_rest, lin_rest, cols[:, k], groups, 0, model)
                           for k in range(cols.shape[1])])
    return multiple_corr(r_s, res), multiple_corr(r_y, res)


def strengths_flexible(r_s, r_y, num_all, lin_all, groups, model):
    """Nonparametric strengths of a variable: alpha^2 (beta^2) is the out-of-fold R^2 of a flexible regression of the
    detector's (label's) residual on all baselines including that variable. The residuals are already orthogonal to
    the other baselines out of fold, so the R^2 measures what the variable adds, interactions included
    (cf. the nonparametric partial R^2 of Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022)."""
    def r2(t):
        t = np.asarray(t, float)
        pred = score_given_baselines(num_all, lin_all, t, groups, 0, model)
        return max(0.0, 1 - np.mean((t - pred) ** 2) / max(np.var(t), 1e-12))
    return float(np.sqrt(r2(r_s))), float(np.sqrt(r2(r_y)))


def sensitivity(df, score, label, group, baselines, families=("linear", "spline", "gbm"), seeds=range(2),
                null="residual", score_transform="rank", rank_within=None, headline=None):
    baselines = list(baselines)
    y = np.asarray(df[label], int)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, baselines)
    if headline is None:
        _, headline = estimate(num, lin, s, y, groups, tuple(families), list(seeds))
    model = "super_learner" if resolve_null(null) == "residual_sl" else "gbm"
    r_s, r_y = residuals(headline, num, lin, s, y, groups, model)
    r = float(np.mean(_z(r_s) * _z(r_y)))
    out = {"headline_family": headline, "residual_correlation": r, "benchmarks": {},
           "label": "sensitivity analysis, linear approximation (validated against simulated omitted baselines; "
                    "no worst-case guarantee)"}
    if r <= 0:
        out.update({"robustness_value": None, "note": "the detector's residual does not track the label's; "
                                                      "nothing to erase"})
        return out
    out["robustness_value"] = float(np.sqrt(r))
    for c in baselines:
        rest = [x for x in baselines if x != c]
        n2, l2 = split_design(df, rest)
        rs_j, ry_j = residuals(headline, n2, l2, s, y, groups, model)
        r_j = float(np.mean(_z(rs_j) * _z(ry_j)))
        nj, lj = split_design(df, [c])
        a, b = strengths(rs_j, ry_j, n2, l2, np.hstack([nj, lj]), groups, model)
        out["benchmarks"][c] = {"alpha": a, "beta": b, "phi": a * b / r_j if r_j > 0 else float("inf"),
                                "r_without_it": r_j}
    if out["benchmarks"]:
        out["strongest_benchmark"] = max(out["benchmarks"], key=lambda k: out["benchmarks"][k]["phi"])
    return out


def sentence(sens):
    if sens.get("robustness_value") is None:
        return "Omitted-baseline sensitivity: " + sens.get("note", "not computed") + "."
    r, rv = sens["residual_correlation"], sens["robustness_value"]
    txt = (f"In the linear approximation, an omitted baseline makes the detector redundant when the product of its "
           f"correlations with the detector's and the label's residuals reaches {r:.2f} (both at least {rv:.2f} if "
           "equal)")
    b = sens.get("strongest_benchmark")
    if b:
        v = sens["benchmarks"][b]
        txt += (f"; the strongest declared baseline, {b}, reaches {v['phi']:.2f} of that "
                f"(correlations {v['alpha']:.2f} and {v['beta']:.2f})")
    return txt + "."
