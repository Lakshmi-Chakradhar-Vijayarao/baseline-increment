"""How good is the detector's raw score as a probability? (pre-registered in docs/PREREG_0.11.md).

Descriptive only, no verdict. Binary labels; the raw score (never its rank) must lie in [0, 1].
- calibration slope b from y ~ a + b * logit(p)            (1 ideal; < 1 overconfident)
- calibration-in-the-large c from y ~ offset(logit(p)) + c   (0 ideal; > 0 the score under-predicts)
- Brier score, mean predicted vs observed rate, a 10-bin reliability table, ECE (biased upward at finite n)
Intervals resample groups (2,000 draws). Scores are clipped to [1e-6, 1 - 1e-6] before logits.
"""
import numpy as np

from .core import check_binary_label

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _logistic_fit(x, y, offset=None, slope=True, iters=50):
    """Newton-Raphson for y ~ a + b*x (+ offset); returns (a, b). b is fixed at 0 if slope=False."""
    off = np.zeros(len(y)) if offset is None else offset
    X = np.column_stack([np.ones(len(y)), x]) if slope else np.ones((len(y), 1))
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta + off, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-12
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return (float(beta[0]), float(beta[1])) if slope else (float(beta[0]), 1.0)


def _stats(p, y):
    lp = _logit(p)
    _, slope = _logistic_fit(lp, y)
    citl, _ = _logistic_fit(None, y, offset=lp, slope=False)
    return {"slope": slope, "citl": citl, "brier": float(np.mean((p - y) ** 2))}


def reliability(p, y, bins=10):
    order = np.argsort(p, kind="stable")
    rows, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        mp, mo = float(p[chunk].mean()), float(y[chunk].mean())
        rows.append({"n": int(len(chunk)), "mean_predicted": mp, "observed_rate": mo})
        ece += len(chunk) / len(p) * abs(mp - mo)
    return rows, float(ece)


def score_calibration(df, score, label, group=None, n_boot=2000, seed=0, show_intervals=True):
    check_binary_label(df[label], label)
    p = np.asarray(df[score], float)
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError(f"score {score!r} must be a probability in [0, 1] for --calibration "
                         f"(range found {np.nanmin(p):.3g} to {np.nanmax(p):.3g})")
    y = np.asarray(df[label], float)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    est = _stats(p, y)
    table, ece = reliability(p, y)
    out = {"n_items": int(len(y)), "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()), **est,
           "ece": ece, "ece_note": "ECE is biased upward at finite sample sizes; descriptive only",
           "reliability": table, "clipped_to": [EPS, 1 - EPS]}
    if show_intervals and n_boot:
        rng = np.random.default_rng(seed)
        g = np.unique(groups)
        idx = {k: np.where(groups == k)[0] for k in g}
        draws = []
        while len(draws) < n_boot:
            ii = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
            if 0 < y[ii].sum() < len(ii):
                draws.append(_stats(p[ii], y[ii]))
        for k in ("slope", "citl", "brier"):
            v = [d[k] for d in draws]
            out[f"{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def fmt_calibration(c):
    ci = lambda k: f" [{c[k + '_ci'][0]:.3f}, {c[k + '_ci'][1]:.3f}]" if k + "_ci" in c else ""
    lines = ["raw-score calibration (descriptive)",
             f"  calibration slope           {c['slope']:.3f}{ci('slope')}   (1 = ideal; < 1 overconfident)",
             f"  calibration-in-the-large    {c['citl']:+.3f}{ci('citl')}   (0 = ideal; > 0 under-predicts)",
             f"  Brier score                 {c['brier']:.4f}{ci('brier')}",
             f"  mean predicted {c['mean_predicted']:.3f}  observed rate {c['observed_rate']:.3f}  ECE {c['ece']:.3f} (biased upward)"]
    return "\n".join(lines)
