"""Development prototype of the 0.12 null score model (E[S | baselines]): a discrete super learner over
three label-free regressors, chosen by grouped out-of-fold MSE of S. Not used by any acceptance run until
it is copied into the engine under docs/prereg/PREREG_0.12.md."""
import warnings as _w
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler

import baseline_increment.core as core

MAX_INTERACTION_FEATURES = 3000


def interaction_basis(num, lin, tr):
    """Spline basis of numeric columns, one-hot columns, and degree-2 products: numeric x numeric and
    numeric x one-hot (standardised on the training rows)."""
    parts = [lin]
    if num.shape[1]:
        st = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant").fit(num[tr])
        sp = st.transform(num)
        parts.append(sp)
        if lin.shape[1] and sp.shape[1] * lin.shape[1] <= MAX_INTERACTION_FEATURES:
            parts.append(np.column_stack([sp[:, i] * lin[:, j] for i in range(sp.shape[1]) for j in range(lin.shape[1])]))
        mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-12
        z = (num - mu) / sd
        k = z.shape[1]
        prods = [z[:, i] * z[:, j] for i in range(k) for j in range(i, k)]
        prods += [z[:, i] * lin[:, j] for i in range(k) for j in range(lin.shape[1])]
        if prods and len(prods) <= MAX_INTERACTION_FEATURES:
            parts.append(np.column_stack(prods))
        parts.append(z)
    return np.hstack(parts)


def _fit(kind, num, lin, s, tr, te, seed):
    X = np.hstack([num, lin])
    if kind == "gbm":
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)
        return m.fit(X[tr], s[tr]).predict(X[te])
    if kind == "gbm_deep":
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=10,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)
        return m.fit(X[tr], s[tr]).predict(X[te])
    B = interaction_basis(num, lin, tr)
    sc = StandardScaler().fit(B[tr])
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(B[tr]), s[tr])
    return m.predict(sc.transform(B[te]))


CANDIDATES = ("gbm", "ridge_interactions")


def score_given_baselines_sl(num, lin, s, groups, seed=0, info=None):
    if num.shape[1] + lin.shape[1] == 0:
        return np.full(len(s), s.mean())
    preds = {}
    for kind in CANDIDATES:
        p = np.zeros(len(s))
        for te in core.group_folds(groups, seed):
            p[te] = _fit(kind, num, lin, s, ~te, te, seed)
        preds[kind] = p
    mse = {k: float(np.mean((s - v) ** 2)) for k, v in preds.items()}
    best = min(CANDIDATES, key=lambda k: mse[k])
    if info is not None:
        info.update({"chosen": best, "mse": mse})
    return preds[best]
