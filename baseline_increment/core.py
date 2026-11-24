"""Does a detector's score add anything beyond cheap baselines?

Version 0.2. Changes from 0.1, prompted by the 8 Nov 2026 calibration simulation:
- A linear-only baseline credited a detector that was only a nonlinear function of the
  metadata (+0.088, 100/100 rejections).
- The CI came from one CV seed while the point estimate averaged ten.

Design:
* Baseline families are fixed in advance:
  - ``linear``: standardised L2 logistic regression;
  - ``spline``: cubic B-splines on numeric columns plus one-hot categories, then logistic;
  - ``gbm``: shallow histogram gradient boosting, which can capture interactions.

  With no baseline columns every family reduces to the intercept (B0). Each family is fit twice,
  once on the baselines alone and once on the baselines plus the detector.
* The headline family is the one with the highest out-of-fold AUROC **for the baselines alone**.
  It is chosen without looking at the detector's increment. Every family's increment is also reported.
* The estimator is Delta = mean over CV seeds of [AUROC(baselines + S) - AUROC(baselines)], each
  out of fold under grouped K-fold CV (no group straddles train and test).
* The CI comes from a group bootstrap that reruns the whole estimator, including the headline choice.
  Resampled copies of a group keep its id, so they stay in one fold.
* The p-value comes from a refit permutation null. Group-level detector scores are permuted, optionally
  only within strata (``perm_strata``, e.g. cell x difficulty decile), so a permutation never crosses a
  stratum. Test labels are never shuffled alone. p = (1 + #{null >= observed}) / (1 + B), with the
  observed and null statistics computed with the same single CV seed.
* v0.2.1 (found by the v0.2.0 simulation battery). By default the permutation is also confined to
  10 bands of the headline baseline's own out-of-fold risk (``risk_bins``). Unconfined, a permuted
  score is pure noise and slightly *hurts* the fit, while a detector that is only a function of the
  baselines costs nothing. Such a detector then beat the null in 17% of runs (interaction scenario)
  despite a negative increment. A detector "adds information" only if p < 0.05 **and** the increment > 0.
* v0.2.2 (found by the v0.2.1 battery: the interaction null was flagged in 27% of runs, with
  p < 0.05 in 66%). A detector that is an *exact* function of the metadata beats any baseline that
  only approximates that function at finite n, so it out-scores every permuted copy. The null is
  now a conditional randomisation:
  - fit S ~ baselines out of fold (gradient-boosting regressor, grouped CV): S_hat;
  - null scores S* = S_hat + group-permuted residuals (within ``perm_strata`` if given).

  Every null detector then carries the same metadata function as S, and only S's residual, the part
  the metadata cannot explain, is tested. Risk-band confinement is off by default (``risk_bins=0``).
* By default the detector enters as its rank, optionally ranked within a column such as model
  (``rank_within``). Affine or monotone rescaling of the score therefore cannot change the result.
"""
import warnings as _w

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

VERSION = "0.11.1"
N_FOLDS = 5
FAMILIES = ("linear", "spline", "gbm")
_SPLINE_MIN_UNIQUE = 6


def check_binary_label(col, name="label"):
    """The binary engine needs 0/1 labels. Before 0.11 a graded label (e.g. severity 0-3) was silently mishandled."""
    v = np.asarray(col)
    try:
        vals = set(np.unique(v[~np.isnan(v.astype(float))]).astype(float))
    except (TypeError, ValueError):
        raise ValueError(f"label {name!r} must be numeric 0/1")
    if np.isnan(v.astype(float)).any():
        raise ValueError(f"label {name!r} has missing values")
    if not vals <= {0.0, 1.0}:
        shown = sorted(vals)[:8]
        raise ValueError(f"label {name!r} must be 0/1 for this command; found values {shown}. "
                         "For graded labels (e.g. severity 0-3) use --outcome ordinal")


def auc(y, s):
    """AUROC with ties averaged (Mann-Whitney)."""
    y = np.asarray(y, bool)
    r = rankdata(np.asarray(s, float))
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _rank01(v):
    v = np.asarray(v, float)
    return np.full(len(v), 0.5) if len(v) < 2 else (rankdata(v) - 1) / (len(v) - 1)


def transform_score(df, score, how="rank", within=None):
    s = np.asarray(df[score], float)
    if how == "raw":
        return s
    if within is None:
        return _rank01(s)
    out = np.empty(len(s))
    w = np.asarray(df[within]).astype(str)
    for k in np.unique(w):
        m = w == k
        out[m] = _rank01(s[m])
    return out


def split_design(df, cols):
    """Numeric columns with >= 6 distinct values -> 'num' (eligible for splines); categories
    (one-hot, all levels) and low-cardinality numerics -> 'lin'."""
    num, lin = [], []
    for c in cols:
        v = np.asarray(df[c])
        if np.issubdtype(v.dtype, np.number):
            v = v.astype(float)
            (num if len(np.unique(v)) >= _SPLINE_MIN_UNIQUE else lin).append(v[:, None])
        else:
            lv = sorted(set(map(str, v)))
            lin.append(np.array([[str(x) == l for l in lv] for x in v], float))
    n = len(df)
    return (np.hstack(num) if num else np.zeros((n, 0)), np.hstack(lin) if lin else np.zeros((n, 0)))


def design(df, cols):
    """0.1-compatible helper: all baseline columns as one numeric matrix."""
    a, b = split_design(df, cols)
    return np.hstack([a, b])


def group_folds(groups, seed):
    g = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(g)
    return [np.isin(groups, perm[f::N_FOLDS]) for f in range(N_FOLDS)]


def _features(fam, num, lin, s, tr):
    if fam == "spline" and num.shape[1]:
        st = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant").fit(num[tr])
        base = np.hstack([st.transform(num), lin])
    else:
        base = np.hstack([num, lin])
    return base if s is None else np.hstack([base, s[:, None]])


def _fit_predict(fam, X, y, tr, te, seed):
    if X.shape[1] == 0 or y[tr].min() == y[tr].max():
        return np.full(int(te.sum()), y[tr].mean())
    if fam == "gbm":
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                           l2_regularization=1.0, early_stopping=False, random_state=seed)
        return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    sc = StandardScaler().fit(X[tr])
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        m = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return m.predict_proba(sc.transform(X[te]))[:, 1]


def cv_predict(fam, num, lin, s, y, groups, seed):
    """Out-of-fold probabilities for one family; s=None means baselines only."""
    p = np.zeros(len(y))
    for te in group_folds(groups, seed):
        tr = ~te
        p[te] = _fit_predict(fam, _features(fam, num, lin, s, tr), y, tr, te, seed)
    return p


def tpr_at_fpr(y, p, fpr):
    """True-positive rate at the threshold where the false-positive rate first reaches `fpr`."""
    y = np.asarray(y, int)
    neg = np.sort(np.asarray(p, float)[y == 0])[::-1]
    if neg.size == 0 or (y == 1).sum() == 0:
        return float("nan")
    k = int(np.floor(fpr * neg.size))
    thr = neg[k] if k < neg.size else -np.inf
    return float((np.asarray(p, float)[y == 1] > thr).mean())


def operating_point(df, score, label, group, baselines, family="linear", fpr=0.10, seeds=range(5),
                    score_transform="rank", rank_within=None):
    """What the increment means in use: TPR at a fixed FPR for the baselines alone and for baselines +
    detector (out-of-fold, same folds and family as the audit), averaged over seeds. Reporting only;
    the test and its calibration are unchanged."""
    y = np.asarray(df[label], int)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, list(baselines))
    b = [tpr_at_fpr(y, cv_predict(family, num, lin, None, y, groups, sd), fpr) for sd in seeds]
    f = [tpr_at_fpr(y, cv_predict(family, num, lin, s, y, groups, sd), fpr) for sd in seeds]
    d = [tpr_at_fpr(y, np.asarray(df[score], float), fpr)]
    return {"family": family, "fpr": fpr, "tpr_baselines": float(np.mean(b)), "tpr_plus_detector": float(np.mean(f)),
            "tpr_gain": float(np.mean(f) - np.mean(b)), "tpr_detector_alone": float(d[0])}


def estimate(num, lin, s, y, groups, families=FAMILIES, seeds=range(5)):
    """The estimator: per family, mean over seeds of baseline-only and baseline+detector OOF AUROC.
    Headline = the family with the highest baseline-only AUROC."""
    out = {}
    for fam in families:
        b = [auc(y, cv_predict(fam, num, lin, None, y, groups, sd)) for sd in seeds]
        f = [auc(y, cv_predict(fam, num, lin, s, y, groups, sd)) for sd in seeds]
        out[fam] = {"baseline_auc": float(np.mean(b)), "plus_detector_auc": float(np.mean(f)),
                    "increment": float(np.mean(f) - np.mean(b))}
    head = max(families, key=lambda k: out[k]["baseline_auc"])
    return out, head


def _boot_once(num, lin, s, y, groups, families, seeds, rng_seed):
    rng = np.random.default_rng(rng_seed)
    g = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    while True:
        ii = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        if 0 < y[ii].sum() < len(ii):
            break
    fam, head = estimate(num[ii], lin[ii], s[ii], y[ii], groups[ii], families, seeds)
    return fam[head]["increment"], {k: v["increment"] for k, v in fam.items()}


def _half_once(num, lin, s, y, groups, families, seeds, rng_seed):
    """Half-sample refit: half the groups, drawn without replacement, then the whole estimator."""
    rng = np.random.default_rng(rng_seed)
    g = np.unique(groups)
    while True:
        keep = np.isin(groups, rng.choice(g, len(g) // 2, replace=False))
        if 0 < y[keep].sum() < keep.sum():
            break
    fam, head = estimate(num[keep], lin[keep], s[keep], y[keep], groups[keep], families, seeds)
    return fam[head]["increment"], {k: v["increment"] for k, v in fam.items()}


def _perm_sequential(fn, args_list, obs, n_jobs, stop):
    """Null increments in index order. With ``stop`` (Besag & Clifford 1991), draws end as soon as
    ``stop`` of them reach ``obs`` and p = stop / draws; otherwise p = (1 + count) / (1 + n).
    Results do not depend on the batch size, because each draw has a fixed seed."""
    if not stop:
        null = _parallel(fn, args_list, n_jobs)
        return null, float((1 + sum(v >= obs for v in null)) / (1 + len(null)))
    batch = max(20, 4 * (n_jobs if n_jobs and n_jobs > 0 else 8))
    null, hits = [], 0
    for i in range(0, len(args_list), batch):
        for v in _parallel(fn, args_list[i:i + batch], n_jobs):
            null.append(v)
            hits += v >= obs
            if hits >= stop:
                return null, float(stop / len(null))
    return null, float((1 + hits) / (1 + len(null)))


def permute_scores(s, groups, strata_key, rng):
    """Permute group-level scores across groups, within strata if strata_key is given (one value per
    item; a group's first item sets its stratum). Item n of a group takes item min(n, size-1) of its
    source group."""
    g = np.unique(groups)
    items = {k: np.where(groups == k)[0] for k in g}
    if strata_key is None:
        target = dict(zip(g, rng.permutation(g)))
    else:
        target = {}
        gs = np.array([strata_key[items[k][0]] for k in g])
        for v in np.unique(gs):
            members = g[gs == v]
            target.update(zip(members, rng.permutation(members)))
    sp = np.empty(len(s))
    for k in g:
        src = items[target[k]]
        for n, i in enumerate(items[k]):
            sp[i] = s[src[min(n, len(src) - 1)]]
    return sp


def score_given_baselines(num, lin, s, groups, seed=0):
    """Out-of-fold E[S | baselines] from a gradient-boosting regressor (grouped CV)."""
    X = np.hstack([num, lin])
    if X.shape[1] == 0:
        return np.full(len(s), s.mean())
    out = np.zeros(len(s))
    for te in group_folds(groups, seed):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)
        out[te] = m.fit(X[~te], s[~te]).predict(X[te])
    return out


def _perm_once(fam, num, lin, s, y, groups, strata_key, base_auc, rng_seed, s_hat=None):
    rng = np.random.default_rng(rng_seed)
    if s_hat is None:
        sp = permute_scores(s, groups, strata_key, rng)
    else:
        sp = s_hat + permute_scores(s - s_hat, groups, strata_key, rng)
    return auc(y, cv_predict(fam, num, lin, sp, y, groups, 0)) - base_auc


def _parallel(fn, args_list, n_jobs):
    if n_jobs == 1:
        return [fn(*a) for a in args_list]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(fn)(*a) for a in args_list)


def boot_ci(y, s, groups, n=2000, seed=0, s2=None):
    """Descriptive 95% CI for AUROC(s) (or AUROC(s) - AUROC(s2)) resampling groups; no refit."""
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    out = []
    while len(out) < n:
        ii = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        if 0 < y[ii].sum() < len(ii):
            a = auc(y[ii], s[ii])
            out.append(a - auc(y[ii], s2[ii]) if s2 is not None else a)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def stratum_base_rate(y, strata, groups, seed):
    """Out of fold: each item scored by its stratum's label rate in the training folds."""
    p = np.zeros(len(y))
    for te in group_folds(groups, seed):
        for st in np.unique(strata[te]):
            tr = (~te) & (strata == st)
            p[te & (strata == st)] = y[tr].mean() if tr.any() else y[~te].mean()
    return p


def audit(df, score, label, group, baselines, strata=(), families=FAMILIES, seeds=range(5), n_perm=200,
          n_boot=200, perm_strata=(), score_transform="rank", rank_within=None, n_jobs=1, n_boot_desc=2000,
          risk_bins=0, null="residual", perm_stop=None, interval="bootstrap"):
    """Full report. n_boot=0 skips the interval refits (fast; permutation p only).

    ``perm_stop``: stop permuting once this many null increments reach the observed one (Besag-Clifford).
    ``interval``: "half" (half-sample refits, centred) or "bootstrap" (refit group bootstrap, the
    pre-0.7 interval, conservative under the null)."""
    if interval not in ("half", "bootstrap"):
        raise ValueError("interval must be 'half' or 'bootstrap'")
    families = tuple(families)
    seeds = list(seeds)
    baselines = list(baselines)
    check_binary_label(df[label], label)
    y = np.asarray(df[label], int)
    s_raw = np.asarray(df[score], float)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, baselines)
    rep = {"version": VERSION, "n_items": int(len(y)), "n_groups": int(len(np.unique(groups))),
           "positive_rate": float(y.mean()), "grouped": bool(group), "warnings": [],
           "settings": {"families": list(families), "n_seeds": len(seeds), "n_perm": n_perm, "n_boot": n_boot,
                        "perm_strata": list(perm_strata), "score_transform": score_transform,
                        "rank_within": rank_within, "risk_bins": risk_bins,
                        "null": null, "perm_stop": perm_stop, "interval": interval}}
    rep["detector_auc"] = auc(y, s_raw)
    rep["detector_auc_ci"] = boot_ci(y, s_raw, groups, n_boot_desc)
    if rank_within:
        rep["detector_auc_ranked_within"] = auc(y, s)

    single = {}                                            # descriptive: each column alone, linear family
    for c in baselines:
        n1, l1 = split_design(df, [c])
        single[c] = float(np.mean([auc(y, cv_predict("linear", n1, l1, None, y, groups, sd)) for sd in seeds]))
    rep["baseline_auc"] = single

    fam, head = estimate(num, lin, s, y, groups, families, seeds)
    rep["families"] = fam
    rep["headline_family"] = head
    rep["baselines_combined_auc"] = fam[head]["baseline_auc"]
    rep["baselines_plus_detector_auc"] = fam[head]["plus_detector_auc"]
    rep["increment"] = fam[head]["increment"]

    if n_boot and interval == "bootstrap":
        res = _parallel(_boot_once, [(num, lin, s, y, groups, families, seeds, 1000 + b) for b in range(n_boot)], n_jobs)
        hb = np.array([r[0] for r in res])
        rep["increment_ci"] = [float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))]
        rep["increment_ci_note"] = "conservative under the null (CALIBRATION.md); not used for adds_information"
        rep["family_increment_ci"] = {k: [float(np.percentile([r[1][k] for r in res], 2.5)),
                                          float(np.percentile([r[1][k] for r in res], 97.5))] for k in families}
    elif n_boot:
        res = _parallel(_half_once, [(num, lin, s, y, groups, families, seeds, 1000 + b) for b in range(n_boot)], n_jobs)

        def _ci(est, draws):
            d = np.asarray(draws) - np.mean(draws)
            return [float(est + np.percentile(d, 2.5)), float(est + np.percentile(d, 97.5))]
        rep["increment_ci"] = _ci(rep["increment"], [r[0] for r in res])
        rep["increment_ci_note"] = "half-sample interval: failed its coverage check (CALIBRATION.md, 0.7.0); experimental"
        rep["family_increment_ci"] = {k: _ci(fam[k]["increment"], [r[1][k] for r in res]) for k in families}
    else:
        rep["increment_ci"] = None

    p_base0 = cv_predict(head, num, lin, None, y, groups, 0)
    base0 = auc(y, p_base0)
    keys = [np.full(len(y), "")]
    if perm_strata:
        user_key = np.array(["|".join(str(df[c].iloc[i]) for c in perm_strata) for i in range(len(y))])
        for k in np.unique(groups):
            if len(set(user_key[groups == k])) > 1:
                rep["warnings"].append("perm_strata vary within a group; the group's first item sets its stratum")
                break
        keys.append(user_key)
    if risk_bins and risk_bins > 1:
        keys.append(np.minimum((_rank01(p_base0) * risk_bins).astype(int), risk_bins - 1).astype(str))
    strata_key = None if len(keys) == 1 else np.array(["|".join(t) for t in zip(*keys)])
    obs0 = auc(y, cv_predict(head, num, lin, s, y, groups, 0)) - base0
    s_hat = score_given_baselines(num, lin, s, groups, 0) if null == "residual" else None
    if s_hat is not None:
        r2 = 1 - np.var(s - s_hat) / max(np.var(s), 1e-12)
        rep["detector_r2_from_baselines"] = float(r2)
    null, p_perm = _perm_sequential(_perm_once, [(head, num, lin, s, y, groups, strata_key, base0, 50000 + b, s_hat)
                                                 for b in range(n_perm)], obs0, n_jobs, perm_stop)
    rep["increment_seed0"] = float(obs0)
    rep["increment_perm_p"] = p_perm
    rep["n_perm_drawn"] = len(null)

    rep["strata"] = {}
    for col in strata:
        st = np.asarray(df[col]).astype(str)
        within = {}
        for v in np.unique(st):
            m = st == v
            entry = {"n": int(m.sum()), "positive_rate": float(y[m].mean())}
            if 0 < y[m].sum() < m.sum():
                entry["detector_auc"] = auc(y[m], s_raw[m])
                entry["detector_auc_ci"] = boot_ci(y[m], s_raw[m], groups[m], n_boot_desc)
            if min(y[m].sum(), m.sum() - y[m].sum()) < 10:
                rep["warnings"].append(f"stratum {col}={v}: fewer than 10 items in one class")
            within[v] = entry
        br = float(np.mean([auc(y, stratum_base_rate(y, st, groups, sd)) for sd in seeds]))
        rep["strata"][col] = {"within": within, "base_rate_only_auc": br}
        if rep["detector_auc"] > 0.5 and (br - 0.5) >= 0.8 * (rep["detector_auc"] - 0.5):
            rep["warnings"].append(f"'{col}' base rate alone reaches {br:.3f}: at least 80% of the detector's "
                                   f"pooled AUROC above chance ({rep['detector_auc']:.3f}) is stratum identity")

    if not group:
        rep["warnings"].append("no group column: items from the same question/response may sit in train and test folds")
    if rep["detector_auc"] < 0.5:
        rep["warnings"].append("detector AUROC < 0.5: check the score's direction")
    best = max(list(single.values()) + [rep["baselines_combined_auc"]])
    if rep["detector_auc"] <= best:
        rep["warnings"].append(f"a cheap baseline ({best:.3f}) matches or beats the detector ({rep['detector_auc']:.3f})")
    rep["adds_information"] = bool(rep["increment_perm_p"] < 0.05 and rep["increment"] > 0)
    if not rep["adds_information"]:
        rep["warnings"].append(f"increment over the {head} baseline is not distinguishable from zero "
                               f"(increment {rep['increment']:+.3f}, permutation p = {rep['increment_perm_p']:.3f})")
    if "linear" in fam and head != "linear" and fam["linear"]["increment"] - fam[head]["increment"] > 0.02:
        rep["warnings"].append(f"the increment over the linear baseline ({fam['linear']['increment']:+.3f}) exceeds the "
                               f"increment over the {head} baseline ({fam[head]['increment']:+.3f}): part of the "
                               f"detector's apparent value is a nonlinear function of the baselines")
    return rep
