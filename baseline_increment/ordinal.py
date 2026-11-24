"""Graded (ordinal) labels, e.g. severity 0-3 (pre-registered in docs/prereg/PREREG_0.11.md).

A separate path that reuses the binary engine's folds, design, score transform and conditional-randomisation null
without modifying them (the binary engine is frozen by tests/test_golden.py).

Metric: concordance (Harrell's C for an ordinal outcome): among pairs of items with different levels, the share
where the higher level has the higher prediction, prediction ties counting 1/2. It equals AUROC on 0/1 labels.
"""
import warnings as _w

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .core import (FAMILIES, VERSION, GroupIndex, resolve_null, _features, _perm_sequential, group_folds, permute_scores,
                   score_given_baselines, split_design, transform_score)

MIN_LEVELS, MAX_LEVELS = 3, 10


def check_ordinal_label(col, name="label"):
    v = np.asarray(col, dtype=float)
    if np.isnan(v).any():
        raise ValueError(f"label {name!r} has missing values")
    if not np.all(v == np.round(v)):
        raise ValueError(f"ordinal label {name!r} must be integer levels; continuous labels are not supported")
    k = len(np.unique(v))
    if not MIN_LEVELS <= k <= MAX_LEVELS:
        raise ValueError(f"ordinal label {name!r} needs {MIN_LEVELS}-{MAX_LEVELS} distinct levels; found {k}"
                         + (" (use the default binary outcome for 0/1 labels)" if k == 2 else ""))
    return v.astype(int)


def concordance(y, p):
    """Harrell's C for an ordinal y: concordant pairs (+1/2 per prediction tie) over pairs with different y."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    levels = np.unique(y)
    conc, pairs = 0.0, 0.0
    for b in levels[1:]:
        m = y <= b
        r = rankdata(p[m])                                  # ranks among items at level <= b
        in_b = y[m] == b
        nb, nlow = in_b.sum(), (~in_b).sum()
        # U statistic of level b against all lower levels: sum of b's ranks minus their ranks among themselves
        conc += r[in_b].sum() - nb * (nb + 1) / 2
        pairs += nb * nlow
    return float(conc / pairs) if pairs else float("nan")


def _fit_predict_reg(fam, X, y, tr, te, seed):
    if X.shape[1] == 0 or y[tr].min() == y[tr].max():
        return np.full(int(te.sum()), y[tr].mean())
    if fam == "gbm":
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)
        return m.fit(X[tr], y[tr]).predict(X[te])
    sc = StandardScaler().fit(X[tr])
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
    return m.predict(sc.transform(X[te]))


def cv_predict_reg(fam, num, lin, s, y, groups, seed):
    p = np.zeros(len(y))
    for te in group_folds(groups, seed):
        tr = ~te
        p[te] = _fit_predict_reg(fam, _features(fam, num, lin, s, tr), y, tr, te, seed)
    return p


def _perm_once_ord(fam, num, lin, s, y, groups, strata_key, base_c, rng_seed, s_hat):
    rng = np.random.default_rng(rng_seed)
    sp = s_hat + permute_scores(s - s_hat, groups, strata_key, rng)
    return concordance(y, cv_predict_reg(fam, num, lin, sp, y, groups, 0)) - base_c


def _boot_c(y, s, groups, n, seed=0):
    rng = np.random.default_rng(seed)
    gi = GroupIndex(groups)
    out = []
    while len(out) < n:
        ii = gi.boot(rng)
        if len(np.unique(y[ii])) > 1:
            out.append(concordance(y[ii], s[ii]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def audit_ordinal(df, score, label, group, baselines, families=FAMILIES, seeds=range(2), n_perm=200, perm_stop=10,
                  score_transform="rank", rank_within=None, n_jobs=1, n_boot_desc=2000, null="residual"):
    families, seeds, baselines = tuple(families), list(seeds), list(baselines)
    y = check_ordinal_label(df[label], label)
    s_raw = np.asarray(df[score], float)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, baselines)
    lv, cnt = np.unique(y, return_counts=True)
    rep = {"version": VERSION, "outcome": "ordinal", "n_items": int(len(y)), "n_groups": int(len(np.unique(groups))),
           "levels": {int(a): int(b) for a, b in zip(lv, cnt)}, "grouped": bool(group), "warnings": [],
           "settings": {"outcome": "ordinal", "families": list(families), "n_seeds": len(seeds), "n_perm": n_perm,
                        "perm_stop": perm_stop, "score_transform": score_transform, "rank_within": rank_within,
                        "null": resolve_null(null), "risk_bins": 0, "n_boot": 0}}
    rep["detector_c"] = concordance(y, s_raw)
    rep["detector_c_ci"] = _boot_c(y, s_raw, groups, n_boot_desc)
    single = {}
    for c in baselines:
        n1, l1 = split_design(df, [c])
        single[c] = float(np.mean([concordance(y, cv_predict_reg("linear", n1, l1, None, y, groups, sd)) for sd in seeds]))
    rep["baseline_c"] = single
    fam = {}
    for f in families:
        b = [concordance(y, cv_predict_reg(f, num, lin, None, y, groups, sd)) for sd in seeds]
        a = [concordance(y, cv_predict_reg(f, num, lin, s, y, groups, sd)) for sd in seeds]
        fam[f] = {"baseline_c": float(np.mean(b)), "plus_detector_c": float(np.mean(a)), "increment": float(np.mean(a) - np.mean(b))}
    head = max(families, key=lambda k: fam[k]["baseline_c"])
    rep.update({"families": fam, "headline_family": head, "baselines_combined_c": fam[head]["baseline_c"],
                "baselines_plus_detector_c": fam[head]["plus_detector_c"], "increment": fam[head]["increment"]})
    base0 = concordance(y, cv_predict_reg(head, num, lin, None, y, groups, 0))
    obs0 = concordance(y, cv_predict_reg(head, num, lin, s, y, groups, 0)) - base0
    null = resolve_null(null)
    if null == "permute":
        raise ValueError("ordinal outcome supports the residual nulls only")
    s_hat = score_given_baselines(num, lin, s, groups, 0, "super_learner" if null == "residual_sl" else "gbm")
    null, p = _perm_sequential(_perm_once_ord, [(head, num, lin, s, y, groups, None, base0, 50000 + b, s_hat)
                                                for b in range(n_perm)], obs0, n_jobs, perm_stop)
    rep.update({"increment_seed0": float(obs0), "increment_perm_p": p, "n_perm_drawn": len(null)})
    rep["adds_information"] = bool(p < 0.05 and rep["increment"] > 0)
    if not group:
        rep["warnings"].append("no group column: items from the same question/response may sit in train and test folds")
    if not rep["adds_information"]:
        rep["warnings"].append(f"increment over the {head} baseline is not distinguishable from zero "
                               f"(increment {rep['increment']:+.3f}, permutation p = {p:.3f})")
    if rep["detector_c"] <= max(list(single.values()) + [rep["baselines_combined_c"]]):
        rep["warnings"].append("a cheap baseline matches or beats the detector's concordance")
    return rep


def fmt_ordinal(rep):
    lines = [f"baseline-increment {rep['version']} (ordinal outcome)",
             f"items {rep['n_items']}  groups {rep['n_groups']}  levels {rep['levels']}",
             f"detector concordance      {rep['detector_c']:.3f} [{rep['detector_c_ci'][0]:.3f}, {rep['detector_c_ci'][1]:.3f}]"]
    for k, v in rep["baseline_c"].items():
        lines.append(f"baseline {k:<17} {v:.3f}  (linear, alone)")
    for f, v in rep["families"].items():
        star = "  <- headline (best baselines-only concordance)" if f == rep["headline_family"] else ""
        lines.append(f"family {f:<7} baselines {v['baseline_c']:.3f}  + detector {v['plus_detector_c']:.3f}  "
                     f"increment {v['increment']:+.3f}{star}")
    lines.append(f"increment ({rep['headline_family']})      {rep['increment']:+.3f}  perm p={rep['increment_perm_p']:.3f}"
                 f"  adds_information={str(rep['adds_information']).lower()}")
    lines += [f"WARNING: {w}" for w in rep["warnings"]]
    return "\n".join(lines)
