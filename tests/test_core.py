import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baseline_increment import audit, auc
from baseline_increment.core import permute_scores

FAST = dict(seeds=range(2), n_perm=40, n_boot=0, n_boot_desc=50)


def make(n_groups=300, per=2, signal=0.0, stratum_effect=1.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        task = ["A", "B", "C"][g % 3]
        base = {"A": 0.8, "B": 0.3, "C": 0.3}[task]
        for _ in range(per):
            y = rng.random() < base
            score = stratum_effect * (task == "A") + signal * y + rng.normal()
            rows.append({"g": g, "task": task, "length": rng.normal(), "y": int(y), "s": score})
    return pd.DataFrame(rows)


def make_quadratic(n_groups=300, seed=0):
    """Label depends on L^2; the detector is L^2 + noise, so it adds nothing beyond L."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        L = rng.normal()
        for _ in range(2):
            y = int(rng.random() < 1 / (1 + np.exp(-1.2 * (L ** 2 - 1))))
            rows.append({"g": g, "L": L, "y": y, "s": L ** 2 + 0.3 * rng.normal()})
    return pd.DataFrame(rows)


def test_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200), 1)          # ties
    assert abs(auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_stratum_only_detector_is_flagged():
    rep = audit(make(signal=0.0), "s", "y", "g", ["task", "length"], ["task"], **FAST)
    assert rep["detector_auc"] > 0.6                       # looks good pooled
    assert abs(rep["increment"]) < 0.03                    # adds nothing over task
    assert rep["increment_perm_p"] > 0.05
    assert any("base rate alone" in w for w in rep["warnings"])
    assert any("not distinguishable from zero" in w for w in rep["warnings"])
    for e in rep["strata"]["task"]["within"].values():
        assert abs(e["detector_auc"] - 0.5) < 0.12


def test_real_signal_survives():
    rep = audit(make(signal=1.5), "s", "y", "g", ["task", "length"], ["task"], **FAST)
    assert rep["increment"] > 0.1
    assert rep["increment_perm_p"] < 0.05
    assert not any("not distinguishable from zero" in w for w in rep["warnings"])


def test_missing_group_warns():
    rep = audit(make(signal=1.0), "s", "y", None, ["task"], **FAST)
    assert any("no group column" in w for w in rep["warnings"])


def test_flexible_baseline_blocks_nonlinear_metadata_detector():
    """v0.1 failure: a linear baseline credited L^2 with +0.09. v0.2 must not."""
    rep = audit(make_quadratic(), "s", "y", "g", ["L"], **FAST)
    assert rep["families"]["linear"]["increment"] > 0.05   # the old failure is still visible...
    assert rep["headline_family"] in ("spline", "gbm")     # ...but the headline uses a flexible baseline
    assert abs(rep["increment"]) < 0.02
    assert rep["increment_perm_p"] > 0.05
    assert any("nonlinear function of the baselines" in w for w in rep["warnings"])


def test_scale_invariance():
    df = make(signal=1.0)
    reps = []
    for f in (lambda v: v, lambda v: 7 * v + 12, np.exp):
        d = df.copy(); d["s"] = f(df["s"])
        reps.append(audit(d, "s", "y", "g", ["task", "length"], **FAST))
    for r in reps[1:]:
        assert r["increment"] == reps[0]["increment"]
        assert r["increment_perm_p"] == reps[0]["increment_perm_p"]


def test_permutation_stays_within_strata():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 2).astype(str)
    strata = np.where(np.arange(80) < 40, "a", "b")
    s = np.where(strata == "a", 0.0, 100.0) + rng.normal(size=80)
    sp = permute_scores(s, groups, strata, rng)
    assert (sp[strata == "a"] < 50).all() and (sp[strata == "b"] > 50).all()
    assert sorted(np.round(sp, 9)) == sorted(np.round(s, 9))


def test_refit_bootstrap_ci_present_and_ordered():
    rep = audit(make(signal=1.5, n_groups=150), "s", "y", "g", ["task"], seeds=range(2), n_perm=10, n_boot=20,
                n_boot_desc=50, families=("linear",))
    lo, hi = rep["increment_ci"]
    assert lo < hi
    assert lo > 0                                          # strong planted signal


def test_perm_stop_never_flags_what_the_full_run_would_not():
    df = make(signal=0.0, seed=4)
    early = audit(df, "s", "y", "g", ["task"], seeds=range(2), n_perm=60, n_boot=0, n_boot_desc=50, perm_stop=5)
    assert early["n_perm_drawn"] <= 60
    if early["n_perm_drawn"] < 60:
        assert early["increment_perm_p"] >= 5 / 60 and not early["adds_information"]


def test_perm_stop_runs_all_draws_on_strong_signal():
    df = make(signal=1.5, seed=5)
    rep = audit(df, "s", "y", "g", ["task"], seeds=range(2), n_perm=40, n_boot=0, n_boot_desc=50, perm_stop=5)
    assert rep["n_perm_drawn"] == 40 and rep["adds_information"]


def test_half_interval_brackets_estimate_and_is_deterministic():
    df = make(signal=0.8, seed=6)
    kw = dict(seeds=range(1), n_perm=5, n_boot=8, n_boot_desc=20, interval="half", families=("linear",))
    a = audit(df, "s", "y", "g", ["task"], **kw)
    b = audit(df, "s", "y", "g", ["task"], **kw)
    assert a["increment_ci"] == b["increment_ci"]
    assert a["increment_ci"][0] <= a["increment"] <= a["increment_ci"][1]
    assert set(a["family_increment_ci"]) == set(a["families"])


def test_bad_interval_name_raises():
    import pytest
    with pytest.raises(ValueError):
        audit(make(), "s", "y", "g", ["task"], interval="bca", **FAST)
