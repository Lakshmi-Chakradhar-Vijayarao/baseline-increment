"""0.12: super-learner null score model and clustered DeLong interval (docs/prereg/PREREG_0.12.md)."""
import numpy as np
import pandas as pd

from baseline_increment import audit
from baseline_increment.core import auc, clustered_delong_diff, score_given_baselines, split_design


def test_delong_matches_group_bootstrap():
    r = np.random.default_rng(0)
    g = np.repeat(np.arange(400), 2)
    u = np.repeat(r.normal(size=400), 2)
    y = (r.random(800) < 1 / (1 + np.exp(-u))).astype(bool)
    a, b = u + r.normal(size=800) + 0.5 * y, u + r.normal(size=800)
    est, se = clustered_delong_diff(y, a, b, g)
    assert np.isclose(est, auc(y, a) - auc(y, b))
    bs = []
    for _ in range(1500):
        ii = np.concatenate([np.where(g == k)[0] for k in r.choice(400, 400)])
        bs.append(auc(y[ii], a[ii]) - auc(y[ii], b[ii]))
    assert abs(se / np.std(bs) - 1) < 0.15


def test_super_learner_fits_an_exact_interaction_better_than_gbm():
    r = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"task": np.array(list("ABC"))[np.arange(n) % 3], "L": r.normal(size=n)})
    f = df.L * np.where(df.task == "A", 1, -1)
    s = (f + 0.3 * r.normal(size=n)).to_numpy()
    num, lin = split_design(df, ["task", "L"])
    groups = np.arange(n).astype(str)
    info = {}
    p_sl = score_given_baselines(num, lin, s, groups, 0, "super_learner", info)
    p_gbm = score_given_baselines(num, lin, s, groups, 0, "gbm")
    assert info["score_model"] == "ridge_interactions"
    assert np.mean((p_sl - f) ** 2) < np.mean((p_gbm - f) ** 2)


def test_audit_reports_options():
    df = pd.read_csv("examples/synthetic_ragstyle.csv")
    r = audit(df, "detector_score", "hallucinated", "response_id", ["task_type", "log_words"], families=["linear"],
              seeds=range(1), n_perm=20, n_boot=0, n_boot_desc=20, null="residual_sl", interval="delong")
    assert r["settings"]["null"] == "residual_sl" and r["null_score_model"] in ("gbm", "ridge_interactions")
    lo, hi = r["increment_ci"]
    assert lo < r["increment"] < hi and r["increment_se"] > 0
