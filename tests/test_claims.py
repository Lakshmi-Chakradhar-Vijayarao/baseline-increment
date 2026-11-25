"""Executable claims: each test reads the recorded evidence behind a public claim (claims/claims.json) and fails if
the evidence no longer supports it, or checks the claimed behaviour directly."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline_increment import audit
from baseline_increment.core import group_folds

ROOT = Path(__file__).parents[1]
CLAIMS = json.loads((ROOT / "claims/claims.json").read_text())["claims"]
NULLS = ["noise", "metadata_copy", "nonlinear_of_metadata", "metadata_plus_noise", "random_labels", "few_groups",
         "unequal_groups", "imbalance", "misspecified_baseline", "interaction_meta", "difficulty_by_model",
         "scale_by_model", "simpson"]


def _small(seed=0, signal=1.0):
    r = np.random.default_rng(seed)
    n = 300
    task = np.array(list("AB") * (n // 2))
    y = (r.random(n) < np.where(task == "A", 0.6, 0.25)).astype(int)
    return pd.DataFrame({"g": np.arange(n), "task": task, "y": y, "s": signal * y + r.normal(size=n)})


KW = dict(families=["linear"], seeds=range(1), n_perm=40, n_boot=0, n_boot_desc=10)


def test_every_established_claim_has_evidence_and_a_test():
    for c in CLAIMS:
        assert c["status"] in ("established", "documented_limitation", "under_validation", "not_established"), c["id"]
        if c["status"] in ("established", "documented_limitation"):
            assert c["evidence"] and c["test"], c["id"]
            f = ROOT / c["test"].split("::")[0]
            assert f.exists(), c["id"]
            assert c["test"].split("::")[1] in f.read_text(), c["id"]


def test_claim_001_decision_rule():
    for seed, sig in [(0, 1.0), (1, 0.0), (2, 0.3)]:
        for mg in (0.0, 0.02):
            r = audit(_small(seed, sig), "s", "y", "g", ["task"], min_gain=mg, **KW)
            lo, hi = r["gain_ci"]["lower"], r["gain_ci"]["upper"]
            want = "PASS" if lo > mg else "FAIL" if hi < mg else "INCONCLUSIVE"
            assert r["incremental_prediction"]["verdict"] == want
            assert r["adds_information"] == (r["increment_perm_p"] < 0.05 and r["increment"] > 0)    # secondary, unchanged


def test_claim_002_groups_never_straddle():
    r = np.random.default_rng(3)
    groups = r.integers(0, 97, 1000).astype(str)
    for seed in range(5):
        folds = group_folds(groups, seed)
        assert np.all(np.sum(folds, axis=0) == 1)
        for te in folds:
            assert not set(groups[te]) & set(groups[~te])


def test_claim_003_rank_invariance():
    df = _small(4, 0.8)
    a = audit(df, "s", "y", "g", ["task"], **KW)
    df2 = df.assign(s=np.exp(3 * df.s) + 7)
    b = audit(df2, "s", "y", "g", ["task"], **KW)
    assert a["adds_information"] == b["adds_information"] and a["increment"] == pytest.approx(b["increment"], abs=1e-12)


def _fast():
    return json.loads((ROOT / "simulations/results/results_v0.7.0_fast.json").read_text())["scenarios"]


def test_claim_004_fast_null_rates():
    s = _fast()
    assert max(s[k]["reject_rate"] for k in NULLS) <= 0.10


def test_claim_005_fast_power():
    s = _fast()
    assert s["moderate_signal"]["reject_rate"] >= 0.80 and s["strong_signal"]["reject_rate"] >= 0.80


def test_claim_006_real_data_agreement():
    v = json.loads((ROOT / "validation/results.json").read_text())["agreement"]
    assert v["verdict"] == "AGREES" and all(c["A1"] and c["A2"] for c in v["checks"].values()) and v["A3_ordering"]["pass"]


def test_claim_007_real_data_nulls():
    v = json.loads((ROOT / "validation/results.json").read_text())["nulls"]
    assert all(x["flag_rate"] <= 0.10 for k, x in v.items() if not k.endswith("/P1"))


def test_claim_019_documented_limitation():
    t = json.loads((ROOT / "validation/judges/trec_strata_results.json").read_text())["arms"]
    assert t["A0"]["flag_rate"] > 0.10 and t["A1"]["flag_rate"] > 0.10
    d = json.loads((ROOT / "simulations/results/diag_v0.12_3000_sl.json").read_text())["cells"]
    assert d["D10|3000|residual_sl"]["flag_rate"] > 0.10 and d["D8|3000|residual_sl"]["flag_rate"] > 0.10


def test_claim_020_model_class_gain():
    d = json.loads((ROOT / "simulations/results/diag_v0.12_3000_sl.json").read_text())["cells"]
    assert d["D4|3000|residual_sl"]["mean_increment"] > 0.02 and d["D4|3000|residual_sl"]["flag_rate"] > 0.5


def test_no_pending_results_at_release():
    import os
    if not os.environ.get("BI_RELEASE"):
        pytest.skip("checked only when BI_RELEASE=1")
    assert "PENDING" not in (ROOT / "docs/technical-report.md").read_text()


def test_claim_013_interval_coverage():
    s = json.loads((ROOT / "simulations/results/coverage_v0.12.0_delong.json").read_text())["scenarios"]
    assert len(s) == 6 and all(0.85 <= v["ci_coverage_of_mean_estimate"] <= 0.99 for v in s.values())


def test_claim_014_sensitivity_no_false_comfort():
    v = json.loads((ROOT / "simulations/results/sensitivity_v0.13.0.json").read_text())["verdict"]
    assert v["V1_pass"] and v["V1_pooled"] >= 0.95 and min(v["V1_by_family"].values()) >= 0.90
