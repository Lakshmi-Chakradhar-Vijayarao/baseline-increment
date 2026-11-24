"""0.13 modules: omitted-baseline sensitivity and the sample-size planner (fast smoke tests; validation is in
simulations/sensitivity_suite.py and simulations/planner_suite.py)."""
import numpy as np
import pandas as pd

import baseline_increment.planner as planner
from baseline_increment.sensitivity import sensitivity, sentence


def test_benchmark_that_carries_the_signal_has_high_phi():
    r = np.random.default_rng(0)
    n = 800
    L, Z = r.normal(size=n), r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(-(0.8 * L + Z)))).astype(int)
    df = pd.DataFrame({"g": np.arange(n), "L": L, "Z": Z, "y": y, "s": Z + r.normal(size=n)})
    out = sensitivity(df, "s", "y", "g", ["L", "Z"], families=("linear",), seeds=range(1))
    # the detector is a noisy copy of Z: with Z among the baselines the residual correlation is ~0; leaving Z out
    # would make it large, so Z's erasure index is high and L's is low
    assert out["benchmarks"]["Z"]["phi"] > out["benchmarks"]["L"]["phi"]
    assert "linear approximation" in sentence(out) or "nothing to erase" in sentence(out)


def test_planner_calibrates_and_power_rises(monkeypatch):
    monkeypatch.setattr(planner, "POP_N", 20_000)
    c = planner.calibrate(0.3, 0.70, 2, 0.05)
    assert c["theta"] > 0
    inc = planner._pop_increment(2, c["a"], c["b"], c["theta"])
    assert abs(inc - 0.05) < 0.01
    res = planner.plan(0.3, 0.70, 2, 0.05, grid=[50, 800], reps=12, n_jobs=1, calib=c)
    p = [r["power"] for r in res["grid"]]
    assert p[1] >= p[0] and p[1] >= 0.75
