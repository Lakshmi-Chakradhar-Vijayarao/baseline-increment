import itertools

import numpy as np
import pandas as pd
import pytest

from baseline_increment import audit
from baseline_increment.calibration import _logistic_fit, score_calibration
from baseline_increment.core import auc
from baseline_increment.ordinal import audit_ordinal, check_ordinal_label, concordance


def brute_c(y, p):
    conc = pairs = 0.0
    for i, j in itertools.combinations(range(len(y)), 2):
        if y[i] == y[j]:
            continue
        hi, lo = (i, j) if y[i] > y[j] else (j, i)
        pairs += 1
        conc += 1.0 if p[hi] > p[lo] else 0.5 if p[hi] == p[lo] else 0.0
    return conc / pairs


def test_concordance_equals_auc_on_binary():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.normal(size=300) + y, 1)
    assert abs(concordance(y, p) - auc(y, p)) < 1e-12


def test_concordance_equals_brute_force_with_ties():
    rng = np.random.default_rng(2)
    for _ in range(5):
        y = rng.integers(0, 4, 80)
        p = np.round(rng.normal(size=80) + 0.5 * y, 0)            # many ties
        assert abs(concordance(y, p) - brute_c(y, p)) < 1e-12


def test_binary_path_rejects_graded_labels():
    df = pd.DataFrame({"g": range(90), "y": [0, 1, 2] * 30, "s": np.arange(90.0), "t": ["a", "b"] * 45})
    with pytest.raises(ValueError, match="--outcome ordinal"):
        audit(df, "s", "y", "g", ["t"], seeds=range(1), n_perm=5, n_boot=0)


def test_ordinal_label_checks():
    with pytest.raises(ValueError, match="integer levels"):
        check_ordinal_label(pd.Series([0.5, 1, 2]))
    with pytest.raises(ValueError, match="binary outcome"):
        check_ordinal_label(pd.Series([0, 1, 0, 1]))


def _ord_data(signal, n_groups=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        task = "ABC"[g % 3]
        L = rng.normal()
        for _ in range(2):
            eta = {"A": 1.2, "B": -0.6, "C": -0.6}[task] + 0.8 * L + rng.logistic()
            sev = int(np.digitize(eta, [-1, 0.5, 2]))
            rows.append({"g": g, "task": task, "L": L, "sev": sev, "s": rng.normal() + signal * sev})
    return pd.DataFrame(rows)


def test_ordinal_audit_finds_signal_and_not_noise():
    kw = dict(families=("linear",), seeds=range(1), n_perm=40, perm_stop=5, n_boot_desc=100)
    assert audit_ordinal(_ord_data(1.0), "s", "sev", "g", ["task", "L"], **kw)["adds_information"]
    assert not audit_ordinal(_ord_data(0.0, seed=3), "s", "sev", "g", ["task", "L"], **kw)["adds_information"]


def test_logistic_fit_recovers_known_slope_and_intercept():
    rng = np.random.default_rng(4)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 0.6 * x)))).astype(float)
    a, b = _logistic_fit(x, y)
    assert abs(a - 0.3) < 0.05 and abs(b - 0.6) < 0.05


def test_calibration_of_a_calibrated_score():
    rng = np.random.default_rng(5)
    p = rng.beta(2, 3, 4000)
    y = (rng.random(4000) < p).astype(int)
    c = score_calibration(pd.DataFrame({"p": p, "y": y}), "p", "y", n_boot=200)
    assert abs(c["slope"] - 1) < 0.15 and abs(c["citl"]) < 0.1 and c["slope_ci"][0] < 1 < c["slope_ci"][1]


def test_calibration_rejects_non_probabilities():
    with pytest.raises(ValueError, match="probability"):
        score_calibration(pd.DataFrame({"p": [0.2, 1.4], "y": [0, 1]}), "p", "y", n_boot=0)


def test_cli_ordinal_and_report(tmp_path):
    from baseline_increment.cli import main
    f = tmp_path / "o.csv"
    _ord_data(1.0).to_csv(f, index=False)
    out = tmp_path / "o.md"
    rc = main([str(f), "--score", "s", "--label", "sev", "--group", "g", "--baselines", "task,L", "--outcome", "ordinal",
               "--families", "linear", "--seeds", "1", "--perms", "40", "--perm-stop", "5", "--report", str(out),
               "--fail-if-no-increment"])
    assert rc == 0
    t = out.read_text()
    assert "graded labels" in t and "concordance" in t and "Operating points" in t


def test_cli_ordinal_rejects_unsupported_flags(tmp_path, capsys):
    from baseline_increment.cli import main
    f = tmp_path / "o.csv"
    _ord_data(1.0).to_csv(f, index=False)
    rc = main([str(f), "--score", "s", "--label", "sev", "--group", "g", "--baselines", "task", "--outcome", "ordinal",
               "--strata", "task"])
    assert rc == 2 and "does not support --strata" in capsys.readouterr().err


def test_cli_calibration_section(tmp_path):
    from baseline_increment.cli import main
    rng = np.random.default_rng(6)
    p = rng.beta(2, 3, 600)
    df = pd.DataFrame({"g": np.repeat(np.arange(300), 2), "t": ["a", "b"] * 300, "y": (rng.random(600) < p).astype(int), "p": p})
    f = tmp_path / "c.csv"
    df.to_csv(f, index=False)
    out = tmp_path / "c.md"
    main([str(f), "--score", "p", "--label", "y", "--group", "g", "--baselines", "t", "--fast", "--families", "linear",
          "--jobs", "1", "--calibration", "--report", str(out)])
    t = out.read_text()
    assert "The raw score as a probability" in t and "Calibration slope" in t and "bin 10" in t
