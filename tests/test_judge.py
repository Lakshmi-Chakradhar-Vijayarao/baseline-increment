import numpy as np
import pandas as pd

from baseline_increment.cli import main
from baseline_increment.judge import LENGTH_COL, agreement, judge_report


def _data(n=400, signal=1.5, seed=0):
    rng = np.random.default_rng(seed)
    task = rng.integers(0, 3, n)
    words = rng.integers(5, 60, n)
    z = 0.8 * (task - 1) + 0.02 * (words - 30) + rng.normal(size=n)
    y = (z > 0).astype(int)
    judge = (signal * (2 * y - 1) + 0.03 * words + rng.normal(size=n) > 0.9).astype(int)
    resp = [" ".join(["w"] * w) for w in words]
    return pd.DataFrame({"item": np.arange(n) // 2, "task": task, "resp": resp, "judge": judge, "human": y})


def test_agreement_and_kappa():
    a = agreement([1, 1, 0, 0], [1, 0, 0, 0])
    assert a["agreement"] == 0.75
    assert abs(a["cohen_kappa"] - 0.5) < 1e-9
    assert agreement([0.2, 0.9], [0, 1]) is None


def test_judge_report_strong_judge_passes_and_derives_length():
    df = _data()
    rep = judge_report(df, "judge", "human", ["task"], group="item", response="resp", n_jobs=1)
    j = rep["judge"]
    assert LENGTH_COL in j["baselines"]
    assert rep["incremental_prediction"]["verdict"] == "PASS"
    assert j["agreement"] is not None and 0 <= j["agreement"]["agreement"] <= 1
    assert rep["conditional_information"]["status"] == "not run in judge mode"
    assert rep["gain_ci"]["method"] == "delong"
    assert "improves held-out prediction" in j["sentence"]


def test_judge_cli(tmp_path, capsys):
    p = tmp_path / "a.csv"
    _data(signal=0.0, seed=1).to_csv(p, index=False)
    assert main(["judge", str(p), "--judge", "judge", "--label", "human", "--baselines", "task", "--response", "resp",
                 "--group", "item", "--jobs", "1", "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "Judge's incremental gain" in out and "Baselines alone, AUROC" in out
    assert (tmp_path / "r.json").exists()


def test_missing_judge_rows_dropped_and_counted():
    df = _data()
    df.loc[[0, 5], "judge"] = np.nan
    rep = judge_report(df, "judge", "human", ["task"], group="item", n_jobs=1)
    assert rep["rows_dropped_missing_judge_or_label"] == 2
    assert rep["n_items"] == len(df) - 2
