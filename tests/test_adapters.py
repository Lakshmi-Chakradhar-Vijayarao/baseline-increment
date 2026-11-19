"""Adapters are checked against known results: the per-sample table must reproduce the harness's
own aggregate metric, computed independently by the harness itself."""
import json
from pathlib import Path

import pytest

from baseline_increment.adapters import from_inspect, from_lm_eval
from baseline_increment.cli import main

FIX = Path(__file__).parent / "fixtures"
SAMPLES = sorted((FIX / "lm_eval").glob("samples_*.jsonl"))


def test_lm_eval_rows_reproduce_the_harness_accuracy():
    df = from_lm_eval(SAMPLES)
    res = json.loads((FIX / "lm_eval/results_pythia-14m.json").read_text())["results"]
    assert set(df["task"]) == {"arc_easy", "piqa"} and len(df) == 80
    for task, g in df.groupby("task"):
        assert 1 - g["wrong"].mean() == pytest.approx(res[task]["acc,none"])
    assert df["risk"].between(0, 1).all() and (df["n_choices"] >= 2).all()
    assert df["group"].is_unique


def test_lm_eval_acc_norm_metric():
    df = from_lm_eval(SAMPLES, metric="acc_norm")
    res = json.loads((FIX / "lm_eval/results_pythia-14m.json").read_text())["results"]
    for task, g in df.groupby("task"):
        assert 1 - g["wrong"].mean() == pytest.approx(res[task]["acc_norm,none"])


def test_inspect_rows_reproduce_scorer_accuracy():
    d = json.loads((FIX / "arith_mockllm.json").read_text())
    acc = next(s for s in d["results"]["scores"] if s["name"] == "match")["metrics"]["accuracy"]["value"]
    df = from_inspect([FIX / "arith_mockllm.json"], label_scorer="match", score_scorer="includes")
    assert 1 - df["wrong"].mean() == pytest.approx(acc)
    assert {"log_input_words", "log_output_words", "model"} <= set(df.columns)


def test_inspect_needs_a_score():
    with pytest.raises(ValueError):
        from_inspect([FIX / "arith_mockllm.json"], label_scorer="match")


def test_cli_from_lm_eval(tmp_path, capsys):
    out = tmp_path / "items.csv"
    assert main(["from-lm-eval", *map(str, SAMPLES), "-o", str(out)]) == 0
    assert "80 rows" in capsys.readouterr().out and out.exists()


def test_tpr_at_fpr_and_operating_point():
    import numpy as np
    import pandas as pd

    from baseline_increment.core import operating_point, tpr_at_fpr
    y = np.array([0] * 10 + [1] * 10)
    p = np.r_[np.linspace(0, 0.9, 10), np.linspace(0.5, 1.4, 10)]
    assert tpr_at_fpr(y, p, 0.0) == 0.5 and tpr_at_fpr(y, p, 0.5) == 1.0
    rng = np.random.default_rng(0)
    n = 400
    task = rng.integers(0, 2, n)
    yy = (rng.uniform(size=n) < np.where(task == 1, 0.6, 0.2)).astype(int)
    df = pd.DataFrame({"task": task, "y": yy, "s": yy + rng.normal(0, 1, n), "g": np.arange(n)})
    op = operating_point(df, "s", "y", "g", ["task"], fpr=0.1, seeds=range(2))
    assert op["tpr_plus_detector"] > op["tpr_baselines"]
