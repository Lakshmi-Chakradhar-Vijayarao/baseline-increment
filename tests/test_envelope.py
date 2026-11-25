"""Validity envelope: status beyond pass/fail, never changing adds_information."""
import numpy as np
import pandas as pd

from baseline_increment import audit
from baseline_increment.cli import main
from baseline_increment.envelope import status


def _df(n, pos, seed=0):
    r = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[:pos] = 1
    r.shuffle(y)
    return pd.DataFrame({"g": np.arange(n), "t": r.choice(list("AB"), n), "y": y, "s": y + r.normal(size=n)})


KW = dict(families=["linear"], seeds=range(1), n_perm=20, n_boot=0, n_boot_desc=10)


def test_status_levels():
    assert status(audit(_df(400, 120), "s", "y", "g", ["t"], **KW))["status"] == "within_validated_range"
    assert status(audit(_df(400, 20), "s", "y", "g", ["t"], **KW))["status"] == "outside_validated_range"
    assert status(audit(_df(200, 6), "s", "y", "g", ["t"], **KW))["status"] == "not_testable"


def test_gate_fails_when_not_testable(tmp_path, capsys):
    f = tmp_path / "d.csv"
    _df(200, 6).to_csv(f, index=False)
    rc = main([str(f), "--score", "s", "--label", "y", "--group", "g", "--baselines", "t", "--families", "linear",
               "--seeds", "1", "--perms", "20", "--boot", "0", "--fail-if-no-increment", "--report", str(tmp_path / "r.md")])
    out = capsys.readouterr().out
    assert rc == 1 and "NOT TESTABLE" in out
    assert "Not testable:" in (tmp_path / "r.md").read_text()


def test_ladder_auprc_and_duplicates(tmp_path):
    import json
    df = _df(400, 120)
    df["q"] = [f"question {i % 150}" for i in range(400)]          # same text in different groups
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    out = tmp_path / "a.json"
    main([str(f), "--score", "s", "--label", "y", "--group", "g", "--baselines", "t", "--families", "linear",
          "--seeds", "1", "--perms", "20", "--boot", "0", "--ladder", "--text", "q", "--json", str(out)])
    rep = json.loads(out.read_text())
    assert len(rep["ladder"]) == 2 and "ap_gain" in rep["precision_recall"]
    assert rep["duplicates"]["duplicate_texts_across_groups"] == 150
    assert any("appear in more than one group" in w for w in rep["warnings"])
