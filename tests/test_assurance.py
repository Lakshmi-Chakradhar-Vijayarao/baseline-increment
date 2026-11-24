"""1.0 assurance layer: declared policy, evidence package, hash-chained run ledger. None of it changes a verdict."""
import json

import numpy as np
import pandas as pd
import pytest

from baseline_increment.cli import main
from baseline_increment.evidence import read_ledger, verify
from baseline_increment.policy import PolicyError, load_policy


def _data(tmp_path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    task = np.array(list("AB") * (n // 2))
    y = (rng.random(n) < np.where(task == "A", 0.6, 0.25)).astype(int)
    df = pd.DataFrame({"g": np.arange(n), "task": task, "L": rng.normal(size=n), "y": y, "s": 1.5 * y + rng.normal(size=n)})
    df["s2"] = df["s"] + rng.normal(size=n)
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    return f


def _policy(tmp_path, **extra):
    p = {"version": 1, "detector": {"name": "demo", "score": "s"}, "label": "y", "group": "g", "baselines": ["task"], **extra}
    f = tmp_path / "policy.json"
    f.write_text(json.dumps(p))
    return f


def test_policy_refuses_uncalibrated_rules(tmp_path):
    with pytest.raises(PolicyError, match="not yet calibrated.*minimum_gain"):
        load_policy(_policy(tmp_path, minimum_gain=0.05))
    with pytest.raises(PolicyError, match="alpha"):
        load_policy(_policy(tmp_path, alpha=0.01))
    with pytest.raises(PolicyError, match="unknown policy keys"):
        load_policy(_policy(tmp_path, baseline=["task"]))
    _, h1 = load_policy(_policy(tmp_path))
    _, h2 = load_policy(_policy(tmp_path))
    assert h1 == h2 and len(h1) == 64


def test_policy_run_matches_command_line_run(tmp_path):
    f = _data(tmp_path)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert main([str(f), "--policy", str(_policy(tmp_path)), "--json", str(a), "--families", "linear"]) == 2   # refused
    main([str(f), "--policy", str(_policy(tmp_path)), "--json", str(a)])
    main([str(f), "--score", "s", "--label", "y", "--group", "g", "--baselines", "task", "--fast", "--json", str(b)])
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    for k in ("increment", "increment_perm_p", "adds_information", "baselines_combined_auc"):
        assert ra[k] == rb[k]


def test_evidence_package_ledger_and_tamper_detection(tmp_path):
    f = _data(tmp_path)
    ev = tmp_path / "evidence"
    pol = _policy(tmp_path)
    main([str(f), "--policy", str(pol), "--evidence-dir", str(ev)])
    main([str(f), "--score", "s", "--label", "y", "--group", "g", "--baselines", "task,L", "--fast", "--evidence-dir", str(ev)])
    runs = read_ledger(ev)
    assert [r["run"] for r in runs] == [1, 2] and runs[1]["prev_entry_sha256"]
    pkg = ev / runs[1]["evaluation_id"]
    names = sorted(x.name for x in pkg.iterdir())
    assert names == sorted(["MANIFEST.sha256", "command.txt", "environment.json", "ledger.jsonl", "provenance.json",
                            "report.html", "results.json", "specification.json"])
    import html
    t = html.unescape((pkg / "report.html").read_text())
    assert "2 of 2 recorded" in t and "Distinct specifications tested on this input" in t and "Bonferroni" in t
    assert "earlier runs on this input used a different specification" in t
    assert "For engineers" in t and "For auditors" in t and "Adds information:" in t
    ok, _ = verify(ev)
    assert ok
    (pkg / "results.json").write_text("{}")
    ok, msgs = verify(ev)
    assert not ok and any("MODIFIED" in m for m in msgs)
    lines = (ev / "run-ledger.jsonl").read_text().splitlines()
    (ev / "run-ledger.jsonl").write_text(lines[1] + "\n")
    ok, msgs = verify(ev)
    assert not ok and any("chain broken" in m for m in msgs)
    assert main(["verify-evidence", str(ev)]) == 1


def test_compare_side_by_side(tmp_path):
    f = _data(tmp_path)
    out = tmp_path / "c.md"
    main(["compare", str(f), "--old", "s2", "--new", "s", "--label", "y", "--group", "g", "--baselines", "task",
          "--families", "linear", "--jobs", "1", "--report", str(out)])
    t = out.read_text()
    assert "| Measure | v1 | v2 | Change |" in t and "Increment beyond the baselines" in t
    assert "Does v1 keep information that v2 lacks" in t
