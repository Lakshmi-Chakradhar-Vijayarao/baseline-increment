import hashlib
import json

import numpy as np
import pandas as pd

from baseline_increment.cli import main
from baseline_increment.report import calibration_for_audit


def _data(tmp_path, n=300, seed=0, bad_name=False):
    rng = np.random.default_rng(seed)
    task = np.array(list("AB") * (n // 2))
    y = (rng.random(n) < np.where(task == "A", 0.6, 0.25)).astype(int)
    col = "<script>alert(1)</script>" if bad_name else "task"
    df = pd.DataFrame({"g": np.arange(n), col: task, "y": y, "s": 1.5 * y + rng.normal(size=n)})
    df["s2"] = df["s"] + rng.normal(size=n)
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    return f, col


ARGS = ["--score", "s", "--label", "y", "--group", "g", "--fast", "--families", "linear", "--jobs", "1"]
FULL = ["--score", "s", "--label", "y", "--group", "g", "--fast", "--jobs", "1"]          # calibrated families


def test_audit_report_html_and_md(tmp_path):
    f, col = _data(tmp_path)
    import html as _h
    for ext in ("html", "md"):
        out = tmp_path / f"r.{ext}"
        main([str(f), *FULL, "--baselines", col, "--report", str(out)])
        t = _h.unescape(out.read_text())
        assert "PASS:" in t and "Secondary, experimental" in t
        assert hashlib.sha256(f.read_bytes()).hexdigest()[:12] in t
        assert "match the validated default preset (calibration run 'fast') exactly" in t
        assert "Not tested by this report" in t and "false-positive rate" in t
    main([str(f), *ARGS, "--baselines", col, "--report", str(tmp_path / "lin.md")])
    assert "do not match a calibrated run" in (tmp_path / "lin.md").read_text()


def test_report_escapes_data_derived_text(tmp_path):
    f, col = _data(tmp_path, bad_name=True)
    out = tmp_path / "r.html"
    main([str(f), *ARGS, "--baselines", col, "--strata", col, "--report", str(out)])
    t = out.read_text()
    assert "<script>" not in t and "&lt;script&gt;" in t


def test_uncalibrated_settings_are_flagged():
    F = ["linear", "spline", "gbm"]
    c = calibration_for_audit({"n_seeds": 5, "n_perm": 200, "perm_stop": None, "null": "residual", "risk_bins": 0,
                               "score_transform": "rank", "families": F})
    assert not c["matched"] and "do not match a calibrated run" in c["status"]
    c = calibration_for_audit({"n_seeds": 2, "n_perm": 100, "perm_stop": None, "null": "residual", "risk_bins": 0,
                               "score_transform": "rank", "families": F})
    assert c["matched"] and "'suite'" in c["status"]


def test_compare_report(tmp_path):
    f, col = _data(tmp_path)
    out = tmp_path / "c.md"
    main(["compare", str(f), "--old", "s", "--new", "s2", "--label", "y", "--group", "g", "--baselines", col,
          "--families", "linear", "--seeds", "1", "--perms", "40", "--perm-stop", "5", "--jobs", "1", "--report", str(out)])
    t = out.read_text()
    assert "Detector release comparison" in t and "v1 alone catches" in t
    assert "differ from the calibrated compare run" in t
    assert "does not mean the effect is absent" in t


def test_mapping_file_hash_in_provenance(tmp_path):
    f, col = _data(tmp_path)
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"columns": {"score2": "s"}}))
    out = tmp_path / "r.md"
    main([str(f), *ARGS, "--baselines", col, "--map", str(m), "--report", str(out)])
    assert hashlib.sha256(m.read_bytes()).hexdigest() in out.read_text()


def test_command_paths_are_reduced_to_file_names(tmp_path):
    f, col = _data(tmp_path)
    out = tmp_path / "r.md"
    main([str(f), *ARGS, "--baselines", col, "--report", str(out)])
    t = out.read_text()
    assert str(tmp_path) not in t and "d.csv" in t
