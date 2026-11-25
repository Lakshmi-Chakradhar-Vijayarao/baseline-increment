"""The stable JSON keys of docs/API.md must stay present, and input errors must exit 2."""
import json

from baseline_increment.cli import main

STABLE = ["version", "n_items", "n_groups", "settings", "detector_auc", "baseline_auc", "families", "headline_family",
          "baselines_combined_auc", "baselines_plus_detector_auc", "increment", "increment_ci", "increment_perm_p",
          "adds_information", "warnings", "gain", "gain_ci", "incremental_prediction", "conditional_information",
          "statistical_preset"]
ARGS = ["examples/synthetic_ragstyle.csv", "--score", "detector_score", "--label", "hallucinated", "--group",
        "response_id", "--baselines", "task_type,generator,log_words", "--families", "linear", "--seeds", "1",
        "--perms", "20", "--boot", "0"]


def test_stable_json_keys(tmp_path):
    out = tmp_path / "a.json"
    assert main(ARGS + ["--json", str(out)]) == 0
    rep = json.loads(out.read_text())
    missing = [k for k in STABLE if k not in rep]
    assert not missing, missing


def test_input_errors_exit_2(tmp_path, capsys):
    assert main(ARGS[:4] + ["no_such_column"] + ARGS[5:]) == 2
    assert main(["no_such_file.csv", "--score", "s", "--label", "y"]) == 2
    assert "error:" in capsys.readouterr().err
