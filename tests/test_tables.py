import json

import numpy as np
import pandas as pd
import pytest

from baseline_increment.tables import read_table


def _records(n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = int(rng.random() < 0.4)
        out.append({"id": i, "metadata": {"doc": f"d{i // 2}", "task": "qa" if i % 2 else "sum"},
                    "scores": {"judge": {"value": float(y + rng.normal())}},
                    "labels": {"verdict": "fail" if y else "pass"}, "output": "word " * int(rng.integers(1, 30))})
    return out


MAP = {"columns": {"score": "scores.judge.value", "group": "metadata.doc", "task": "metadata.task"},
       "derive": {"wrong": {"from": "labels.verdict", "map": {"pass": 0, "fail": 1}}}}


def test_jsonl_json_csv_parquet_agree(tmp_path):
    recs = _records()
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (tmp_path / "a.json").write_text(json.dumps({"results": recs}))
    base = read_table(tmp_path / "a.jsonl", MAP)
    assert list(base.columns[-4:]) == ["score", "group", "task", "wrong"]
    pd.testing.assert_frame_equal(base, read_table(tmp_path / "a.json", MAP))
    base.to_csv(tmp_path / "a.csv", index=False)
    assert np.allclose(read_table(tmp_path / "a.csv")["score"], base["score"])
    pytest.importorskip("pyarrow")
    base.to_parquet(tmp_path / "a.parquet", index=False)
    pd.testing.assert_frame_equal(read_table(tmp_path / "a.parquet"), base)


def test_unmapped_values_and_missing_columns_raise(tmp_path):
    recs = _records()
    recs[3]["labels"]["verdict"] = "unsure"
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    with pytest.raises(ValueError, match="without a mapping"):
        read_table(tmp_path / "a.jsonl", MAP)
    with pytest.raises(ValueError, match="not found"):
        read_table(tmp_path / "a.jsonl", {"columns": {"score": "scores.nope"}})


def test_cli_audits_a_mapped_jsonl_export(tmp_path):
    from baseline_increment.cli import main
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in _records(300)))
    (tmp_path / "map.json").write_text(json.dumps(MAP))
    assert main([str(tmp_path / "a.jsonl"), "--map", str(tmp_path / "map.json"), "--score", "score", "--label", "wrong",
                 "--group", "group", "--baselines", "task", "--fast", "--families", "linear", "--fail-if-no-increment"]) == 0
