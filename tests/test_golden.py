"""The binary audit engine must reproduce this output exactly. Every calibration number the reports quote was measured
with this engine; any change to it must be deliberate and re-calibrated (fixture made at 0.10.0, commit 5711f9a)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_increment import audit

FIX = Path(__file__).parent / "fixtures" / "golden_audit_synthetic.json"
EX = Path(__file__).parents[1] / "examples" / "synthetic_ragstyle.csv"


def _close(a, b, path="rep"):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ {set(a) ^ set(b)}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert np.isclose(a, b, rtol=0, atol=1e-12), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_binary_engine_is_unchanged():
    df = pd.read_csv(EX)
    rep = audit(df, "detector_score", "hallucinated", "response_id", ["task_type", "generator", "log_words"],
                strata=["task_type"], seeds=range(1), n_perm=20, n_boot=5, n_boot_desc=50, perm_stop=None, interval="bootstrap")
    rep.pop("version")
    _close(json.loads(json.dumps(rep, sort_keys=True, default=float)), json.load(open(FIX)))
