"""v0.6: question matching tolerant to a trailing '?', the protocol-bound HELM-NQ table, and own-panel
leave-one-model-out difficulty from harness logs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline_increment.adapters import add_lomo_difficulty, from_inspect
from baseline_increment.checks import _match_keys, check, load_benchmark, qhash
from baseline_increment.cli import main

FIX = Path(__file__).parent / "fixtures"
FAST = dict(n_boot=30, n_perm=99, seeds=2)


def test_match_keys_exact_wins_and_question_mark_fallback():
    keys = [qhash("who wrote hamlet"), qhash("what is the capital of france?")]
    got = _match_keys(["Who wrote Hamlet?", "What is the capital of France", "what is the capital of france?",
                       "unknown question?"], keys)
    assert list(got[:3]) == [keys[0], keys[1], keys[1]]
    assert got[3] == qhash("unknown question?")


def test_exact_key_always_preferred_when_both_variants_exist():
    keys = [qhash("is it raining?"), qhash("is it raining")]
    assert list(_match_keys(["Is it raining?", "Is it raining"], keys)) == keys


def test_helm_nq_table_and_protocol_warning():
    tab, meta = load_benchmark("helm_nq")
    assert len(tab) == 1000 and len(meta["families"]) == 41 and "protocol" in meta
    assert meta["within_protocol_reliability_curve"]["by_k"]["1"]["median"] < meta["within_protocol_reliability_curve"]["by_k"]["20"]["median"]
    df = pd.DataFrame({"qh": tab["qhash"].iloc[:300], "y": (tab["share_wrong"].iloc[:300] > 0.8).astype(int)})
    df["s"] = np.random.default_rng(0).normal(size=300)
    r = check(df, "s", "y", question="qh", benchmark="helm_nq", **FAST)
    assert r["benchmark"]["match_rate"] == 1.0 and r["difficulty_panel"]["n_families"] == 41
    assert any("fixed protocol" in w for w in r["warnings"])


def test_zero_match_is_an_error_not_an_empty_report():
    df = pd.DataFrame({"t": ["not a real question %d" % i for i in range(50)], "y": [0, 1] * 25, "s": range(50)})
    with pytest.raises(ValueError, match="no rows matched"):
        check(df, "s", "y", question_text="t", benchmark="helm_nq", **FAST)


def test_lomo_difficulty_is_other_models_error():
    df = pd.DataFrame({"item": ["a", "a", "a", "b", "b", "c"], "model": ["m1", "m2", "m3", "m1", "m2", "m1"],
                       "wrong": [1, 0, 1, 0, 1, 1]})
    out, meta = add_lomo_difficulty(df)
    got = {(r.item, r.model): (r.lomo_share_wrong, r.n_ref_models) for r in out.itertuples()}
    assert got[("a", "m1")] == (0.5, 2) and got[("a", "m2")] == (1.0, 2) and got[("b", "m2")] == (0.0, 1)
    assert ("c", "m1") not in got and meta["dropped_rows"] == 1 and meta["n_models"] == 2
    assert (out["group"] == out["item"]).all()


def test_lomo_averages_epochs_within_model_first():
    df = pd.DataFrame({"item": ["a"] * 4, "model": ["m1", "m2", "m2", "m2"], "wrong": [0, 1, 0, 0]})
    out, _ = add_lomo_difficulty(df)
    assert out.loc[out.model == "m1", "lomo_share_wrong"].iloc[0] == pytest.approx(1 / 3)


def test_cli_from_inspect_lomo(tmp_path, capsys):
    d = json.loads((FIX / "arith_mockllm.json").read_text())
    paths = []
    for i, name in enumerate(["mockllm/a", "mockllm/b", "mockllm/c"]):
        e = json.loads(json.dumps(d)); e["eval"]["model"] = name
        p = tmp_path / f"log{i}.json"; p.write_text(json.dumps(e)); paths.append(str(p))
    out = tmp_path / "items.csv"
    assert main(["from-inspect", *paths, "-o", str(out), "--label-scorer", "match", "--score-scorer", "includes", "--lomo"]) == 0
    df = pd.read_csv(out); meta = json.loads(Path(str(out) + ".panel.json").read_text())
    assert meta["n_models"] == 2 and len(meta["models"]) == 3 and "lomo_share_wrong" in df
    assert df["item"].nunique() == len(d["samples"]) and (df["n_ref_models"] == 2).all()
    # identical logs: each model's own error equals the others' error on that item
    assert (df["lomo_share_wrong"] == df.groupby(["item", "model"])["wrong"].transform("mean")).all()
    assert "own panel: 3 models" in capsys.readouterr().out


def test_check_accepts_own_panel_difficulty():
    rng = np.random.default_rng(3); d = rng.normal(size=300); rows = []
    for m, ab in enumerate([-1, -0.3, 0.3, 1, 1.5, 2]):
        for q in range(300):
            y = int(rng.random() < 1 / (1 + np.exp(-(1.5 * d[q] - ab))))
            rows.append({"item": f"t:{q}", "model": f"m{m}", "wrong": y, "score": d[q] + rng.normal()})
    out, meta = add_lomo_difficulty(pd.DataFrame(rows))
    r = check(out, "score", "wrong", question="item", model="model", difficulty=["lomo_share_wrong"], panel_meta=meta, **FAST)
    assert r["difficulty_panel"]["source"] == "own logs, leave-one-model-out" and r["difficulty_panel"]["n_models"] == 5
    # The score tracks TRUE difficulty, so it adds something beyond any finite panel's estimate; that excess must
    # shrink as the panel grows (the panel-size effect the tool warns about).
    df = pd.DataFrame(rows)
    small, small_meta = add_lomo_difficulty(df[df.model.isin(["m2", "m3"])])
    r1 = check(small, "score", "wrong", question="item", model="model", difficulty=["lomo_share_wrong"], panel_meta=small_meta, **FAST)
    inc = lambda rep: rep["checks"]["your difficulty columns"]["increment"]
    assert inc(r1) > inc(r) + 0.02
    assert any("only 1 reference models" in w for w in r1["warnings"])
