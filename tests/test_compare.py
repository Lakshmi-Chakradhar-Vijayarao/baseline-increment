import numpy as np
import pandas as pd

from baseline_increment.compare import compare, holm2

FAST = dict(families=("linear",), seeds=range(1), n_perm=40, perm_stop=5, n_boot_diff=200)


def make(kind, n_groups=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        task = "ABC"[g % 3]
        for _ in range(2):
            y = int(rng.random() < (0.6 if task == "A" else 0.3))
            e1, e2 = rng.normal(size=2)
            if kind == "upgrade":                      # v1 = v2 + independent noise: v1 carries nothing beyond v2
                v2 = 1.2 * y + 0.5 * e1
                v1 = v2 + 1.5 * e2
            else:
                v1 = 1.2 * y + e1
                v2 = v1 if kind == "same" else v1 + 2 * e2
            rows.append({"g": g, "task": task, "model": "m" + str(g % 2), "y": y, "v1": v1, "v2": v2})
    return pd.DataFrame(rows)


def test_holm2():
    assert holm2(0.01, 0.04) == (0.02, 0.04)
    assert holm2(0.04, 0.01) == (0.04, 0.02)
    assert holm2(0.03, 0.02) == (0.04, 0.04)
    assert holm2(0.8, 0.6) == (1.0, 1.0)


def test_identical_scores_detect_nothing():
    r = compare(make("same"), "v1", "v2", "y", "g", ["task"], **FAST)
    assert not r["improvement"]["detected"] and not r["regression"]["detected"]
    assert r["standalone"]["difference"] == 0 and r["standalone"]["verdict"] == "difference not detected"
    assert "not detected (600 items, 300 groups)" in r["summary"][0]


def test_clear_upgrade_is_improvement_and_better_alone():
    r = compare(make("upgrade"), "v1", "v2", "y", "g", ["task"], **FAST)
    assert r["improvement"]["detected"] and not r["regression"]["detected"]
    assert r["standalone"]["verdict"] == "v2 better alone"


def test_noisier_v2_is_a_regression():
    r = compare(make("noise"), "v1", "v2", "y", "g", ["task"], **FAST)
    assert r["regression"]["detected"] and not r["improvement"]["detected"]
    assert r["standalone"]["verdict"] == "v2 worse alone"


def test_both_scores_get_the_same_rank_within_transform():
    df = make("same")
    df.loc[df.model == "m1", "v2"] = df.loc[df.model == "m1", "v1"] * 100 + 7   # per-model rescaling only
    r = compare(df, "v1", "v2", "y", "g", ["task"], rank_within="model", **FAST)
    assert r["standalone"]["difference"] == 0            # identical within-model ranks


def test_cli_fail_on_regression(tmp_path):
    from baseline_increment.cli import main
    f = tmp_path / "d.csv"
    make("noise").to_csv(f, index=False)
    args = [str(f), "--old", "v1", "--new", "v2", "--label", "y", "--group", "g", "--baselines", "task",
            "--families", "linear", "--seeds", "1", "--perms", "40", "--perm-stop", "5", "--jobs", "1"]
    assert main(["compare"] + args + ["--fail-on-regression"]) == 1
    make("upgrade").to_csv(f, index=False)
    assert main(["compare"] + args + ["--fail-on-regression"]) == 0
