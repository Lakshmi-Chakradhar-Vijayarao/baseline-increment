import numpy as np
import pandas as pd

from baseline_increment.checks import check, load_benchmark, qhash, same_question, to_html, to_text

FAST = dict(n_boot=30, n_perm=99, seeds=2)


def multi_model(n_q=400, models=4, item_signal=0.0, model_only=False, seed=0):
    """Several models answer the same questions. Error depends on question difficulty and model ability."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=n_q); ability = np.linspace(-1, 1, models)
    rows = []
    for m in range(models):
        for q in range(n_q):
            y = int(rng.random() < 1 / (1 + np.exp(-(1.5 * d[q] - ability[m]))))
            if model_only:
                s = -ability[m] + 0.01 * rng.normal()
            else:
                s = d[q] + item_signal * y + rng.normal()
            rows.append({"q": q, "model": f"m{m}", "difficulty": d[q], "y": y, "s": s,
                         "resp": " ".join(["w"] * int(rng.integers(1, 6)))})
    return pd.DataFrame(rows)


def test_pure_difficulty_detector_is_not_credited():
    df = multi_model(item_signal=0.0)
    r = check(df, "s", "y", question="q", difficulty=["difficulty"], **FAST)
    assert r["checks"]["your difficulty columns"]["status"] != "PASS"


def test_planted_signal_is_found():
    df = multi_model(item_signal=1.5)
    r = check(df, "s", "y", question="q", model="model", difficulty=["difficulty"], **FAST)
    assert r["checks"]["your difficulty columns"]["status"] == "PASS"
    assert r["checks"]["same question, across models"]["status"] == "PASS"


def test_model_identity_score_fails_same_question():
    df = multi_model(model_only=True)
    sq = same_question(df, "s", "y", "q", "model", n_boot=30)
    assert sq["status"] != "PASS" and abs(sq["pair_auc"] - 0.5) < 0.05


def test_not_checkable_listed_and_rendering():
    df = multi_model(item_signal=1.0)
    r = check(df, "s", "y", question="q", response="resp", **FAST)
    assert any("Question difficulty" in x for x in r["not_checkable"])
    assert "response length" in r["checks"]
    assert "Not tested / not checkable" in to_text(r) and "<table>" in to_html(r)
    # v0.4: untested rungs are NOT TESTED, never a pass; the manifest records the environment
    lad = {x["rung"]: x["status"] for x in r["ladder"]}
    assert lad["Out-of-distribution generalisation"] == "NOT TESTED"
    assert lad["Independent replication"] == "NOT TESTED"
    assert r["manifest"]["packages"]["numpy"]


def test_user_difficulty_without_panel_meta_is_never_headline():
    df = multi_model(item_signal=1.0)
    df["mydiff"] = np.random.default_rng(1).normal(size=len(df))
    r = check(df, "s", "y", question="q", difficulty=["mydiff"], **FAST)
    assert r["headline"]["status"] == "NO HEADLINE"
    assert any("panel metadata" in w for w in r["warnings"])
    assert not r["manifest"]["complete"]
    r2 = check(df, "s", "y", question="q", difficulty=["mydiff"], panel_meta={"models": ["a", "b", "c"], "n_models": 3}, **FAST)
    assert r2["headline"]["metric"].startswith("AUROC increment over difficulty from a 3-model panel")
    assert r2["manifest"]["complete"]
    assert any("only 3 reference models" in w for w in r2["warnings"])
    r3 = check(df, "s", "y", question="q", difficulty=["mydiff"], panel_meta={"models": list("abcdef"), "n_models": 6}, **FAST)
    assert not any("reference models: increments" in w for w in r3["warnings"])


def test_benchmark_join_by_question_text():
    tab, meta = load_benchmark("sciq")
    assert len(tab) == 2000 and meta["n_questions"] == 2000
    assert qhash("  What  IS this? ") == qhash("what is this?")
    df = pd.DataFrame({"qh": tab["qhash"].iloc[:300], "y": (tab["share_wrong"].iloc[:300] > 0.4).astype(int)})
    df["s"] = np.random.default_rng(0).normal(size=300)
    r = check(df, "s", "y", question="qh", benchmark="sciq", **FAST)
    assert r["benchmark"]["match_rate"] == 1.0
    assert r["checks"]["benchmark difficulty (other models' error rate)"]["status"] != "PASS"


def test_ragtruth_leave_one_generator_out():
    tab, meta = load_benchmark("ragtruth")
    assert "llama-2-7b-chat" in meta["models"]
    s = tab[tab["generator"] == "llama-2-7b-chat"].head(200)
    df = pd.DataFrame({"src": s["source_id"], "gen": "llama-2-7b-chat",
                       "y": (np.random.default_rng(1).random(len(s)) < .4).astype(int), "s": np.random.default_rng(2).normal(size=len(s))})
    r = check(df, "s", "y", question="src", model="gen", benchmark="ragtruth", **FAST)
    assert r["benchmark"]["match_rate"] == 1.0 and r["benchmark"]["leave_one_generator_out_rows"] == len(s)


def test_module_and_function_names_do_not_clash():
    import baseline_increment, baseline_increment.checks as m
    assert callable(baseline_increment.check) and hasattr(m, "load_benchmark")


def test_cli_fail_if_no_increment_exit_code(tmp_path):
    import numpy as np
    import pandas as pd
    from baseline_increment.cli import main
    rng = np.random.default_rng(0)
    n = 300
    task = np.array(list("AB") * (n // 2))
    y = (rng.random(n) < np.where(task == "A", 0.7, 0.2)).astype(int)
    df = pd.DataFrame({"g": np.arange(n), "task": task, "y": y, "s": (task == "A") + rng.normal(size=n)})
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    assert main([str(f), "--score", "s", "--label", "y", "--group", "g", "--baselines", "task", "--fast",
                 "--families", "linear", "--fail-if-no-increment"]) == 1
    df["s"] = df["y"] * 2 + rng.normal(size=n)
    df.to_csv(f, index=False)
    assert main([str(f), "--score", "s", "--label", "y", "--group", "g", "--baselines", "task", "--fast",
                 "--families", "linear", "--fail-if-no-increment"]) == 0
