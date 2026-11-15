"""Known-answer calibration tests for the v0.4 estimands (they must pass before any card uses them)."""
import numpy as np
import pandas as pd

from baseline_increment.validity import decomposition, incremental_dprime, panel_sensitivity


def _sim(Q=600, M=6, K=10, sd_item=1.5, sd_gamma=0.0, signal=0.0, seed=1):
    """logit p_mq = theta_q + gamma_mq; a detector = signal * gamma + noise; K sampled labels."""
    rng = np.random.default_rng(seed); th = rng.normal(0, sd_item, Q)[:, None]; g = rng.normal(0, 1, (Q, M)) * sd_gamma
    p = 1 / (1 + np.exp(-(th + g))); rows = []
    det = signal * g + rng.normal(0, 1, (Q, M))
    for q in range(Q):
        for m in range(M):
            ys = rng.binomial(1, p[q, m], K)
            rows.append({"item": q, "model": f"m{m}", **{f"s{t}": ys[t] for t in range(K)}, "det": det[q, m],
                         "noise": rng.normal(), "greedy": int(p[q, m] > .5)})
    return pd.DataFrame(rows)


SAMPLES = [f"s{t}" for t in range(10)]


def test_null_no_model_specific_part():
    r = decomposition(_sim(sd_gamma=0.0), "item", "model", SAMPLES, scores=["noise"], n_boot=50)
    assert abs(r["estimate"]["model_specific_share"]) < 0.05
    assert not r["r2_v_defined"]                              # nothing model-specific: R2_v must be refused


def test_model_specific_part_detected_and_repeatable():
    r = decomposition(_sim(sd_gamma=1.0), "item", "model", SAMPLES, scores=["noise"], n_boot=50)
    assert r["estimate"]["model_specific_share"] > 0.15
    assert r["estimate"]["split_half_r"] > 0.3                # 5 samples/half: binomial noise caps r well below 1
    r2 = decomposition(_sim(sd_gamma=2.0), "item", "model", SAMPLES, n_boot=10)
    assert r2["estimate"]["split_half_r"] > r["estimate"]["split_half_r"]   # more model-specific spread -> more repeatable
    assert r["r2_v_defined"] and abs(r["estimate"]["R2_v[noise]"]) < 0.03     # a pure-noise detector explains ~nothing


def test_planted_detector_recovered():
    r = decomposition(_sim(sd_gamma=1.0, signal=3.0), "item", "model", SAMPLES, scores=["det", "noise"], n_boot=50)
    assert r["estimate"]["R2_v[det]"] > 0.3
    assert r["ci"]["R2_v[det]"][0] > r["estimate"]["R2_v[noise]"]


def test_dprime_known_values():
    d = incremental_dprime(0.855, 0.889)
    assert 1.45 < d["d_base"] < 1.55 and 0.8 < d["d_add"] < 0.92 and 0.70 < d["standalone_equivalent_auc"] < 0.76
    assert incremental_dprime(0.8, 0.8)["d_add"] == 0.0


def test_panel_sensitivity_shrinks_with_k():
    """A detector that tracks item difficulty: its increment over a 1-model D1 is large and shrinks as k grows."""
    df = _sim(Q=500, M=6, sd_item=1.5, sd_gamma=0.3, seed=3)
    th_proxy = df.groupby("item")[SAMPLES].mean().mean(1)
    df["diff_det"] = df["item"].map(th_proxy) + np.random.default_rng(0).normal(0, .05, len(df))
    r = panel_sensitivity(df, "item", "model", "greedy", "diff_det", n_subsets=3, seeds=range(1))
    assert r["delta_by_k"][1] > r["delta_all_others"] > -0.01
    assert r["sensitive"]


def test_panel_sensitivity_not_checkable_with_two_models():
    df = _sim(Q=100, M=2)
    assert panel_sensitivity(df, "item", "model", "greedy", "det")["status"] == "NOT CHECKABLE"


def test_additive_null_absorbs_ability_artefact():
    """No interaction but very different model abilities: the raw share is inflated; the null explains it."""
    rng = np.random.default_rng(5); Q, M, K = 800, 6, 10
    th = rng.normal(0, 3, Q)[:, None]; a = np.linspace(-2.5, 2.5, M)[None, :]; p = 1 / (1 + np.exp(-(th + a)))
    rows = [{"item": q, "model": f"m{m}", **{f"s{t}": v for t, v in enumerate(rng.binomial(1, p[q, m], K))}}
            for q in range(Q) for m in range(M)]
    r = decomposition(pd.DataFrame(rows), "item", "model", SAMPLES, n_boot=10)
    assert r["estimate"]["model_specific_share"] > 0.08            # the artefact is real
    assert abs(r["additive_null"]["excess_share"]) < 0.06           # and the null removes it


def test_additive_null_leaves_real_interaction():
    r = decomposition(_sim(sd_gamma=1.0), "item", "model", SAMPLES, n_boot=10)
    assert r["additive_null"]["excess_share"] > 0.12
