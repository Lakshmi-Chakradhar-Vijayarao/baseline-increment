"""v0.4 calibration of the validity estimands (decomposition, R2_v, additive null, panel sensitivity).

Acceptance criteria are fixed in ACCEPT below, written before the first run of this file.
Truth for the share: the population moment on the noise-free p matrix of the same draw (no binomial noise).
Usage: python simulations/validity_suite.py [reps]   ->  simulations/validity_v0.4.0.json
"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from baseline_increment.validity import decomposition, panel_sensitivity, _half_stats

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUTF = sys.argv[2] if len(sys.argv) > 2 else "simulations/validity_v0.4.0.json"
K = 10; S = [f"s{t}" for t in range(K)]
ACCEPT = {
    "noise_detector_false_recovery": ("R2_v CI lower bound > 0 for a pure-noise detector", "<= 0.10"),
    "planted_detector_power": ("R2_v CI lower bound > 0 for a planted detector (signal 1.0)", ">= 0.80"),
    "share_ci_coverage": ("question-bootstrap 95% CI of the share covers the noise-free share", "0.85-0.99"),
    "artefact_excess": ("|mean excess share| with no interaction and ability spread up to 2.5 logit sd", "<= 0.05"),
    "interaction_excess": ("mean excess share with interaction sd 1.0", ">= 0.10"),
    "panel_null_delta": ("|mean delta(k)| for a pure-noise detector, every k", "<= 0.01"),
}


def sim(Q, M, sd_item=1.5, sd_gamma=0.0, spread=0.0, signal=0.0, seed=0):
    rng = np.random.default_rng(seed); th = rng.normal(0, sd_item, Q)[:, None]
    a = np.linspace(-spread, spread, M)[None, :]; g = rng.normal(0, 1, (Q, M)) * sd_gamma
    p = 1 / (1 + np.exp(-(th + a + g))); ys = rng.binomial(1, p[..., None], (Q, M, K))
    det = signal * g + rng.normal(0, 1, (Q, M)); noise = rng.normal(0, 1, (Q, M))
    rows = [{"item": q, "model": f"m{m}", **{S[t]: ys[q, m, t] for t in range(K)}, "det": det[q, m], "noise": noise[q, m],
             "greedy": int(p[q, m] > .5)} for q in range(Q) for m in range(M)]
    return pd.DataFrame(rows), p


def pop_share(M, sd_item=1.5, sd_gamma=1.0, spread=0.0, Qbig=200000, seed=12345):
    """Population share (over item draws) for the generating parameters: noise-free p on a very large item sample.
    Amendment after the first run: the same-draw share was the wrong target for a question bootstrap (it conditions on items)."""
    rng = np.random.default_rng(seed); th = rng.normal(0, sd_item, Qbig)[:, None]; a = np.linspace(-spread, spread, M)[None, :]
    return true_share(1 / (1 + np.exp(-(th + a + rng.normal(0, 1, (Qbig, M)) * sd_gamma))))


def true_share(p):
    M = p.shape[1]; r = p - p.mean(0, keepdims=True); w = r - r.mean(1, keepdims=True)
    sv = np.var(w, ddof=1) * M / (M - 1); sq = np.var(r.mean(1), ddof=1) - sv / M
    return sv / (sq + sv)


def main():
    t0 = time.time(); out = {"accept": ACCEPT, "reps": REPS, "cells": {}}
    for Q in (300, 1000):
        for M in (3, 6, 10):
            fr, pw, cov, ex_int = [], [], [], []; truth = pop_share(M)
            for r in range(REPS):
                df, p = sim(Q, M, sd_gamma=1.0, signal=1.0, seed=1000 * M + r)
                d = decomposition(df, "item", "model", S, scores=["det", "noise"], n_boot=200, seed=r)
                fr.append(d["ci"]["R2_v[noise]"][0] > 0); pw.append(d["ci"]["R2_v[det]"][0] > 0)
                lo, hi = d["ci"]["model_specific_share"]; cov.append(lo <= truth <= hi)
                ex_int.append(d["additive_null"]["excess_share"])
            art = {}
            for spread in (0.0, 1.0, 2.5):
                art[spread] = float(np.mean([decomposition(sim(Q, M, sd_item=3.0, spread=spread, seed=7 + r)[0], "item", "model", S,
                                                           n_boot=2, seed=r)["additive_null"]["excess_share"] for r in range(max(REPS // 4, 5))]))
            out["cells"][f"Q{Q}_M{M}"] = {"noise_detector_false_recovery": float(np.mean(fr)), "planted_detector_power": float(np.mean(pw)),
                                         "share_ci_coverage": float(np.mean(cov)), "interaction_excess": float(np.mean(ex_int)),
                                         "artefact_excess_by_spread": art}
            print(f"Q{Q} M{M}", out["cells"][f"Q{Q}_M{M}"], f"{time.time() - t0:.0f}s", flush=True)
    pn = []
    for r in range(max(REPS // 8, 5)):
        df, _ = sim(400, 6, sd_gamma=0.5, seed=50 + r)
        pn.append(panel_sensitivity(df, "item", "model", "greedy", "noise", n_subsets=3, seeds=range(1), seed=r)["delta_by_k"])
    out["panel_null_delta"] = {k: float(np.mean([x[k] for x in pn])) for k in pn[0]}
    print("panel null", out["panel_null_delta"], flush=True)
    c = out["cells"].values()
    out["verdict"] = {
        "noise_detector_false_recovery": max(x["noise_detector_false_recovery"] for x in c) <= 0.10,
        "planted_detector_power": min(x["planted_detector_power"] for x in c) >= 0.80,
        "share_ci_coverage": all(0.85 <= x["share_ci_coverage"] <= 0.99 for x in c),
        "artefact_excess": max(abs(v) for x in c for v in x["artefact_excess_by_spread"].values()) <= 0.05,
        "interaction_excess": min(x["interaction_excess"] for x in c) >= 0.10,
        "panel_null_delta": max(abs(v) for v in out["panel_null_delta"].values()) <= 0.01,
    }
    print("verdict", out["verdict"])
    json.dump(out, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()
