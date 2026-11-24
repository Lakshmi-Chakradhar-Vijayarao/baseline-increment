"""Development diagnostic (not an acceptance run): is the excess false-flag rate caused by the null's
score model (s_hat) or by the outcome baseline model? Dev seeds (offset 900_000) are disjoint from every
acceptance run. Arms: current | oracle s_hat (true E[S|X]) | oracle baseline (true eta added as a column)."""
import json, sys, os, zlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import baseline_increment.core as core
from suite import gen as suite_gen
from stress_suite import gen as stress_gen, TASK

def make(name, rng):
    if name == "interaction_meta":
        df, base, kw = suite_gen(name, rng)
        sign = np.where(df.task == "A", 1.0, -1.0)
        df["f"] = df.L * sign; df["eta"] = 1.5 * df["f"]
    elif name == "large_copy":
        df, base = stress_gen(name, rng); kw = {}
        df["f"] = df.task.map(TASK) + 0.8 * df.L; df["eta"] = df["f"]
    else:
        df, base, kw = suite_gen(name, rng)
    return df, base, kw

def one(name, arm, rep):
    rng = np.random.default_rng(900_000 + 1000 * (zlib.crc32(name.encode()) % 97) + rep)
    df, base, kw = make(name, rng)
    orig = core.score_given_baselines
    if arm == "oracle_shat":
        f = df["f"].to_numpy()
        # the audit ranks S; E[rank(S)|X] is a monotone map of f -> use isotonic-free proxy: rank-scale f via its own mapping
        s_rank = core._rank01(df["s"].to_numpy())
        from sklearn.isotonic import IsotonicRegression
        oracle = IsotonicRegression().fit(f, s_rank).predict(f)   # E[rank S | f], f is the true sufficient statistic
        core.score_given_baselines = lambda *a, **k: oracle
    if arm == "super_learner":
        from shat_candidates import score_given_baselines_sl
        core.score_given_baselines = score_given_baselines_sl
    if arm == "oracle_base":
        base = base + ["eta"]
    try:
        r = core.audit(df, "s", "y", "g", base, (), seeds=range(2), n_perm=200, perm_stop=10, n_boot=0, n_boot_desc=20, **kw)
    finally:
        core.score_given_baselines = orig
    return r["adds_information"]

if __name__ == "__main__":
    from joblib import Parallel, delayed
    name, arm, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    res = Parallel(n_jobs=8)(delayed(one)(name, arm, r) for r in range(reps))
    print(json.dumps({"scenario": name, "arm": arm, "reps": reps, "flag_rate": sum(res) / reps}), flush=True)
