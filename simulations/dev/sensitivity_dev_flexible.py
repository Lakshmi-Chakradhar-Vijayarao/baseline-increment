"""Development run disclosed in PREREG_0.13 addendum 2 (not an acceptance run; dev seed salt 99)."""
import sys, os, numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "..")]
from sensitivity_suite import gen
from baseline_increment import audit
from baseline_increment.core import split_design, transform_score
from baseline_increment.sensitivity import residuals, strengths, strengths_flexible
from joblib import Parallel, delayed
import zlib

def run(fam, regime, rep):
    rng = np.random.default_rng(990_000 + 1000 * (zlib.crc32(f"{fam}{regime}".encode()) % 97) + rep)
    df = gen(fam, regime, 600, rng)
    kw = dict(seeds=range(2), n_perm=200, perm_stop=10, n_boot=0, n_boot_desc=20, null="residual_sl")
    a0 = audit(df, "s", "y", "g", ["L", "task"], **kw)
    if not a0["adds_information"]:
        return None
    g = df.g.astype(str).to_numpy(); y = df.y.to_numpy(); s = transform_score(df, "s", "rank")
    num, lin = split_design(df, ["L", "task"])
    rs, ry = residuals(a0["headline_family"], num, lin, s, y, g, "super_learner")
    r = float(np.corrcoef(rs, ry)[0, 1])
    nz, lz = split_design(df, ["Z"]); na, la = split_design(df, ["L", "task", "Z"])
    al, be = strengths(rs, ry, num, lin, np.hstack([nz, lz]), g, "super_learner")
    af, bf = strengths_flexible(rs, ry, na, la, g, "super_learner")
    a1 = audit(df, "s", "y", "g", ["L", "task", "Z"], **kw)
    return fam, regime, al * be / r, af * bf / r, a1["increment"], a0["increment"]

res = [x for x in Parallel(n_jobs=8)(delayed(run)(f, rg, k) for f in ["linear", "nonlinear", "categorical", "interaction"]
                                      for rg in ["none", "half", "all"] for k in range(6)) if x]
for f in ["linear", "nonlinear", "categorical", "interaction"]:
    for rg in ["none", "half", "all"]:
        rows = [x for x in res if x[0] == f and x[1] == rg]
        if rows:
            print(f"{f:12s} {rg:5s} n={len(rows)} phi_lin={np.mean([x[2] for x in rows]):.2f} phi_flex={np.mean([x[3] for x in rows]):.2f} "
                  f"erased={sum(x[4] <= 0 for x in rows)}  rel_drop={np.mean([1 - x[4] / x[5] for x in rows]):.2f}")
