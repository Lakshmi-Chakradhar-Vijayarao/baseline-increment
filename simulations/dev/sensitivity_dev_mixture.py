"""Development run disclosed in PREREG_0.13 addendum 1 (not an acceptance run)."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/Users/chakrivijayarao/Desktop/baseline-increment")
from baseline_increment.sensitivity import sensitivity, sentence
r = np.random.default_rng(0); n = 800
L = r.normal(size=n); Z = r.normal(size=n); task = r.choice(list("ABC"), n)
y = (r.random(n) < 1/(1+np.exp(-(0.8*L + 1.0*Z)))).astype(int)
for name, s in [("S carries Z only", Z + r.normal(size=n)), ("S independent signal", r.normal(size=n) + 1.0*y)]:
    df = pd.DataFrame({"g": np.arange(n), "L": L, "task": task, "y": y, "s": s})
    t = time.time()
    out = sensitivity(df, "s", "y", "g", ["L", "task"], families=("linear",), seeds=range(1), grid=(0.3, 0.7))
    print(name, round(time.time()-t,1), "s; RV", out["robustness_value"], out["robustness_note"], "bench", {k:{a:round(b,2) for a,b in v.items()} for k,v in out["benchmarks"].items()})
    print("  ", sentence(out))
    # actual Z strengths
    from baseline_increment.sensitivity import _residuals, _multiple_corr
    from baseline_increment.core import split_design, transform_score, score_given_baselines
    num, lin = split_design(df, ["L","task"]); ss = transform_score(df,"s","rank")
    rs, ry = _residuals("linear", num, lin, ss, y, df.g.astype(str).values, "gbm")
    zres = Z - score_given_baselines(num, lin, Z, df.g.astype(str).values, 0, "gbm")
    print("   Z strengths alpha %.2f beta %.2f" % (_multiple_corr(rs, zres), _multiple_corr(ry, zres)))
