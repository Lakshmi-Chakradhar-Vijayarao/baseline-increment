"""Development run disclosed in PREREG_0.13 addendum 1 (not an acceptance run)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/chakrivijayarao/Desktop/baseline-increment")
from baseline_increment.sensitivity import _residuals, _multiple_corr, _z
from baseline_increment.core import split_design, transform_score, score_given_baselines, estimate
from baseline_increment import audit
for seed in range(3):
  r = np.random.default_rng(100+seed); n = 1200
  L = r.normal(size=n); Z = r.normal(size=n); task = r.choice(list("ABC"), n)
  y = (r.random(n) < 1/(1+np.exp(-(0.8*L + 1.0*Z)))).astype(int)
  for name, s in [("all via Z", Z + r.normal(size=n)), ("half", 0.5*Z + 0.7*y + r.normal(size=n)), ("none", r.normal(size=n) + 1.0*y)]:
    df = pd.DataFrame({"g": np.arange(n), "L": L, "task": task, "y": y, "s": s, "Z": Z})
    g = df.g.astype(str).values
    num, lin = split_design(df, ["L","task"]); ss = transform_score(df,"s","rank")
    rs, ry = _residuals("linear", num, lin, ss, y, g, "gbm")
    rr = float(np.mean(_z(rs)*_z(ry)))
    zres = Z - score_given_baselines(num, lin, Z, g, 0, "gbm")
    a, b = _multiple_corr(rs, zres), _multiple_corr(ry, zres)
    kw = dict(families=["linear"], seeds=range(1), n_perm=50, perm_stop=10, n_boot=0, n_boot_desc=10)
    i0 = audit(df,"s","y","g",["L","task"],**kw)["increment"]; i1 = audit(df,"s","y","g",["L","task","Z"],**kw)
    print(f"{seed} {name:10s} r={rr:.3f} sqrt_r={np.sqrt(max(rr,0)):.2f} aZ={a:.2f} bZ={b:.2f} ab={a*b:.3f} ab/r={a*b/rr:.2f} | inc w/o Z {i0:+.3f} with Z {i1['increment']:+.3f} adds={i1['adds_information']}")
