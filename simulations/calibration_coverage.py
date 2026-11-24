"""Coverage of the raw-score calibration intervals (acceptance fixed in docs/PREREG_0.11.md before the first run).

    python3 simulations/calibration_coverage.py --reps 200 --jobs 8 --out simulations/calcov_v0.11.0.json
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment.calibration import score_calibration  # noqa: E402


def gen(rep, squared):
    rng = np.random.default_rng(40_000 + rep)
    p = np.repeat(rng.beta(2, 3, 300), 2)                     # two items per group share p
    y = (rng.random(600) < p).astype(int)
    return pd.DataFrame({"g": np.repeat(np.arange(300), 2), "y": y, "p": p ** 2 if squared else p})


def one(rep, squared):
    c = score_calibration(gen(rep, squared), "p", "y", "g", n_boot=2000, seed=rep)
    return {"slope_ci": c["slope_ci"], "citl_ci": c["citl_ci"], "slope": c["slope"], "citl": c["citl"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    t0 = time.time()
    cal = Parallel(n_jobs=a.jobs)(delayed(one)(r, False) for r in range(a.reps))
    mis = Parallel(n_jobs=a.jobs)(delayed(one)(r, True) for r in range(a.reps))
    cs = float(np.mean([x["slope_ci"][0] <= 1 <= x["slope_ci"][1] for x in cal]))
    cc = float(np.mean([x["citl_ci"][0] <= 0 <= x["citl_ci"][1] for x in cal]))
    out = {"reps": a.reps, "design": "300 groups x 2 items sharing p ~ Beta(2,3); y ~ Bernoulli(p); 2000 group-bootstrap draws",
           "calibrated": {"slope_coverage_of_1": cs, "citl_coverage_of_0": cc,
                          "mean_slope": float(np.mean([x["slope"] for x in cal])), "mean_citl": float(np.mean([x["citl"] for x in cal]))},
           "miscalibrated_p_squared": {"slope_ci_excludes_1": float(np.mean([not (x["slope_ci"][0] <= 1 <= x["slope_ci"][1]) for x in mis])),
                                       "mean_slope": float(np.mean([x["slope"] for x in mis])), "mean_citl": float(np.mean([x["citl"] for x in mis]))},
           "pass": bool(0.85 <= cs <= 0.99 and 0.85 <= cc <= 0.99), "sec": round(time.time() - t0)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
