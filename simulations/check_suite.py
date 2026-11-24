"""Battery for `check` (v0.3): does each check refuse the trap it is meant to catch, and find a real signal?

Data: 300 questions x 4 models; P(wrong) = sigmoid(1.5*d_q - ability_m) unless stated. Pass criteria:
  pure_difficulty   score = d + noise                 -> difficulty SUPPORTED rate <= 0.10
  duplicated        pure_difficulty, every row duplicated (same question id) -> difficulty SUPPORTED <= 0.10
  nonlinear         P(wrong) depends on d^2; score = d^2 + noise -> difficulty SUPPORTED <= 0.10
  pure_length       P(wrong) depends on length; score = length + noise -> length and "all" SUPPORTED <= 0.10
  pure_model        score = -ability_m (+ tiny noise)  -> same-question and model-identity SUPPORTED <= 0.10
  planted           score = d + 0.8*y + noise          -> difficulty SUPPORTED >= 0.80 and same-question >= 0.80
  planted_noisy     planted, 15% of labels flipped     -> power reported only
Usage: python3 simulations/check_suite.py --reps 40 --jobs 6 --out simulations/results/check_v0.3.0.json"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment.checks import check  # noqa: E402

SETTINGS = dict(n_boot=50, n_perm=99, seeds=2)


def data(kind, seed, n_q=300, models=4):
    rng = np.random.default_rng(seed); d = rng.normal(size=n_q); ab = np.linspace(-1, 1, models); rows = []
    for m in range(models):
        for q in range(n_q):
            L = rng.normal()
            eta = {"nonlinear": 1.5 * (d[q] ** 2 - 1) - ab[m], "pure_length": 1.5 * L - ab[m]}.get(kind, 1.5 * d[q] - ab[m])
            y = int(rng.random() < 1 / (1 + np.exp(-eta)))
            s = {"nonlinear": d[q] ** 2, "pure_length": L, "pure_model": -ab[m]}.get(kind, d[q])
            s = s + (0.01 if kind == "pure_model" else 1.0) * rng.normal() + (0.8 * y if kind.startswith("planted") else 0)
            if kind == "planted_noisy" and rng.random() < 0.15:
                y = 1 - y
            rows.append({"q": q, "model": f"m{m}", "difficulty": d[q], "length": L, "y": y, "s": s})
    df = pd.DataFrame(rows)
    return pd.concat([df, df], ignore_index=True) if kind == "duplicated" else df


def one(kind, seed):
    df = data(kind, seed)
    r = check(df, "s", "y", question="q", model="model", length="length", difficulty=["difficulty"], **SETTINGS)
    return {k: v["status"] for k, v in r["checks"].items()}


CRIT = {"pure_difficulty": [("your difficulty columns", "<=", .10)],
        "duplicated": [("your difficulty columns", "<=", .10)],
        "nonlinear": [("your difficulty columns", "<=", .10)],
        "pure_length": [("response length", "<=", .10), ("all of the above together", "<=", .10)],
        "pure_model": [("same question, across models", "<=", .10), ("model identity", "<=", .10)],
        "planted": [("your difficulty columns", ">=", .80), ("same question, across models", ">=", .80)],
        "planted_noisy": []}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=40); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only"); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = {"version": __import__("baseline_increment").__version__, "reps": a.reps, "settings": SETTINGS, "scenarios": {}}
    for kind in (a.only.split(",") if a.only else CRIT):
        t = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(one)(kind, 7000 + r) for r in range(a.reps))
        rates = {c: float(np.mean([x.get(c) == "SUPPORTED" for x in res])) for c in res[0]}
        ok = all((rates[c] <= v) if op == "<=" else (rates[c] >= v) for c, op, v in CRIT[kind])
        out["scenarios"][kind] = {"supported_rate": rates, "criteria": CRIT[kind], "pass": ok, "sec": round(time.time() - t)}
        print(kind, json.dumps(out["scenarios"][kind]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    out["all_pass"] = all(v["pass"] for v in out["scenarios"].values())
    json.dump(out, open(a.out, "w"), indent=1)
    print("ALL PASS" if out["all_pass"] else "SOME FAIL")
