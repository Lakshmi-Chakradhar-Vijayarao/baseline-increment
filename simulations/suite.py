"""Calibration suite for baseline-increment (permanent regression test; acceptance rules fixed below,
before the first run of v0.2 on 8 Nov 2026).

    python3 simulations/suite.py --reps 100 --jobs 8 --out simulations/results/results_v0.2.0.json
    python3 simulations/suite.py --coverage --reps 40 --jobs 8 --out simulations/results/coverage_v0.2.0.json

Each scenario generates a fresh dataset per replicate. The suite records:
- the rejection rate of the permutation test at 0.05;
- the mean increment;
- the headline family chosen.

For the null scenarios the detector carries no information about the label beyond the baseline
columns (in the adversarial ones, possibly through a nonlinear or interaction function the linear
baseline cannot express). The positive scenarios plant real signal.

Acceptance (pre-specified):
  null and adversarial-null scenarios  rejection rate <= 0.10   (nominal 0.05; 100 reps)
  moderate, strong, signal_with_difficulty    rejection rate >= 0.80
  hard_only_signal                            rejection rate >= 0.50
  weak_signal                                 reported, no threshold
  coverage run: 95% CI contains the mean point estimate in 0.85-0.99 of replicates
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment import audit  # noqa: E402


def _sig(x):
    return 1 / (1 + np.exp(-x))


def gen(name, rng):
    """Return (df, kwargs for audit). Columns: g (group), task, L, model, y, s."""
    G, per = 300, 2
    task_eff = {"A": 1.2, "B": -0.6, "C": -0.6}
    rows = []
    sizes = rng.integers(1, 9, G) if name == "unequal_groups" else np.full(G, per)
    if name == "few_groups":
        G, sizes = 60, np.full(60, per)
    for g in range(G):
        task = "ABC"[g % 3]
        model = g % 2
        L = rng.normal(loc=0.8 * model if name == "difficulty_by_model" else 0.0)
        for _ in range(sizes[g]):
            eta = task_eff[task] + 0.8 * L
            if name == "misspecified_baseline":
                eta = task_eff[task] + 1.2 * (L ** 2 - 1)
            elif name == "interaction_meta":
                eta = 1.5 * L * (1 if task == "A" else -1)
            elif name == "imbalance":
                eta = -3.2 + 0.5 * task_eff[task] + 0.8 * L
            elif name in ("simpson", "scale_by_model", "difficulty_by_model"):
                eta = (-1.4 if model == 0 else 0.9) + 0.8 * L
            y = int(rng.random() < _sig(eta)) if name != "random_labels" else int(rng.random() < 0.4)
            e = rng.normal()
            s = {
                "noise": e, "few_groups": e, "unequal_groups": e, "imbalance": e, "random_labels": L + e,
                "metadata_copy": eta + 0.3 * e,
                "metadata_plus_noise": eta + e,
                "nonlinear_of_metadata": L ** 2 + 0.3 * e,
                "misspecified_baseline": L ** 2 + 0.3 * e,
                "interaction_meta": L * (1 if task == "A" else -1) + 0.3 * e,
                "difficulty_by_model": 2.0 * model + e,
                "scale_by_model": (10.0 if model else 1.0) * e + 5 * model,
                "simpson": 3.0 * model + e,
                "weak_signal": e + 0.3 * y, "moderate_signal": e + 0.6 * y, "strong_signal": e + 1.2 * y,
                "signal_with_difficulty": L + e + 0.6 * y,
                "hard_only_signal": e + (1.2 * y if L > 0 else 0.0),
            }[name]
            rows.append({"g": g, "task": task, "L": L, "model": f"m{model}", "y": y, "s": s})
    df = pd.DataFrame(rows)
    base = ["task", "L"]
    kw = {}
    if name in ("simpson", "scale_by_model", "difficulty_by_model"):
        base = ["model", "L"]
        kw["perm_strata"] = ["model"]
    if name == "scale_by_model":
        kw["rank_within"] = "model"
    return df, base, kw


NULLS = ["noise", "metadata_copy", "nonlinear_of_metadata", "metadata_plus_noise", "random_labels", "few_groups",
         "unequal_groups", "imbalance"]
ADVERSARIAL_NULLS = ["misspecified_baseline", "interaction_meta", "difficulty_by_model", "scale_by_model", "simpson"]
POSITIVES = ["weak_signal", "moderate_signal", "strong_signal", "signal_with_difficulty", "hard_only_signal"]
THRESH = {**{k: ("max", 0.10) for k in NULLS + ADVERSARIAL_NULLS},
          "moderate_signal": ("min", 0.80), "strong_signal": ("min", 0.80), "signal_with_difficulty": ("min", 0.80),
          "hard_only_signal": ("min", 0.50)}


def one(name, rep, n_boot, n_perm, seeds, perm_stop=None, interval="bootstrap"):
    rng = np.random.default_rng(10_000 * (1 + zlib.crc32(name.encode()) % 1000) + rep)
    df, base, kw = gen(name, rng)
    r = audit(df, "s", "y", "g", base, (), seeds=range(seeds), n_perm=n_perm, n_boot=n_boot, n_boot_desc=50,
              perm_stop=perm_stop, interval=interval, **kw)
    return {"p": r["increment_perm_p"], "adds": r["adds_information"], "inc": r["increment"], "head": r["headline_family"], "ci": r["increment_ci"],
            "lin_inc": r["families"]["linear"]["increment"], "drawn": r["n_perm_drawn"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--perm-stop", type=int, default=None, help="Besag-Clifford stopping count")
    ap.add_argument("--interval", default="bootstrap", choices=["bootstrap", "half"])
    ap.add_argument("--min-weak", type=float, default=None, help="pre-registered floor for weak_signal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    names = a.only.split(",") if a.only else (["noise", "moderate_signal", "misspecified_baseline"] if a.coverage
                                              else NULLS + ADVERSARIAL_NULLS + POSITIVES)
    n_boot = 100 if a.coverage else 0
    out = {"version": __import__("baseline_increment").__version__, "reps": a.reps, "perms": a.perms,
           "seeds": a.seeds, "n_boot": n_boot, "perm_stop": a.perm_stop, "interval": a.interval, "scenarios": {}}
    ok_all = True
    for name in names:
        t0 = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(one)(name, r, n_boot, a.perms, a.seeds, a.perm_stop, a.interval) for r in range(a.reps))
        p = np.array([x["p"] for x in res]); inc = np.array([x["inc"] for x in res])
        rec = {"reject_rate": float(np.mean([x["adds"] for x in res])), "perm_p_below_05": float((p < 0.05).mean()), "mean_increment": float(inc.mean()),
               "mean_linear_increment": float(np.mean([x["lin_inc"] for x in res])),
               "headline": {h: sum(x["head"] == h for x in res) for h in ("linear", "spline", "gbm")},
               "mean_perms_drawn": float(np.mean([x["drawn"] for x in res])), "sec": round(time.time() - t0)}
        if a.coverage:
            truth = inc.mean()
            rec["ci_coverage_of_mean_estimate"] = float(np.mean([x["ci"][0] <= truth <= x["ci"][1] for x in res]))
            rec["ci_excludes_zero"] = float(np.mean([x["ci"][0] > 0 for x in res]))
            rec["mean_ci_width"] = float(np.mean([x["ci"][1] - x["ci"][0] for x in res]))
            rec["pass"] = 0.85 <= rec["ci_coverage_of_mean_estimate"] <= 0.99
        elif name == "weak_signal" and a.min_weak is not None:
            rec["pass"] = rec["reject_rate"] >= a.min_weak
        elif name in THRESH:
            kind, v = THRESH[name]
            rec["pass"] = rec["reject_rate"] <= v if kind == "max" else rec["reject_rate"] >= v
        ok_all &= rec.get("pass", True)
        out["scenarios"][name] = rec
        print(name, json.dumps(rec), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    out["all_pass"] = bool(ok_all)
    json.dump(out, open(a.out, "w"), indent=1)
    print("ALL PASS" if ok_all else "SOME FAIL", flush=True)


if __name__ == "__main__":
    main()
