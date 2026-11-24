"""Deterministic-function diagnostic (docs/prereg/PREREG_DIAG.md).

    python3 simulations/diag_suite.py --jobs 8 --out simulations/results/diag_v0.12.json
"""
import argparse, json, math, os, sys, time, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment import audit  # noqa: E402

F = {
    "D1": lambda L1, L2, s: L1,
    "D2": lambda L1, L2, s: L1 + L2,
    "D3": lambda L1, L2, s: L1 * L2,
    "D4": lambda L1, L2, s: L1 * L2 * s,
    "D5": lambda L1, L2, s: 2.0 * (L1 > 0.5) - 0.7,
    "D6": lambda L1, L2, s: (L1 > 0.5) * L2,
    "D7": lambda L1, L2, s: np.exp(L1) / 2,
    "D8": lambda L1, L2, s: np.sign(L1 * L2),
    "D9": lambda L1, L2, s: np.sin(2 * L1) + L2 ** 2 - 1,
    "D10": lambda L1, L2, s: L1 + (s > 0) * L2,
}
CELLS = list(F) + ["N0", "P1", "P2"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def gen(cell, n, rng):
    G = n // 2
    g = np.repeat(np.arange(G), 2)
    L1, L2 = np.repeat(rng.normal(size=G), 2), np.repeat(rng.normal(size=G), 2)
    task = np.array(list("ABC"))[g % 3]
    s = np.where(task == "A", 1.0, -1.0)
    flab = {"N0": L1, "P1": L1 * L2, "P2": L1 * L2}.get(cell)
    f = F[cell](L1, L2, s) if cell in F else flab
    y = (rng.random(n) < _sig(1.2 * f)).astype(int)
    e = rng.normal(size=n)
    S = {"N0": e, "P1": e + 0.6 * y, "P2": L1 * L2 + e + 0.6 * y}.get(cell, f + 0.3 * e) if cell not in F else f + 0.3 * e
    return pd.DataFrame({"g": g, "task": task, "L1": L1, "L2": L2, "y": y, "s": S})


def one(cell, n, null, rep, salt):
    rng = np.random.default_rng(60_000 * (1 + zlib.crc32(f"{cell}|{n}".encode()) % 1000) + rep + 7_919_000 * salt)
    df = gen(cell, n, rng)
    r = audit(df, "s", "y", "g", ["task", "L1", "L2"], seeds=range(2), n_perm=200, perm_stop=10, n_boot=0,
              n_boot_desc=20, null=null)
    return r["adds_information"], r["increment"], r["increment_perm_p"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed-salt", type=int, default=21)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"prereg": "docs/prereg/PREREG_DIAG.md", "reps": a.reps, "cells": {}}
    for n in (600, 3000):
        for null in ("residual_gbm", "residual_sl"):
            for cell in CELLS:
                key = f"{cell}|{n}|{null}"
                if key in out["cells"]:
                    continue
                t0 = time.time()
                res = Parallel(n_jobs=a.jobs)(delayed(one)(cell, n, null, r, a.seed_salt) for r in range(a.reps))
                k = sum(x[0] for x in res)
                lo, hi = wilson(k, a.reps)
                rec = {"flag_rate": k / a.reps, "wilson95": [lo, hi], "mean_increment": float(np.mean([x[1] for x in res])),
                       "mean_p": float(np.mean([x[2] for x in res])), "sec": round(time.time() - t0)}
                if cell in F or cell == "N0":
                    rec["class"] = "handled" if hi <= 0.10 else "borderline" if k / a.reps <= 0.10 else "not handled"
                    if rec["class"] == "not handled" and rec["mean_increment"] >= 0.01:
                        rec["reading"] = "model-class increment"
                out["cells"][key] = rec
                print(key, json.dumps(rec), flush=True)
                json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
