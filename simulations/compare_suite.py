"""Calibration suite for `compare` (acceptance rules fixed in docs/prereg/PREREG_0.8.md before the first run).

    python3 simulations/compare_suite.py --reps 100 --jobs 8 --out simulations/results/compare_v0.8.0.json
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment.compare import compare  # noqa: E402

SCEN = ["reseed", "monotone", "interaction", "coarsened", "strict_upgrade", "complementary"]
# (test, kind, threshold): kind "max" = ship-blocking null, "min" = power target (reported)
RULES = {
    "reseed": [("T1", "max", 0.10), ("T2", "min", 0.80)],
    "monotone": [("T1", "max", 0.10), ("T2", "max", 0.10), ("T3", "max", 0.10)],
    "interaction": [("T1", "max", 0.10)],
    "coarsened": [("T1", "max", 0.10), ("T2", "min", 0.80)],
    "strict_upgrade": [("T1", "min", 0.80), ("T2", "max", 0.10), ("T3", "min", 0.80)],
    "complementary": [("T1", "min", 0.80), ("T2", "min", 0.80), ("T3", "max", 0.10)],
}
COVERAGE = ["complementary", "strict_upgrade"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def gen(name, rng):
    G, per = 300, 2
    task_eff = {"A": 1.2, "B": -0.6, "C": -0.6}
    rows = []
    for g in range(G):
        task = "ABC"[g % 3]
        L = rng.normal()
        for _ in range(per):
            y = int(rng.random() < _sig(task_eff[task] + 0.8 * L))
            e1, e2, e3 = rng.normal(size=3)
            if name == "strict_upgrade":
                v1, v2 = 0.8 * y + e1 + e2, 0.8 * y + e1
            elif name == "complementary":
                v1, v2 = 0.8 * y + e1, 0.8 * y + e3
            else:
                v1 = 0.8 * y + e1
                v2 = {"reseed": v1 + e2, "monotone": np.exp(v1 / 2),
                      "interaction": v1 * (2.0 if task == "A" else 0.5), "coarsened": np.nan}[name]
            rows.append({"g": g, "task": task, "L": L, "y": y, "v1": v1, "v2": v2})
    df = pd.DataFrame(rows)
    if name == "coarsened":
        df["v2"] = np.digitize(df["v1"], np.quantile(df["v1"], [1 / 3, 2 / 3])).astype(float)
    return df


NULL, SALT = "residual", 0


def one(name, rep, null="residual", salt=0):
    global NULL, SALT
    NULL, SALT = null, salt
    rng = np.random.default_rng(20_000 * (1 + zlib.crc32(name.encode()) % 1000) + rep + 7_919_000 * SALT)
    r = compare(gen(name, rng), "v1", "v2", "y", "g", ["task", "L"], n_jobs=1, null=NULL)
    return {"T1": r["improvement"]["detected"], "T2": r["regression"]["detected"],
            "T3": r["standalone"]["verdict"] != "difference not detected", "diff": r["standalone"]["difference"],
            "ci": r["standalone"]["ci"], "inc1": r["improvement"]["increment"], "inc2": r["regression"]["increment"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--null", default="residual")
    ap.add_argument("--seed-salt", type=int, default=0)
    a = ap.parse_args()
    global NULL, SALT
    NULL, SALT = a.null, a.seed_salt
    from joblib import Parallel, delayed
    import baseline_increment
    out = {"version": baseline_increment.__version__, "reps": a.reps, "settings": "compare defaults (= --fast preset)", "scenarios": {}}
    out["null"], out["seed_salt"] = a.null, a.seed_salt
    block = True
    for name in (a.only.split(",") if a.only else SCEN):
        t0 = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(one)(name, r, a.null, a.seed_salt) for r in range(a.reps))
        rec = {t: float(np.mean([x[t] for x in res])) for t in ("T1", "T2", "T3")}
        rec["mean_diff"] = float(np.mean([x["diff"] for x in res]))
        rec["mean_inc_T1"] = float(np.mean([x["inc1"] for x in res]))
        rec["mean_inc_T2"] = float(np.mean([x["inc2"] for x in res]))
        rec["checks"] = []
        for t, kind, v in RULES[name]:
            ok = rec[t] <= v if kind == "max" else rec[t] >= v
            rec["checks"].append({"test": t, "rule": f"{'<=' if kind == 'max' else '>='} {v}", "kind": "null (ship-blocking)" if kind == "max" else "power (reported)", "pass": bool(ok)})
            if kind == "max" and not ok:
                block = False
        if name in COVERAGE:
            truth = rec["mean_diff"]
            cov = float(np.mean([x["ci"][0] <= truth <= x["ci"][1] for x in res]))
            rec["T3_coverage_of_mean_diff"] = cov
            rec["T3_coverage_pass"] = 0.85 <= cov <= 0.99
        rec["sec"] = round(time.time() - t0)
        out["scenarios"][name] = rec
        print(name, json.dumps(rec), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    out["ship_blocking_all_pass"] = bool(block)
    out["t3_coverage_all_pass"] = bool(all(out["scenarios"][n].get("T3_coverage_pass", True) for n in out["scenarios"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print("SHIP-BLOCKING ALL PASS" if block else "SHIP-BLOCKING FAIL", "| T3 coverage", "pass" if out["t3_coverage_all_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
