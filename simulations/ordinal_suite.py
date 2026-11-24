"""Calibration of --outcome ordinal (acceptance fixed in docs/prereg/PREREG_0.11.md before the first run).

    python3 simulations/ordinal_suite.py --reps 100 --jobs 8 --out simulations/results/ordinal_v0.11.0.json
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment.ordinal import audit_ordinal  # noqa: E402

NULLS = ["noise", "metadata_copy", "nonlinear_of_metadata", "interaction_meta", "few_groups"]
POS = ["weak_signal", "moderate_signal"]
CUTS = [-1, 0.5, 2]
TASK = {"A": 1.2, "B": -0.6, "C": -0.6}


def gen(name, rng):
    G = 60 if name == "few_groups" else 300
    rows = []
    for g in range(G):
        task = "ABC"[g % 3]
        L = rng.normal()
        for _ in range(2):
            if name == "nonlinear_of_metadata":
                eta = TASK[task] + 1.2 * (L ** 2 - 1)
            elif name == "interaction_meta":
                eta = 1.5 * L * (1 if task == "A" else -1)
            else:
                eta = TASK[task] + 0.8 * L
            sev = int(np.digitize(eta + rng.logistic(), CUTS))
            e = rng.normal()
            s = {"noise": e, "few_groups": e, "metadata_copy": eta + 0.3 * e,
                 "nonlinear_of_metadata": L ** 2 + 0.3 * e,
                 "interaction_meta": L * (1 if task == "A" else -1) + 0.3 * e,
                 "weak_signal": e + 0.3 * sev, "moderate_signal": e + 0.6 * sev}[name]
            rows.append({"g": g, "task": task, "L": L, "sev": sev, "s": s})
    return pd.DataFrame(rows)


def one(name, rep):
    rng = np.random.default_rng(30_000 * (1 + zlib.crc32(name.encode()) % 1000) + rep)
    r = audit_ordinal(gen(name, rng), "s", "sev", "g", ["task", "L"], seeds=range(2), n_perm=200, perm_stop=10,
                      n_boot_desc=50)
    return {"adds": r["adds_information"], "p": r["increment_perm_p"], "inc": r["increment"], "head": r["headline_family"],
            "drawn": r["n_perm_drawn"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    import baseline_increment
    out = {"version": baseline_increment.__version__, "reps": a.reps,
           "settings": {"outcome": "ordinal", "seeds": 2, "n_perm": 200, "perm_stop": 10,
                        "families": ["linear", "spline", "gbm"]}, "scenarios": {}}
    ship = True
    for name in NULLS + POS:
        t0 = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(one)(name, r) for r in range(a.reps))
        rec = {"flag_rate": float(np.mean([x["adds"] for x in res])), "mean_increment": float(np.mean([x["inc"] for x in res])),
               "headline": {h: sum(x["head"] == h for x in res) for h in ("linear", "spline", "gbm")},
               "mean_perms_drawn": float(np.mean([x["drawn"] for x in res])), "sec": round(time.time() - t0)}
        if name in NULLS:
            rec["rule"], rec["pass"] = "<= 0.10 (ship-blocking)", rec["flag_rate"] <= 0.10
            ship &= rec["pass"]
        elif name == "moderate_signal":
            rec["rule"], rec["pass"] = ">= 0.80 (power target)", rec["flag_rate"] >= 0.80
        else:
            rec["rule"] = "reported"
        out["scenarios"][name] = rec
        print(name, json.dumps(rec), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    out["ship_blocking_all_pass"] = bool(ship)
    json.dump(out, open(a.out, "w"), indent=1)
    print("SHIP-BLOCKING ALL PASS" if ship else "SHIP-BLOCKING FAIL", flush=True)


if __name__ == "__main__":
    main()
