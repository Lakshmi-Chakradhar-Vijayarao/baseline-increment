"""R1: coverage of the primary gain interval (docs/prereg/PREREG_R1.md). Checkpointed per cell and phase.

    python simulations/r1_coverage.py --cells D1@600,D4@3000,uneven_noise --jobs 4 --out simulations/results/r1_part.json
    python simulations/r1_coverage.py --merge a.json b.json ... --out simulations/results/r1_coverage.json
"""
import argparse, json, math, os, sys, time, zlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from baseline_increment import audit  # noqa: E402
import diag_suite, stress_suite, suite  # noqa: E402

R_REF, R_COV = 1000, 200
SALT_COV, SALT_REF = 31, 32
FROZEN = dict(seeds=range(2), n_perm=1, perm_stop=None, n_boot=0, n_boot_desc=10, interval="delong", null="residual_gbm")
DIAG = list(diag_suite.CELLS)
STRESS = ["uneven_noise", "uneven_copy", "clustered_noise", "clustered_shared_noise"]
SUITE = ["imbalance", "few_groups"]
ALL = [f"{c}@{n}" for n in (600, 3000) for c in DIAG] + STRESS + SUITE


def make(cell, rep, salt):
    if "@" in cell:
        c, n = cell.split("@")
        rng = np.random.default_rng(80_000 * (1 + zlib.crc32(cell.encode()) % 1000) + rep + 7_919_000 * salt)
        return diag_suite.gen(c, int(n), rng), ["task", "L1", "L2"], {}
    rng = np.random.default_rng(90_000 * (1 + zlib.crc32(cell.encode()) % 1000) + rep + 7_919_000 * salt)
    if cell in STRESS:
        df, base = stress_suite.gen(cell, rng)
        return df, base, {}
    df, base, kw = suite.gen(cell, rng)
    return df, base, kw


def one(cell, rep, salt):
    for attempt in (0, 1):
        try:
            df, base, kw = make(cell, rep, salt)
            r = audit(df, "s", "y", "g", base, (), **FROZEN, **kw)
            return {"inc": r["increment"], "ci": r["increment_ci"]}
        except Exception as e:  # technical failure: one retry with the same seed, then report
            err = repr(e)[:200]
    return {"error": err}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarise(cell, ref, cov):
    ok_ref = [x["inc"] for x in ref if "inc" in x]
    ok_cov = [x for x in cov if "inc" in x]
    d = float(np.mean(ok_ref))
    se = float(np.std(ok_ref, ddof=1) / math.sqrt(len(ok_ref)))
    widths = [x["ci"][1] - x["ci"][0] for x in ok_cov]
    hit = [x["ci"][0] <= d <= x["ci"][1] for x in ok_cov]
    k, n = int(sum(hit)), len(hit)
    cov_rate = k / n
    estimable = se <= 0.10 * float(np.mean(widths))
    status = ("not estimable" if not estimable else "pass" if 0.85 <= cov_rate <= 0.99 else
              "under-covers" if cov_rate < 0.85 else "over-conservative")
    return {"reference_gain": d, "reference_se": se, "n_ref": len(ok_ref), "n_cov": n, "coverage": cov_rate,
            "coverage_wilson95": wilson(k, n), "mean_gain": float(np.mean([x["inc"] for x in ok_cov])),
            "mean_width": float(np.mean(widths)), "miss_low": int(sum(x["ci"][1] < d for x in ok_cov)),
            "miss_high": int(sum(x["ci"][0] > d for x in ok_cov)),
            "technical_failures": (len(ref) - len(ok_ref)) + (len(cov) - len(ok_cov)), "status": status}


def run(a):
    from joblib import Parallel, delayed
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"prereg": "docs/prereg/PREREG_R1.md", "cells": {}}
    for cell in a.cells.split(","):
        rec = out["cells"].setdefault(cell, {})
        for phase, R, salt in (("ref", R_REF, SALT_REF), ("cov", R_COV, SALT_COV)):
            if phase in rec:
                continue
            t0 = time.time()
            rec[phase] = Parallel(n_jobs=a.jobs)(delayed(one)(cell, r, salt) for r in range(R))
            rec[f"{phase}_sec"] = round(time.time() - t0)
            json.dump(out, open(a.out, "w"))
        rec["summary"] = summarise(cell, rec["ref"], rec["cov"])
        print(cell, json.dumps(rec["summary"]), flush=True)
        json.dump(out, open(a.out, "w"))


def merge(a):
    cells = {}
    for f in a.merge:
        for k, v in json.load(open(f))["cells"].items():
            if "summary" in v:
                cells[k] = v["summary"]
    est = {k: v for k, v in cells.items() if v["status"] != "not estimable"}
    fails = [k for k, v in est.items() if v["status"] != "pass"]
    controls = [f"{c}@{n}" for c in ("N0", "P1", "P2") for n in (600, 3000)]
    control_fail = [k for k in controls if k in est and est[k]["status"] != "pass"]
    missing = [k for k in ALL if k not in cells]
    if missing:
        decision = "incomplete"
    elif not fails:
        decision = "PASS"
    elif len(fails) <= 0.25 * len(est) and not control_fail:
        decision = "PASS WITH SCOPE RESTRICTION"
    else:
        decision = "FAIL"
    out = {"prereg": "docs/prereg/PREREG_R1.md", "cells": cells, "failing_cells": fails, "control_failures": control_fail,
           "not_estimable": [k for k, v in cells.items() if v["status"] == "not estimable"], "missing": missing,
           "decision": decision}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("decision", "failing_cells", "control_failures", "not_estimable", "missing")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a) if a.merge else run(a)
