"""Staged, resumable runner for the planner validation of docs/prereg/PREREG_0.13.md Part 2.

The same computations, seeds and settings as planner_suite.py, split so each piece fits a Kaggle CPU session. Each
sample size (search) or (size, arm) cell is checkpointed, so a timed-out session loses at most one piece.

    search: python simulations/planner_staged.py search --scenario S1 --out <ckpt.json>
    cell:   python simulations/planner_staged.py cell --scenario S1 --rec 1500 --arm fresh --size rec --out <ckpt.json>
    merge:  python simulations/planner_staged.py merge --s3 simulations/results/planner_v0.13.0_S3.json --staged a.json b.json ...
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import baseline_increment.planner as P  # noqa: E402
from planner_suite import SCEN, calibrate_alt, one_alt  # noqa: E402

SALT = 13
NULL = "residual_gbm"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save(path, d):
    json.dump(d, open(path, "w"), indent=1, default=float)


def search(a):
    pi, a0, m, delta = SCEN[a.scenario]
    ck = load(a.out)
    ck.setdefault("scenario", a.scenario)
    if "calib" not in ck:
        ck["calib"] = P.calibrate(pi, a0, m, delta)
        save(a.out, ck)
    ck.setdefault("search", {})
    for N in P.default_grid(pi, m):
        if str(N) in ck["search"]:
            row = ck["search"][str(N)]
        else:
            row = P.plan(pi, a0, m, delta, grid=[N], reps=200, n_jobs=a.jobs, null=NULL, seed=1, calib=ck["calib"], criterion=a.criterion)["grid"][0]
            ck["search"][str(N)] = row
            save(a.out, ck)
            print("search", a.scenario, row, flush=True)
        if row["power"] >= 0.80:
            ck["recommended_groups"] = N
            save(a.out, ck)
            return
    ck["recommended_groups"] = P.default_grid(pi, m)[-1]
    save(a.out, ck)


def cell(a):
    pi, a0, m, delta = SCEN[a.scenario]
    ck = load(a.out)
    key = f"{a.arm}|{a.size}"
    if key in ck.get("cells", {}):
        return
    c = ck.get("calib") or P.calibrate(pi, a0, m, delta)
    ck["calib"] = c
    N = {"half": max(2, a.rec // 2), "rec": a.rec, "double": a.rec * 2}[a.size]
    if a.arm == "pred":
        row = P.plan(pi, a0, m, delta, grid=[N], reps=200, n_jobs=a.jobs, null=NULL, seed=2, calib=c, criterion=a.criterion)["grid"][0]
    elif a.arm == "fresh":
        row = P.plan(pi, a0, m, delta, grid=[N], reps=400, n_jobs=a.jobs, null=NULL, seed=1000 + SALT, calib=c, criterion=a.criterion)["grid"][0]
    else:
        from joblib import Parallel, delayed
        ca = calibrate_alt(pi, a0, m, delta)
        flags = Parallel(n_jobs=a.jobs)(delayed(one_alt)(N, m, ca, 5_000_000 + 7919 * SALT + N * 1009 + r, NULL, a.criterion) for r in range(400))
        k = int(sum(flags))
        row = {"groups": N, "power": k / 400, "wilson95": P._wilson(k, 400), "calibration_alt": ca}
    ck.setdefault("cells", {})[key] = row
    ck["rec"] = a.rec
    save(a.out, ck)
    print("cell", a.scenario, key, row, flush=True)


def merge(a):
    out = json.load(open(a.s3))
    scen = dict(out["scenarios"])
    staged = {}
    for f in a.staged:
        d = load(f)
        s = d.get("scenario")
        staged.setdefault(s, {"cells": {}})
        staged[s]["cells"].update(d.get("cells", {}))
        if "recommended_groups" in d:
            staged[s]["recommended_groups"] = d["recommended_groups"]
            staged[s]["search"] = d.get("search")
    for s, d in staged.items():
        rec = d["recommended_groups"]
        cells = []
        for size in ("half", "rec", "double"):
            pr, fr, al = (d["cells"].get(f"{arm}|{size}") for arm in ("pred", "fresh", "alt"))
            if not (pr and fr and al):
                continue
            cells.append({"groups": pr["groups"], "predicted": pr["power"], "fresh": fr["power"], "fresh_wilson95": fr["wilson95"],
                          "independent": al["power"], "P1_inside": fr["wilson95"][0] <= pr["power"] <= fr["wilson95"][1],
                          "P2_close": abs(pr["power"] - al["power"]) <= 0.10})
        indep = d["cells"].get("alt|rec", {}).get("power")
        scen[s] = {"inputs": list(SCEN[s]), "recommended_groups": rec, "search": d.get("search"), "cells": cells,
                   "independent_power_at_recommended": indep, "staged": True}
    out["scenarios"] = scen
    cells = [c for s in scen.values() for c in s["cells"]]
    v = {"P1_inside": sum(c["P1_inside"] for c in cells), "P2_close": sum(c["P2_close"] for c in cells), "n_cells": len(cells),
         "P3_min_independent_at_recommended": min(s["independent_power_at_recommended"] for s in scen.values())}
    v["complete"] = len(cells) == 9
    v["pass"] = bool(v["complete"] and v["P1_inside"] >= 8 and v["P2_close"] >= 8 and v["P3_min_independent_at_recommended"] >= 0.70)
    out["verdict"] = v
    save(a.out, out)
    print(json.dumps(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "cell", "merge"])
    ap.add_argument("--scenario")
    ap.add_argument("--rec", type=int)
    ap.add_argument("--arm", choices=["pred", "fresh", "alt"])
    ap.add_argument("--size", choices=["half", "rec", "double"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--criterion", choices=["pass", "adds_information"], default="adds_information")
    ap.add_argument("--out", required=True)
    ap.add_argument("--s3")
    ap.add_argument("--staged", nargs="*", default=[])
    a = ap.parse_args()
    {"search": search, "cell": cell, "merge": merge}[a.stage](a)


if __name__ == "__main__":
    main()
