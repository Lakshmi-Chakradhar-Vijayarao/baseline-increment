"""Stress test of the binary audit (acceptance fixed in docs/prereg/PREREG_VALIDATION.md, Part B, before the first run).

    python3 simulations/stress_suite.py --jobs 8 --out simulations/results/stress_v0.11.1.json
"""
import argparse, json, math, os, sys, time, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_increment import audit  # noqa: E402
from suite import gen as suite_gen    # noqa: E402

NULLS = ["many_baselines_noise", "many_baselines_copy", "large_noise", "large_copy", "uneven_noise", "uneven_copy",
         "clustered_noise", "clustered_shared_noise"]
POWER = ["many_baselines_signal", "large_weak"]
BORDER = ["misspecified_baseline", "interaction_meta", "few_groups"]
# held out from the 0.12 development (docs/prereg/PREREG_0.12.md); 400 replicates each
HELDOUT = ["product_meta", "threshold_meta", "three_way_meta", "large_copy_400"]
TASK = {"A": 1.2, "B": -0.6, "C": -0.6}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def gen_heldout(name, rng):
    n = 600
    g = np.repeat(np.arange(n // 2), 2)
    L1 = np.repeat(rng.normal(size=n // 2), 2)
    L2 = np.repeat(rng.normal(size=n // 2), 2)
    task = np.array(list("ABC"))[g % 3]
    sign = np.where(task == "A", 1.0, -1.0)
    f = {"product_meta": L1 * L2, "threshold_meta": ((L1 > 0.5) & (task == "A")).astype(float) * 2.0 - 0.7,
         "three_way_meta": L1 * L2 * sign}[name]
    y = (rng.random(n) < _sig(1.2 * f)).astype(int)
    df = pd.DataFrame({"g": g, "task": task, "L1": L1, "L2": L2, "y": y, "s": f + 0.3 * rng.normal(size=n)})
    return df, (["L1", "L2"] if name == "product_meta" else ["task", "L1", "L2"])


def gen(name, rng):
    if name in HELDOUT[:3]:
        return gen_heldout(name, rng)
    if name == "large_copy_400":
        name = "large_copy"
    if name.startswith("many_baselines"):
        n = 600
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        c1 = rng.choice(list("ABC"), n)
        eta = 0.8 * x1 - 0.6 * x2 + np.vectorize(TASK.get)(c1)
        y = (rng.random(n) < _sig(eta)).astype(int)
        df = pd.DataFrame({"g": np.arange(n), "x1": x1, "x2": x2, "c1": c1, "y": y})
        for k in range(4):
            df[f"z{k}"] = rng.normal(size=n)
        for k in range(3):
            df[f"k{k}"] = rng.choice(list("PQRS"), n)
        e = rng.normal(size=n)
        df["s"] = {"many_baselines_noise": e, "many_baselines_copy": np.tanh(x1) * 2 + x2 ** 2 + (c1 == "A") + 0.3 * e,
                   "many_baselines_signal": e + 0.6 * y}[name]
        return df, ["x1", "x2", "c1", "z0", "z1", "z2", "z3", "k0", "k1", "k2"]
    if name.startswith("large"):
        G, per = 5000, 2
    elif name.startswith("uneven"):
        G = 1000
    else:
        G, per = 600, 3
    rows = []
    for g in range(G):
        task = "ABC"[g % 3]
        L = rng.normal()
        size = min(40, int(rng.geometric(1 / 4))) if name.startswith("uneven") else per
        u_y, u_s = rng.normal(scale=1.5), rng.normal(scale=1.5)
        for _ in range(size):
            eta = TASK[task] + 0.8 * L + (u_y if name.startswith("clustered") else 0.0)
            y = int(rng.random() < _sig(eta))
            e = rng.normal()
            if name in ("large_noise", "uneven_noise"):
                s = e
            elif name in ("large_copy", "uneven_copy"):
                s = TASK[task] + 0.8 * L + 0.3 * e
            elif name == "large_weak":
                s = e + 0.15 * y
            else:                                            # clustered: detector has its own group effect
                s = u_s + e
            row = {"g": g, "task": task, "L": L, "y": y, "s": s}
            if name == "clustered_shared_noise":
                row["Lg"] = L + u_s                            # a baseline column that shares the detector's group effect
            rows.append(row)
    df = pd.DataFrame(rows)
    base = ["task", "L"] + (["Lg"] if name == "clustered_shared_noise" else [])
    return df, base


def one(name, rep, null="residual", salt=0):
    rng = np.random.default_rng(50_000 * (1 + zlib.crc32(name.encode()) % 1000) + rep + 7_919_000 * salt)
    if name in BORDER:
        df, base, kw = suite_gen(name, rng)
    else:
        (df, base), kw = gen(name, rng), {}
    t0 = time.time()
    r = audit(df, "s", "y", "g", base, (), seeds=range(2), n_perm=200, perm_stop=10, n_boot=0, n_boot_desc=50, null=null, **kw)
    return {"adds": r["adds_information"], "inc": r["increment"], "head": r["headline_family"], "sec": time.time() - t0,
            "n": len(df), "groups": r["n_groups"]}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--null", default="residual")
    ap.add_argument("--seed-salt", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    import baseline_increment
    out = {"version": baseline_increment.__version__, "settings": {"seeds": 2, "n_perm": 200, "perm_stop": 10,
           "families": ["linear", "spline", "gbm"], "null": a.null, "seed_salt": a.seed_salt}, "scenarios": {}}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    names = a.only.split(",") if a.only else NULLS + POWER + BORDER + HELDOUT
    for name in names:
        if name in out["scenarios"]:
            continue
        reps = 400 if name in BORDER + HELDOUT else 100
        t0 = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(one)(name, r, a.null, a.seed_salt) for r in range(reps))
        k = sum(x["adds"] for x in res)
        lo, hi = wilson(k, reps)
        rec = {"reps": reps, "flag_rate": k / reps, "wilson95": [round(lo, 3), round(hi, 3)],
               "mean_increment": float(np.mean([x["inc"] for x in res])),
               "headline": {h: sum(x["head"] == h for x in res) for h in ("linear", "spline", "gbm")},
               "mean_rows": float(np.mean([x["n"] for x in res])), "mean_groups": float(np.mean([x["groups"] for x in res])),
               "sec_per_replicate_one_core": round(float(np.mean([x["sec"] for x in res])), 1), "wall_sec": round(time.time() - t0)}
        if name in NULLS:
            rec["rule"], rec["verdict"] = "<= 0.10", "pass" if rec["flag_rate"] <= 0.10 else "FAIL"
        elif name in BORDER + HELDOUT:
            rec["rule"] = "Wilson upper <= 0.10 pass; point <= 0.10 borderline; else FAIL"
            rec["verdict"] = "pass" if hi <= 0.10 else "borderline" if rec["flag_rate"] <= 0.10 else "FAIL"
        else:
            rec["rule"], rec["verdict"] = "power, reported", "reported"
        out["scenarios"][name] = rec
        print(name, json.dumps(rec), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
