"""PREREG_TREC.md: within-query stratification of the null on the TREC LLMJudge structure (single run)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from baseline_increment.ordinal import audit_ordinal  # noqa: E402

FAST = dict(seeds=range(2), n_perm=200, perm_stop=10, n_jobs=1, null="residual_gbm", n_boot_desc=20)
BASE = ["qid", "log_words"]


def load():
    d = pd.read_csv(HERE / "data/j3_llmjudge.csv")
    d["len_dec"] = pd.qcut(d.log_words.rank(method="first"), 10, labels=False).astype(str)
    return d


def permuted(d, rep, judge="NISTRetrieval-instruct0"):
    rng = np.random.default_rng(880_000 + rep)
    strata = d.qid.astype(str) + "|" + d.len_dec
    s = d[judge].to_numpy().copy()
    for ix in pd.Series(range(len(d))).groupby(strata.to_numpy()).indices.values():
        s[ix] = s[rng.permutation(ix)]
    return s


def arm(d, rep, strata, power=False):
    x = d.copy()
    s = permuted(d, rep)
    if power:
        r = pd.Series(s).rank().to_numpy()
        s = r + 0.6 * r.std() * (x.human.to_numpy() - x.human.mean())
    x["perm"] = s
    return bool(audit_ordinal(x, "perm", "human", "docid", BASE, perm_strata=strata, **FAST)["adds_information"])


def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main():
    from joblib import Parallel, delayed
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    d = load()
    out = {"prereg": "docs/prereg/PREREG_TREC.md", "arms": {}}
    t0 = time.time()
    for name, strata, power in [("A0", (), False), ("A1", ("qid",), False), ("A2", ("qid", "len_dec"), False), ("P1", ("qid",), True)]:
        flags = Parallel(n_jobs=jobs)(delayed(arm)(d, r, strata, power) for r in range(100))
        k = int(sum(flags))
        out["arms"][name] = {"strata": list(strata), "flag_rate": k / 100, "wilson95": wilson(k, 100)}
        print(name, out["arms"][name], flush=True)
        json.dump(out, open(HERE / "trec_strata_results.json", "w"), indent=1)
    a1, p1 = out["arms"]["A1"], out["arms"]["P1"]
    out["decision"] = ("pass" if a1["wilson95"][1] <= 0.10 and p1["flag_rate"] >= 0.80 else
                       "borderline" if a1["flag_rate"] <= 0.10 else "fail")
    judges = [c for c in d.columns if c not in ("qid", "docid", "human", "log_words", "len_dec")]
    def one(j):
        r = audit_ordinal(d, j, "human", "docid", BASE, perm_strata=("qid",), **FAST)
        return j, {"increment": r["increment"], "p": r["increment_perm_p"], "adds": r["adds_information"]}
    out["judges_under_A1"] = dict(Parallel(n_jobs=jobs)(delayed(one)(j) for j in judges))
    out["wall_sec"] = round(time.time() - t0)
    json.dump(out, open(HERE / "trec_strata_results.json", "w"), indent=1)
    print("decision", out["decision"], flush=True)


if __name__ == "__main__":
    main()
