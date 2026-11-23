"""Build baseline_increment/data/difficulty_helm_nq.csv from HELM Lite's public per-instance results.

Card: decision-point/prereg/TOOL_PANEL_TABLES_CARD.md (frozen before download; addenda 1-4).
- Source: gs://crfm-helm-public, lite/benchmark_output/runs/v1.0.0 ... v1.13.0, scenario natural_qa:mode=closedbook
  (95 runs listed in helm_nq_runs.txt; same 1,000 NaturalQuestions validation questions in every run).
- Correctness: HELM's per-instance quasi_exact_match (binary). Error = 1 - quasi_exact_match.
- Runs with overall accuracy < 0.05 are excluded as format/refusal failures (3 runs).
- Difficulty = mean over model families (developer + generation, helm_families.py) of the family's mean error.
- Reliability curve: Spearman of k-family difficulty with the held-out half-panel difficulty, 50 random splits.
Only question hashes and numbers are written; no question text, answers or model outputs.

Usage: python scripts/build_helm_nq.py CACHE_DIR      (downloads into CACHE_DIR if absent; about 600 MB)
"""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[1]))
from helm_families import MIN_ACC, family                       # noqa: E402
from baseline_increment.checks import qhash                     # noqa: E402

HERE = Path(__file__).parent
DATA = HERE.parent / "baseline_increment" / "data"
URL = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/{v}/natural_qa:mode=closedbook,model={m}/{f}"


def fetch(cache):
    for line in (HERE / "helm_nq_runs.txt").read_text().split("\n"):
        if not line.strip():
            continue
        v, m = line.split()
        d = Path(cache) / f"{v}__{m}"; d.mkdir(parents=True, exist_ok=True)
        for f, short in (("instances.json", "i.json"), ("per_instance_stats.json", "p.json")):
            if not (d / short).exists():
                subprocess.run(["curl", "-sf", URL.format(v=v, m=m, f=f), "-o", str(d / short)], check=True)


def main(cache):
    fetch(cache)
    rows, text = [], {}
    for d in sorted(os.listdir(cache)):
        v, m = d.split("__", 1)
        for x in json.load(open(f"{cache}/{d}/i.json")):
            if not x.get("perturbation"):
                text[x["id"]] = x["input"]["text"]
        for p in json.load(open(f"{cache}/{d}/p.json")):
            if p.get("perturbation"):
                continue
            st = {s["name"]["name"]: s.get("mean") for s in p["stats"] if not s["name"].get("perturbation")}
            rows.append((v, m, p["instance_id"], st["quasi_exact_match"]))
    df = pd.DataFrame(rows, columns=["rel", "model", "iid", "qem"])
    acc = df.groupby("model").qem.mean()
    excluded = sorted(acc[acc < MIN_ACC].index)
    df = df[~df.model.isin(excluded)].copy()
    df["fam"] = df.model.map(family); df["err"] = 1 - df.qem
    F = df.groupby(["fam", "iid"]).err.mean().unstack()
    F = F.loc[:, F.notna().all()]
    rng = np.random.default_rng(20261123); fams = np.array(F.index); half = len(fams) // 2
    ks = [k for k in (1, 2, 3, 5, 8, 10, 15, 20) if k <= half]; res = {k: [] for k in ks}
    for _ in range(50):
        p = rng.permutation(fams); A, B = p[:half], p[half:]; tgt = F.loc[B].mean()
        for k in ks:
            for _ in range(10 if k < half else 1):
                res[k].append(spearmanr(F.loc[rng.choice(A, k, replace=False)].mean(), tgt).correlation)
    curve = {str(k): {"median": round(float(np.median(v)), 3), "p05": round(float(np.percentile(v, 5)), 3),
                      "p95": round(float(np.percentile(v, 95)), 3)} for k, v in res.items()}
    out = pd.DataFrame({"qhash": [qhash(text[i]) for i in F.columns], "qid": [f"helm_nq:{i}" for i in F.columns],
                        "n_models": int(df.model.nunique()), "n_families": len(fams),
                        "share_wrong": F.mean().round(4).values})
    out.to_csv(DATA / "difficulty_helm_nq.csv", index=False)
    meta = json.load(open(DATA / "difficulty_meta.json"))
    meta["helm_nq"] = {
        "models": sorted(df.model.unique()), "families": sorted(fams.tolist()), "excluded_runs": excluded,
        "label_source": "HELM quasi_exact_match (error = 1 - qem), family-averaged",
        "protocol": "HELM Lite natural_qa closed-book, 5-shot, quasi-exact-match",
        "n_questions": int(F.shape[1]), "source": "HELM Lite per-instance results, releases v1.0.0-v1.13.0 (gs://crfm-helm-public)",
        "licence": "NaturalQuestions: CC BY-SA 3.0; per-instance results published by Stanford CRFM (HELM code: Apache-2.0). Only question hashes and difficulty values are shipped.",
        "within_protocol_reliability_curve": {"definition": "Spearman of k-family difficulty with held-out half-panel "
                                              "difficulty, 50 random family splits", "half_panel_families": int(len(fams) - half),
                                              "by_k": curve},
        "transfer_note": "Exploratory (card addendum 4), one benchmark: on 478 questions shared with the nq_open table, this table's difficulty predicted WQA models' own errors with AUROC 0.59-0.65 (judge labels) and 0.71-0.80 (string-match labels), vs 0.77-0.84 and 0.80-0.90 for difficulty from 5 other WQA models under WQA's own protocol (inflated by Qwen siblings; OLMo, the one non-Qwen model: judge 0.65 vs 0.78, string match 0.79 vs 0.80).",
    }
    json.dump(meta, open(DATA / "difficulty_meta.json", "w"), indent=1)
    print(len(out), "questions,", len(fams), "families,", df.model.nunique(), "runs; excluded", excluded)
    print(json.dumps(curve))


if __name__ == "__main__":
    main(sys.argv[1])
