"""Build the LLM-judge validation table (docs/prereg/PREREG_0.12.md, Part C). Per-item, no text.

Label: WQA's deterministic rule label `wrong` (gold-alias matching), which no judge produced. Restricted to PopQA and
TriviaQA, the two datasets where the rule agrees with human labels at weighted kappa >= 0.80
(decision-point/h1/results/evaluator_sensitivity.json: 0.858 and 0.802).
Detectors: two LLM judges' risk = 1 - P(Yes "answer is correct"): Falcon3-10B-Instruct (WQA stage 2) and GLM-4-9B
(decision-point judge3).
Baselines: d1 = the other models' rule error rate on the same question; log answer words.

    python validation/build_judge_table.py
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

DP = Path.home() / "Desktop/decision-point/h1"
WQA = Path.home() / "Downloads/SUBMISSIONS/WQA_data/main"
OUT = Path(__file__).parent / "data"
sys.path.insert(0, str(DP))
from common import eligible, load  # noqa: E402

falcon = {}
for f in ("wqa-judge-falcon-a/judge_falcon_nq_open-triviaqa.json", "wqa-judge-falcon-b/judge_falcon_popqa-sciq.json"):
    falcon.update(json.load(open(WQA / "stage2" / f))["p_yes"])
rows = []
for ds in ("popqa", "triviaqa"):
    glm = json.load(open(DP / "results/judge3" / f"judge3_{ds}.json"))["p_yes_mass"]
    ms = eligible(ds)
    C = {m: load(ds, m) for m in ms}
    W = np.array([np.asarray(C[m]["wrong"], int) for m in ms])
    for i, m in enumerate(ms):
        q = C[m]["qid"].astype(str)
        keys = [f"{m}|{x}" for x in q]
        rows.append(pd.DataFrame({
            "dataset": ds, "model": m, "qid": q, "wrong": W[i], "d1": (W.sum(0) - W[i]) / (len(ms) - 1),
            "log_answer_words": [round(math.log1p(len(str(a).split())), 4) for a in C[m]["served_norm"]],
            "falcon_risk": [1 - falcon[k] if k in falcon else np.nan for k in keys],
            "glm_risk": [1 - glm[k][0] if k in glm else np.nan for k in keys]}))
df = pd.concat(rows, ignore_index=True)
n0 = len(df)
df = df.dropna().reset_index(drop=True)
df.to_csv(OUT / "wqa_judge.csv", index=False)
print(f"wqa_judge.csv {df.shape} (dropped {n0 - len(df)} rows without both judge scores)")
print(df.groupby("dataset").agg(n=("wrong", "size"), wrong=("wrong", "mean"), models=("model", "nunique")))
