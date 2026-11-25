"""Baseline-relative audits of public judge data (docs/judge-audits.md).

FaithBench (j2) is rebuilt locally by build_judge_tables.py (licence CC BY-NC-SA); TREC 2024 LLMJudge (j3) is in the repo.
    python validation/judges/audit_public_judges.py
"""
import json, sys
import pandas as pd
import os
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
from baseline_increment.judge import judge_report
D = os.path.join(os.path.dirname(__file__), "data") + "/"
out = {"faithbench": [], "llmjudge": []}
fb = pd.read_csv(D + "j2_faithbench.csv")
for lab in ("hallucinated", "hallucinated_or_questionable"):
    for j in ["gpt_3_5_turbo_risk", "gpt_4_turbo_risk", "gpt_4o_risk", "hhem_2_1_risk", "trueteacher_risk", "true_nli_risk"]:
        for name, base in (("label_free", ["summarizer", "log_words"]), ("with_source_difficulty", ["summarizer", "log_words", f"d_src_{lab}"])):
            r = judge_report(fb, j, lab, base, group="source_id", n_jobs=4)
            out["faithbench"].append({"label": lab, "judge": j, "baselines": name, **{k: r["judge"][k] for k in ("agreement", "judge_alone_auroc", "baselines_alone_auroc", "baselines_plus_judge_auroc")},
                                      "gain": r["gain"], "ci": [r["gain_ci"]["lower"], r["gain_ci"]["upper"]], "verdict": r["incremental_prediction"]["verdict"], "head": r["headline_family"]})
            print("FB", lab, j, name, round(r["gain"], 3), r["incremental_prediction"]["verdict"], flush=True)
lj = pd.read_csv(D + "j3_llmjudge.csv")
lj["relevant"] = (lj["human"] >= 2).astype(int)
judges = [c for c in lj.columns if c not in ("qid", "docid", "human", "log_words", "relevant")]
for j in judges:
    r = judge_report(lj, j, "relevant", ["qid", "log_words"], group="docid", n_jobs=4)
    out["llmjudge"].append({"judge": j, **{k: r["judge"][k] for k in ("judge_alone_auroc", "baselines_alone_auroc", "baselines_plus_judge_auroc")},
                            "gain": r["gain"], "ci": [r["gain_ci"]["lower"], r["gain_ci"]["upper"]], "verdict": r["incremental_prediction"]["verdict"], "head": r["headline_family"]})
    print("LJ", j, round(r["gain"], 3), r["incremental_prediction"]["verdict"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "judge_audits_results.json"), "w"), indent=1, default=float)
print("DONE")
