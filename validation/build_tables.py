"""Build the real-data validation tables (docs/prereg/PREREG_VALIDATION.md, Part A) from the original sources.

Inputs live outside this repository (decision-point analysis outputs and the WQA cells); the outputs hold only ids,
labels, baseline values and detector scores, no text. Run once:  python validation/build_tables.py
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DP = Path(os.path.expanduser("~/Desktop/decision-point/h1"))
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

# ---- RAGTruth test: LettuceDetect + HHEM, D_src = leave-one-generator-out share for the same source
rt = pd.read_csv(HERE.parent / "examples/ragtruth_lettucedetect/kaggle_output/ragtruth_lettucedetect.csv")
sc = json.load(open(DP / "results/ragtruth_finetuned_scores.json"))
ids = [str(r["id"]) for r in sc["responses"]]
assert ids == rt.response_id.astype(str).tolist(), "RAGTruth response order differs between sources"
rt["lettucedetect"] = sc["lettucedetect"]["scores"]
rt["hhem"] = sc["hhem"]["scores"]
assert np.allclose(rt["lettucedetect"], rt["detector_score"], atol=1e-4), "LettuceDetect scores disagree between sources"  # example CSV rounds to 6 dp
grp = rt.groupby("source_id")["hallucinated"]
rt["d_src"] = (grp.transform("sum") - rt.hallucinated) / (grp.transform("count") - 1)
rt[["response_id", "source_id", "task_type", "generator", "log_words", "d_src", "hallucinated", "lettucedetect", "hhem"]] \
    .to_csv(OUT / "ragtruth.csv", index=False)

# ---- TofuEval originals: LettuceDetect + HHEM, D_src = leave-one-out share for the same (dataset, doc, topic)
items = json.load(open(DP / "results/tofueval_items.json"))
ts = json.load(open(DP / "results/tofueval_scores.json"))
orig = [k for k, i in enumerate(items) if i["variant"] == "original"]
O = [items[k] for k in orig]
tf = pd.DataFrame({"source": [f"{i['dataset']}|{i['doc_id']}|{i['topic']}" for i in O], "dataset": [i["dataset"] for i in O],
                   "split": [i["split"] for i in O], "label": [int(i["label"]) for i in O],
                   "log_words": [round(math.log1p(len(i["text"].split())), 4) for i in O],
                   "lettucedetect": np.array(ts["lettucedetect"]["scores"])[orig], "hhem": np.array(ts["hhem"]["scores"])[orig]})
g = tf.groupby("source")["label"]
tf["d_src"] = (g.transform("sum") - tf.label) / (g.transform("count") - 1)
tf = tf[tf.groupby("source")["label"].transform("count") > 1].reset_index(drop=True)
tf.to_csv(OUT / "tofueval.csv", index=False)

# ---- WQA TriviaQA: 8 models, judge labels, three self-uncertainty detectors, D1 = other models' error rate
sys.path.insert(0, str(DP))
from common import eligible, load            # noqa: E402
ms = eligible("triviaqa")
C = {m: load("triviaqa", m) for m in ms}
W = np.array([np.asarray(C[m]["wrong_judge"], int) for m in ms])
rows = []
for i, m in enumerate(ms):
    d1 = (W.sum(0) - W[i]) / (len(ms) - 1)
    rows.append(pd.DataFrame({"model": m, "qid": C[m]["qid"].astype(str), "wrong": W[i], "d1": d1,
                              "p_true": np.asarray(C[m]["det_p_true_risk"], float),
                              "logprob": np.asarray(C[m]["det_neg_mean_lp"], float),
                              "sem_ent": np.asarray(C[m]["det_sem_entropy"], float)}))
wq = pd.concat(rows, ignore_index=True)
wq.to_csv(OUT / "wqa_triviaqa.csv", index=False)
for f in sorted(OUT.glob("*.csv")):
    d = pd.read_csv(f)
    print(f.name, d.shape, {c: round(float(d[c].mean()), 3) for c in d.columns if c in ("hallucinated", "label", "wrong")})
