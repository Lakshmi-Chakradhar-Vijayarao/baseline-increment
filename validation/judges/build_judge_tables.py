"""Build the per-item tables for docs/prereg/PREREG_JUDGES.md from the public sources (no text kept).

    git clone https://github.com/nlpyang/geval; git clone https://github.com/vectara/FaithBench
    git clone https://github.com/llm4eval/LLMJudge; git clone https://github.com/llm4eval/LLMJudge-benchmark
    python -I validation/judges/build_judge_tables.py --src <dir holding the four clones>

Writes validation/judges/data/{j1_summeval.csv, j3_llmjudge.csv} (MIT sources) and j2_faithbench.csv, which is not
committed (CC BY-NC-SA; rebuild it with this script).
"""
import argparse, ast, glob, hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"


def words(t):
    return round(math.log1p(len(str(t).split())), 4)


def j1(src):
    meta = {(x["doc_id"], x["system_id"]): x for x in json.load(open(src / "geval/data/summeval.json"))}
    rows, dropped = [], 0
    judge = {}
    for dim, f in [("consistency", "con"), ("coherence", "coh"), ("fluency", "flu"), ("relevance", "rel")]:
        for x in json.load(open(src / f"geval/results/gpt4_{f}_detailed.json")):
            resp = x["all_responses"]
            resp = ast.literal_eval(resp) if isinstance(resp, str) else resp
            vals = []
            for r in resp:
                m = re.search(r"[1-5]", str(r))
                if m:
                    vals.append(int(m.group()))
            judge[(x["doc_id"], x["system_id"], dim)] = float(np.mean(vals)) if len(vals) >= 10 else np.nan
    for (doc, sysid), x in meta.items():
        sc = x["scores"] if isinstance(x["scores"], dict) else ast.literal_eval(x["scores"])
        r = {"doc_id": doc, "system_id": sysid, "log_words": words(x["system_output"])}
        for dim in ("consistency", "coherence", "fluency", "relevance"):
            r[f"human_{dim}"] = int(round(sc[dim]))
            r[f"human_{dim}_mean"] = sc[dim]
            r[f"geval_{dim}"] = judge.get((doc, sysid, dim), np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    for dim in ("consistency", "coherence", "fluency", "relevance"):
        g = df.groupby("doc_id")[f"human_{dim}_mean"]
        df[f"d_doc_{dim}"] = (g.transform("sum") - df[f"human_{dim}_mean"]) / (g.transform("count") - 1)
    n0 = len(df)
    df = df.dropna(subset=["geval_consistency"]).reset_index(drop=True)
    print(f"J1: {len(df)} items ({n0 - len(df)} dropped for < 10 parsable G-Eval ratings on consistency)")
    return df


def j2(src):
    rows = []
    for f in sorted(glob.glob(str(src / "FaithBench/data_for_release/*.json"))):
        for s in json.load(open(f))["samples"]:
            labs = {l for a in s["annotations"] for l in a["label"]}
            top = {l.split(".")[0] for l in labs}
            m = s["metadata"]
            r = {"sample_id": f"{Path(f).stem}:{s['sample_id']}", "source_id": hashlib.sha1(s["source"].encode()).hexdigest()[:12],
                 "summarizer": m["summarizer"], "log_words": words(s["summary"]),
                 "hallucinated": int("Unwanted" in top), "hallucinated_or_questionable": int(bool(top & {"Unwanted", "Questionable"}))}
            for k in ("hhem-2.1", "gpt-3.5-turbo", "gpt-4-turbo", "gpt_4o", "trueteacher", "true_nli"):
                v = m.get(k)
                r[k.replace("-", "_").replace(".", "_") + "_risk"] = (1 - v) if v is not None else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    for lab in ("hallucinated", "hallucinated_or_questionable"):
        g = df.groupby("source_id")[lab]
        df[f"d_src_{lab}"] = (g.transform("sum") - df[lab]) / (g.transform("count") - 1)
    df = df[df.groupby("source_id")["hallucinated"].transform("count") > 1].reset_index(drop=True)
    print(f"J2: {len(df)} items, {df.source_id.nunique()} sources")
    return df


def j3(src):
    q = pd.read_csv(src / "LLMJudge/data/llmjudge_test_qrel_idx.txt", sep=r"\s+", header=None,
                    names=["qid", "zero", "docid", "human"])
    docs = {}
    with open(src / "LLMJudge/data/llm4eval_document_2024.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[d["docid"]] = words(d["doc"])
    q["log_words"] = q.docid.map(docs)
    for f in sorted(glob.glob(str(src / "LLMJudge-benchmark/submissions/scores/*.txt"))):
        name = Path(f).stem
        s = pd.read_csv(f, sep=r"\s+", header=None, names=["qid", "zero", "docid", name])
        q = q.merge(s[["qid", "docid", name]], on=["qid", "docid"], how="left")
    judges = [c for c in q.columns if c not in ("qid", "zero", "docid", "human", "log_words")]
    print(f"J3: {len(q)} pairs, {q.qid.nunique()} queries, {len(judges)} judges; missing judge cells "
          f"{int(q[judges].isna().sum().sum())}; missing lengths {int(q.log_words.isna().sum())}")
    return q.drop(columns=["zero"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    j1(src).to_csv(OUT / "j1_summeval.csv", index=False)
    j2(src).to_csv(OUT / "j2_faithbench.csv", index=False)
    j3(src).to_csv(OUT / "j3_llmjudge.csv", index=False)


if __name__ == "__main__":
    main()
