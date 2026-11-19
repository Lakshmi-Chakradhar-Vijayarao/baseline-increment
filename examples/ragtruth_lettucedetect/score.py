"""Worked example on a real public detector: LettuceDetect on the RAGTruth test split.

Inputs (both MIT-licensed):
  RAGTruth           github.com/ParticleMedia/RAGTruth  (dataset/response.jsonl, dataset/source_info.jsonl)
  LettuceDetect base KRLabsOrg/lettucedect-base-modernbert-en-v1 (revision recorded in the output)

Writes ragtruth_lettucedetect.csv with one row per test response: the detector score (max token
hallucination probability, as in LettuceDetect's own RAGTruth evaluation), the label (any annotated
hallucination span), and three cheap baselines a release review would ask about: task type, the
generating model, and log response length. No response or source text is written.

    Run on a GPU, not a laptop: kaggle_kernel.py does this on Kaggle
    (copy kernel-metadata.example.json to kernel-metadata.json, set your username, `kaggle kernels push -p .`).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

DETECTOR = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


def main(dataset_dir: str, out: str = "ragtruth_lettucedetect.csv") -> None:
    import torch
    from huggingface_hub import model_info
    from lettucedetect.detectors.transformer import TransformerDetector

    d = Path(dataset_dir)
    src = {s["source_id"]: s for s in map(json.loads, open(d / "source_info.jsonl"))}
    resp = [r for r in map(json.loads, open(d / "response.jsonl")) if r["split"] == "test"]
    rev = model_info(DETECTOR).sha
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    det = TransformerDetector(model_path=DETECTOR, revision=rev, device=device)
    t0 = time.time()
    header = ["response_id", "source_id", "task_type", "generator", "log_words", "hallucinated", "detector_score"]
    done: set[str] = set()
    if Path(out).exists():                      # resumable: keep complete rows, continue after them
        with open(out, newline="") as f:
            rows = [r for r in csv.reader(f)][1:]
        rows = [r for r in rows if len(r) == len(header)]
        done = {r[0] for r in rows}
        with open(out, "w", newline="") as f:
            csv.writer(f).writerows([header, *rows])
    else:
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(header)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        for i, r in enumerate(resp):
            if r["id"] in done:
                continue
            s = src[r["source_id"]]
            tok = det.predict_prompt(s["prompt"], r["response"], output_format="tokens")
            score = max((x["prob"] for x in tok), default=0.0)
            w.writerow([r["id"], r["source_id"], s["task_type"], r["model"],
                        round(math.log(1 + len(r["response"].split())), 4), int(bool(r["labels"])), round(score, 6)])
            f.flush()
            if i % 300 == 0:
                print(f"{i}/{len(resp)} {time.time() - t0:.0f}s", flush=True)
    Path(out).with_suffix(".meta.json").write_text(json.dumps(
        {"detector": DETECTOR, "revision": rev, "device": str(device), "n": len(resp),
         "score": "max token hallucination probability", "dataset": "RAGTruth test split"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
