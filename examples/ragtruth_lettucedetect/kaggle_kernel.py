"""Kaggle kernel (GPU, internet on): score RAGTruth's test split with LettuceDetect, then run the audit.

Runs score.py's logic on a Kaggle GPU (never on a laptop). RAGTruth is fetched at a pinned commit.
Outputs in /kaggle/working: ragtruth_lettucedetect.csv, .meta.json, audit.txt, audit.json.
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"   # pinned 18 Nov 2026
BASE = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_COMMIT}/dataset"
SCORE_PY = "https://raw.githubusercontent.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment/main/examples/ragtruth_lettucedetect/score.py"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lettucedetect==0.2.3", "baseline-increment"], check=True)
d = Path("/kaggle/working/ragtruth")
d.mkdir(parents=True, exist_ok=True)
for name in ("response.jsonl", "source_info.jsonl"):
    urllib.request.urlretrieve(f"{BASE}/{name}", d / name)
here = Path("/kaggle/working")
score_src = next(iter(Path("/kaggle/input").glob("**/score.py")), None)   # attached copy, if any
if score_src is None:
    urllib.request.urlretrieve(SCORE_PY, here / "score.py")
    score_src = here / "score.py"
subprocess.run([sys.executable, str(score_src), str(d), str(here / "ragtruth_lettucedetect.csv")], check=True)
with open(here / "audit.txt", "w") as f:
    subprocess.run(["baseline-increment", str(here / "ragtruth_lettucedetect.csv"), "--score", "detector_score",
                    "--label", "hallucinated", "--group", "source_id", "--baselines", "task_type,generator,log_words",
                    "--strata", "task_type", "--jobs", "-1", "--json", str(here / "audit.json")], check=True, stdout=f)
print(open(here / "audit.txt").read())
