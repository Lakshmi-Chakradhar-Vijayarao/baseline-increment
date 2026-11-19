"""Turn harness logs into the per-item table baseline-increment audits.

The audit needs one row per (item, sample): a 0/1 label for the event a detector should flag, the
detector's score, a group id (rows sharing it never straddle folds), and cheap baselines.

lm-evaluation-harness (`--log_samples` writes samples_<task>_<date>.jsonl):
    label  wrong   = 1 - <metric> (default acc)
    score  risk    = 1 - max softmax over the choices' log-likelihoods (the model's own confidence)
    baselines      task, log_prompt_words, n_choices
    group          task:doc_id

Inspect AI (a `.json` log; `.eval` needs inspect_ai installed):
    label  wrong   = 1 if the `label_scorer` value is incorrect ('I', 0, False, 'N'), 0 if correct
    score          a numeric scorer value (`score_scorer`) or a numeric sample metadata field; a C/I
                   scorer used as a detector (e.g. an LLM judge) becomes 1 = "says wrong"
    baselines      log_input_words, log_output_words, plus numeric/categorical metadata fields
    group          sample id (epochs of one sample share a group)
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pandas as pd

_INCORRECT = {"I", "N", "incorrect", 0, 0.0, False}
_CORRECT = {"C", "Y", "correct", 1, 1.0, True}


def _logwords(text) -> float:
    return round(math.log1p(len(str(text).split())), 4)


def _ll(x) -> float:
    """lm-eval stores each choice's (loglikelihood, is_greedy) pair, sometimes as a string."""
    if isinstance(x, str):
        x = ast.literal_eval(x)
    while isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return float(x[0] if isinstance(x, (list, tuple)) else x)


def from_lm_eval(sample_files: list[str | Path], metric: str = "acc") -> pd.DataFrame:
    rows = []
    for f in sample_files:
        name = Path(f).name
        task = name[len("samples_"):].rsplit("_", 1)[0] if name.startswith("samples_") else Path(f).stem
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            if metric not in s:
                raise ValueError(f"{name}: metric {metric!r} not in samples (have {s.get('metrics')})")
            resps = s["filtered_resps"]
            if isinstance(resps, str):
                resps = ast.literal_eval(resps)
            lls = [_ll(r) for r in resps]
            m = max(lls)
            z = sum(math.exp(v - m) for v in lls)
            conf = 1.0 / z                                      # softmax probability of the top choice
            args = s.get("arguments")
            if isinstance(args, str):
                args = ast.literal_eval(args)
            if isinstance(args, dict):                      # {"gen_args_0": {"arg_0": prompt, "arg_1": choice}, ...}
                first = args.get("gen_args_0") or next(iter(args.values()), {})
                prompt = first.get("arg_0", "") if isinstance(first, dict) else ""
            elif args and isinstance(args[0], (list, tuple)):  # older versions: [[context, continuation], ...]
                prompt = args[0][0]
            else:
                prompt = ""
            rows.append({"group": f"{task}:{s['doc_id']}", "task": task, "doc_id": s["doc_id"],
                         "wrong": int(float(s[metric]) == 0.0), "risk": round(1 - conf, 6),
                         "log_prompt_words": _logwords(prompt), "n_choices": len(lls)})
    return pd.DataFrame(rows)


def _inspect_samples(path: str | Path) -> tuple[list[dict], str]:
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return d.get("samples") or [], (d.get("eval") or {}).get("model", "")
    from inspect_ai.log import read_eval_log            # .eval logs are compressed; needs inspect_ai
    log = json.loads(read_eval_log(str(p)).model_dump_json())
    return log.get("samples") or [], (log.get("eval") or {}).get("model", "")


def _correctness(v) -> int | None:
    if v in _INCORRECT:
        return 1
    if v in _CORRECT:
        return 0
    if isinstance(v, (int, float)) and 0 <= v <= 1:
        return int(v < 0.5)
    return None


def from_inspect(log_files: list[str | Path], label_scorer: str, score_scorer: str | None = None,
                 score_metadata: str | None = None) -> pd.DataFrame:
    if not (score_scorer or score_metadata):
        raise ValueError("give score_scorer (a numeric scorer) or score_metadata (a numeric metadata field)")
    rows = []
    for f in log_files:
        samples, model = _inspect_samples(f)
        for s in samples:
            sc = s.get("scores") or {}
            if label_scorer not in sc:
                raise ValueError(f"{Path(f).name}: scorer {label_scorer!r} not in sample scores ({sorted(sc)})")
            wrong = _correctness(sc[label_scorer].get("value"))
            if wrong is None:
                continue
            score = (sc.get(score_scorer) or {}).get("value") if score_scorer else (s.get("metadata") or {}).get(score_metadata)
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                score = _correctness(score)                 # a C/I judge as the detector: 1 = it says wrong
                if score is None:
                    continue
            out = (s.get("output") or {}).get("completion", "")
            row = {"group": f"{Path(f).stem}:{s['id']}", "sample_id": s["id"], "epoch": s.get("epoch", 1),
                   "model": model, "wrong": wrong, "score": float(score),
                   "log_input_words": _logwords(s.get("input", "")), "log_output_words": _logwords(out)}
            for k, v in (s.get("metadata") or {}).items():
                if k != score_metadata and isinstance(v, (int, float, str)) and not isinstance(v, bool):
                    row[f"meta_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)
