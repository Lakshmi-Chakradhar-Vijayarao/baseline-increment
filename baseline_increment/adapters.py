"""Turn harness logs into the per-item table baseline-increment audits.

The audit needs one row per (item, sample): a 0/1 label for the event a detector should flag, the
detector's score, a group id (rows sharing it never straddle folds), and cheap baselines.

lm-evaluation-harness (`--log_samples` writes samples_<task>_<date>.jsonl):
    label  wrong   = 1 - <metric> (default acc)
    score  risk    = 1 - max softmax over the choices' log-likelihoods (the model's own confidence)
    baselines      task, log_prompt_words, n_choices
    group          task:doc_id

Inspect AI (`.json` or `.eval` logs, or directories of them; `.eval` is read natively, no inspect_ai needed):
    label  wrong   = 1 if the `label_scorer` value is incorrect ('I', 0, False, 'N'), 0 if correct
    score          a numeric scorer value (`score_scorer`) or a numeric sample metadata field; a C/I
                   scorer used as a detector (e.g. an LLM judge) becomes 1 = "says wrong"
    baselines      log_input_words, log_output_words, plus numeric/categorical metadata fields
    group          task:sample id, shared across models and epochs (0.9.0; pre-0.9 grouped per log file)
    text           input/completion length counts message text only (chat lists, content parts, attachments)

Own-panel difficulty (`--lomo`, both harnesses): given logs from several models on the same task, each row gets
`lomo_share_wrong`, the error rate of the *other* models on the same item (epochs averaged per model first), and
`n_ref_models`. Difficulty then comes from your own prompts and grader, which transfers far better than a table built
under another protocol (see the README). Rows are grouped by item, so one question never straddles folds.
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
                         "item": f"{task}:{s['doc_id']}", "model": Path(f).parent.name,
                         "wrong": int(float(s[metric]) == 0.0), "risk": round(1 - conf, 6),
                         "log_prompt_words": _logwords(prompt), "n_choices": len(lls)})
    return pd.DataFrame(rows)


def _expand_logs(paths) -> list[Path]:
    """Files as given; directories are searched recursively for Inspect logs (*.eval, *.json)."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.suffix in (".eval", ".json") and not x.name.startswith("."))
        else:
            out.append(p)
    return out


def _read_eval_zip(p: Path) -> dict:
    """Native reader for Inspect's .eval format (a zip: header.json + samples/<id>_epoch_<n>.json), so inspect_ai is
    not required. Checked for equality against inspect_ai.log.read_eval_log (tests/test_inspect_native.py)."""
    import zipfile
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        head = "header.json" if "header.json" in names else "_journal/start.json"
        d = json.loads(z.read(head))
        samples = [json.loads(z.read(n)) for n in names if n.startswith("samples/") and n.endswith(".json")]
    samples.sort(key=lambda x: (x.get("epoch", 1), str(x.get("id"))))
    d["samples"] = samples
    return d


def _inspect_log(path: str | Path) -> dict | None:
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
    elif p.suffix == ".eval":
        try:
            d = _read_eval_zip(p)
        except Exception as native_error:                 # e.g. zstd members on Python < 3.14, or an unknown layout
            try:
                from inspect_ai.log import read_eval_log
            except ImportError:
                raise SystemExit(f"{p.name}: cannot read this .eval natively ({native_error}). Logs written by Inspect on "
                                 "Python 3.14 use zstd compression, which needs Python >= 3.14 or `pip install inspect_ai`.")
            d = json.loads(read_eval_log(str(p)).model_dump_json())
    else:
        return None
    return d if isinstance(d, dict) and "eval" in d else None


def _inspect_samples(path: str | Path) -> tuple[list[dict], str, str]:
    d = _inspect_log(path)
    if d is None:
        raise ValueError(f"{Path(path).name}: not an Inspect log")
    ev = d.get("eval") or {}
    return d.get("samples") or [], ev.get("model", ""), ev.get("task", "")


def _resolve(x, att):
    if isinstance(x, str) and x.startswith("attachment://"):
        return att.get(x[len("attachment://"):], x)
    return x


def _text(x, att=None) -> str:
    """Plain text of an Inspect input or completion: a string, a list of chat messages, or content parts
    (attachment:// references resolved). Only message text is counted, never ids, roles or other fields."""
    att = att or {}
    x = _resolve(x, att)
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "content" in x:
            return _text(x["content"], att)
        return str(_resolve(x.get("text", ""), att)) if x.get("type", "text") == "text" else ""
    if isinstance(x, (list, tuple)):
        return " ".join(t for t in (_text(m, att) for m in x) if t)
    return str(x)


def _scorer_value(scores: dict, spec: str, fname: str):
    """`name` or `name:key` (for scorers whose value is a dict)."""
    name, _, key = spec.partition(":")
    if name not in scores:
        raise ValueError(f"{fname}: scorer {name!r} not in sample scores ({sorted(scores)})")
    v = (scores[name] or {}).get("value")
    if isinstance(v, dict):
        if not key:
            raise ValueError(f"{fname}: scorer {name!r} returns a dict {sorted(v)}; use '{name}:<key>'")
        if key not in v:
            raise ValueError(f"{fname}: key {key!r} not in scorer {name!r} value ({sorted(v)})")
        v = v[key]
    return v


def _correctness(v, partial_as: str = "drop", noanswer_as: str = "wrong") -> int | None:
    """1 = wrong, 0 = correct, None = drop. 'P' (partial) and 'N' (no answer) follow the chosen policy."""
    if isinstance(v, (dict, list)):
        return None
    if v in ("P", "partial"):
        return {"wrong": 1, "correct": 0, "drop": None}[partial_as]
    if v in ("N",):
        return {"wrong": 1, "correct": 0, "drop": None}[noanswer_as]
    if v in _INCORRECT:
        return 1
    if v in _CORRECT:
        return 0
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
        return int(v < 0.5)
    return None


def list_inspect(log_files) -> dict:
    """Scorers (with value types and dict keys) and metadata fields found in the logs."""
    scorers, meta, n = {}, set(), 0
    for f in _expand_logs(log_files):
        d = _inspect_log(f)
        if d is None:
            continue
        for s in d.get("samples") or []:
            n += 1
            for k, sc in (s.get("scores") or {}).items():
                v = (sc or {}).get("value")
                e = scorers.setdefault(k, {"types": set(), "keys": set(), "examples": set()})
                e["types"].add(type(v).__name__)
                if isinstance(v, dict):
                    e["keys"].update(v)
                elif len(e["examples"]) < 5:
                    e["examples"].add(repr(v))
            meta.update((s.get("metadata") or {}).keys())
    return {"n_samples": n, "scorers": {k: {a: sorted(b) for a, b in v.items()} for k, v in scorers.items()},
            "metadata_fields": sorted(meta)}


def from_inspect(log_files: list[str | Path], label_scorer: str, score_scorer: str | None = None,
                 score_metadata: str | None = None, group_by: str = "item", partial_as: str = "drop",
                 noanswer_as: str = "wrong") -> pd.DataFrame:
    """One row per (log, sample, epoch). `group_by="item"` (default from 0.9.0): rows of the same task:id share a
    group across models and epochs, so a question never straddles folds; "file" restores the pre-0.9 grouping.
    Rows that cannot be used are counted by reason in `df.attrs["dropped"]`, never dropped silently."""
    if not (score_scorer or score_metadata):
        raise ValueError("give score_scorer (a numeric scorer) or score_metadata (a numeric metadata field)")
    if group_by not in ("item", "file"):
        raise ValueError("group_by must be 'item' or 'file'")
    rows, dropped, seen_input = [], {}, {}
    files = [f for f in _expand_logs(log_files)]
    for f in files:
        d = _inspect_log(f)
        if d is None:
            dropped["not an Inspect log (file)"] = dropped.get("not an Inspect log (file)", 0) + 1
            continue
        ev = d.get("eval") or {}
        model, task = ev.get("model", ""), ev.get("task", "")
        for s in d.get("samples") or []:
            att = s.get("attachments") or {}
            sc = s.get("scores") or {}
            wrong = _correctness(_scorer_value(sc, label_scorer, Path(f).name), partial_as, noanswer_as)
            if wrong is None:
                k = f"label scorer value not usable ({_scorer_value(sc, label_scorer, Path(f).name)!r})"
                dropped[k] = dropped.get(k, 0) + 1
                continue
            if score_scorer:
                score = _scorer_value(sc, score_scorer, Path(f).name)
            else:
                score = (s.get("metadata") or {}).get(score_metadata)
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                score = _correctness(score)                 # a C/I judge as the detector: 1 = it says wrong
                if score is None:
                    dropped["detector score missing or not numeric"] = dropped.get("detector score missing or not numeric", 0) + 1
                    continue
            inp = _text(s.get("input", ""), att)
            item = f"{task}:{s['id']}"
            h = hash(" ".join(inp.split()))
            if seen_input.setdefault(item, h) != h:
                raise ValueError(f"sample id {s['id']!r} of task {task!r} has different input text in different logs; "
                                 "ids must name the same question in every log (give samples explicit ids)")
            out = _text(_resolve((s.get("output") or {}).get("completion", ""), att), att)
            row = {"group": item if group_by == "item" else f"{Path(f).stem}:{s['id']}", "sample_id": s["id"],
                   "epoch": s.get("epoch", 1), "item": item, "model": model, "log": Path(f).name, "wrong": wrong,
                   "score": float(score), "log_input_words": _logwords(inp), "log_output_words": _logwords(out)}
            for k, v in (s.get("metadata") or {}).items():
                if k != score_metadata and isinstance(v, (int, float, str)) and not isinstance(v, bool):
                    row[f"meta_{k}"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["dropped"] = dropped
    return df


def add_lomo_difficulty(df: pd.DataFrame, item: str = "item", model: str = "model", label: str = "wrong",
                        min_ref: int = 1) -> tuple[pd.DataFrame, dict]:
    """Leave-one-model-out difficulty from the models in `df`: for each row, the mean error of the other models on
    the same item (each model's epochs averaged first). Rows with fewer than `min_ref` reference models are dropped.
    Returns (rows, panel metadata for `check --panel-meta`)."""
    per = df.groupby([item, model])[label].mean().rename("_e").reset_index()
    tot = per.groupby(item)["_e"].agg(["sum", "count"])
    per = per.join(tot, on=item)
    per["n_ref_models"] = per["count"] - 1
    per["lomo_share_wrong"] = ((per["sum"] - per["_e"]) / per["n_ref_models"].where(per["n_ref_models"] > 0)).round(6)
    out = df.merge(per[[item, model, "lomo_share_wrong", "n_ref_models"]], on=[item, model], how="left")
    keep = out["n_ref_models"] >= min_ref
    out = out[keep].reset_index(drop=True)
    out["group"] = out[item].astype(str)
    models = sorted(df[model].astype(str).unique())
    meta = {"source": "own logs, leave-one-model-out", "models": models, "n_models": len(models) - 1,
            "n_ref_models_min": int(out["n_ref_models"].min()) if len(out) else 0, "dropped_rows": int((~keep).sum()),
            "note": "n_models = reference models per row (panel size minus the model being audited)"}
    return out, meta
