"""Read the per-item table from CSV, TSV, Parquet, JSON Lines or a JSON array, and map columns.

Nested JSON fields are flattened to dot paths (`scores.judge.value`). A mapping file renames source columns to the
names you pass on the command line, so an export from any evaluation platform can be audited without a conversion
script:

    {"columns": {"score": "scores.judge.value", "wrong": "labels.is_hallucination", "group": "metadata.doc_id"},
     "derive": {"wrong": {"from": "labels.verdict", "map": {"pass": 0, "fail": 1}}}}

`columns` renames (new name -> source dot path); `derive` builds a column by mapping values of a source column
(unmapped values raise an error, never a silent NaN).
"""
import json
from pathlib import Path

import pandas as pd


def read_table(path, mapping=None) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(p)
        except ImportError as e:
            raise SystemExit("reading Parquet needs pyarrow: pip install 'baseline-increment[parquet]'") from e
    elif suf in (".jsonl", ".ndjson"):
        recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        df = pd.json_normalize(recs)
    elif suf == ".json":
        d = json.loads(p.read_text())
        if isinstance(d, dict):
            d = next((v for v in d.values() if isinstance(v, list)), None)
            if d is None:
                raise ValueError(f"{p.name}: expected a JSON array of records (or an object holding one)")
        df = pd.json_normalize(d)
    elif suf == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        df = pd.read_csv(p)
    if mapping:
        df = apply_mapping(df, json.loads(Path(mapping).read_text()) if not isinstance(mapping, dict) else mapping)
    return df


def apply_mapping(df: pd.DataFrame, m: dict) -> pd.DataFrame:
    df = df.copy()
    for new, src in (m.get("columns") or {}).items():
        if src not in df.columns:
            raise ValueError(f"mapping: source column {src!r} not found (have {sorted(df.columns)[:30]})")
        df[new] = df[src]
    for new, spec in (m.get("derive") or {}).items():
        src, vmap = spec["from"], spec["map"]
        if src not in df.columns:
            raise ValueError(f"mapping: source column {src!r} not found")
        bad = sorted(set(map(str, df[src].unique())) - set(map(str, vmap)))
        if bad:
            raise ValueError(f"mapping: values of {src!r} without a mapping: {bad[:10]}")
        df[new] = df[src].map(lambda v: vmap[str(v)])
    return df
