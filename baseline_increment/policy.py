"""Declared evaluation policy (1.0).

A policy file states, before a run, what is being tested: the detector score, the label, the grouping, the baselines,
the outcome type and the operating points. It may contain only rules this version has calibrated. Anything else is
refused, by name, rather than silently ignored.

The policy does not change the statistics. It fixes the inputs to the existing engine and is hashed (SHA-256 of its
canonical JSON form) into the report, the evidence package and the run ledger.

Example (JSON; YAML is read too if PyYAML is installed):

    {"version": 1,
     "detector": {"name": "LettuceDetect v0.1", "score": "lettucedetect"},
     "label": "hallucinated",
     "group": "source_id",
     "baselines": ["d_src", "task_type"],
     "operating_points": [0.05, 0.10]}
"""
import hashlib
import json
from pathlib import Path

ALLOWED = {
    "version": "policy format version; must be 1",
    "description": "free text, recorded only",
    "owner": "free text, recorded only",
    "construct": "free text, recorded only: what the label is meant to measure (e.g. 'answer contains an unsupported claim')",
    "intended_use": "free text, recorded only: the decision the result will support (e.g. 'release gate for detector v2')",
    "outcome": "'binary' (default) or 'ordinal'",
    "detector": "{'name': free text, 'score': column}",
    "label": "label column",
    "group": "group column (items sharing a value never straddle train/test); null for none",
    "baselines": "list of baseline columns",
    "rank_within": "rank the score within this column (e.g. model)",
    "preset": "'fast' (default; the calibrated CI preset) or 'full' (5 seeds, 200 permutations, refit bootstrap)",
    "operating_points": "false-positive rates for the 'in use' section, each in (0, 0.5]",
    "alpha": "significance level; only 0.05 is calibrated",
    "metric": "'auroc' (binary) or 'concordance' (ordinal); no other metric is calibrated",
    "minimum_gain": "minimum gain in AUROC the lower 95% bound must exceed for PASS (default 0; binary outcome only)",
}

# Recognised but refused: each needs its own pre-registered calibration before a policy may rely on it.
DEFERRED = {
    "minimum_operational_gain": "a minimum recall-gain gate is not calibrated",
    "auprc": "the increment in precision-recall area is not calibrated",
    "omitted_baseline_bound": "the omitted-baseline sensitivity bound is not implemented yet",
    "heterogeneity_test": "the test for differences in increment across datasets or models is not implemented yet",
    "sample_size": "the sample-size planner is not implemented yet",
}


class PolicyError(ValueError):
    pass


def _read(path):
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise PolicyError("reading a YAML policy needs PyYAML (pip install pyyaml); or write the policy as JSON")
        return yaml.safe_load(text)
    return json.loads(text)


def validate(p):
    if not isinstance(p, dict):
        raise PolicyError("a policy must be a mapping")
    deferred = [k for k in p if k in DEFERRED]
    if deferred:
        raise PolicyError("not yet calibrated in this version: " +
                          "; ".join(f"'{k}' ({DEFERRED[k]})" for k in deferred))
    unknown = [k for k in p if k not in ALLOWED]
    if unknown:
        raise PolicyError(f"unknown policy keys {unknown}; allowed: {sorted(ALLOWED)}")
    if p.get("version") != 1:
        raise PolicyError("policy 'version' must be 1")
    det = p.get("detector")
    if not isinstance(det, dict) or not isinstance(det.get("score"), str):
        raise PolicyError("policy 'detector' must be a mapping with a 'score' column")
    if set(det) - {"name", "score"}:
        raise PolicyError("policy 'detector' takes only 'name' and 'score'")
    if not isinstance(p.get("label"), str):
        raise PolicyError("policy 'label' must be a column name")
    if "group" in p and p["group"] is not None and not isinstance(p["group"], str):
        raise PolicyError("policy 'group' must be a column name or null")
    b = p.get("baselines", [])
    if not isinstance(b, list) or not all(isinstance(c, str) for c in b):
        raise PolicyError("policy 'baselines' must be a list of column names")
    outcome = p.get("outcome", "binary")
    if outcome not in ("binary", "ordinal"):
        raise PolicyError("policy 'outcome' must be 'binary' or 'ordinal'")
    if p.get("preset", "fast") not in ("fast", "full"):
        raise PolicyError("policy 'preset' must be 'fast' or 'full'")
    if p.get("alpha", 0.05) != 0.05:
        raise PolicyError("not yet calibrated in this version: alpha other than 0.05")
    want = "concordance" if outcome == "ordinal" else "auroc"
    if p.get("metric", want) != want:
        raise PolicyError(f"not yet calibrated in this version: metric {p['metric']!r} for a {outcome} outcome "
                          f"(calibrated: {want!r})")
    ops = p.get("operating_points", [0.05, 0.10])
    if not isinstance(ops, list) or not ops or not all(isinstance(f, (int, float)) and 0 < f <= 0.5 for f in ops):
        raise PolicyError("policy 'operating_points' must be a non-empty list of false-positive rates in (0, 0.5]")
    mg = p.get("minimum_gain", 0.0)
    if not isinstance(mg, (int, float)) or not 0 <= mg < 0.5:
        raise PolicyError("policy 'minimum_gain' must be a number in [0, 0.5)")
    if outcome == "ordinal" and "minimum_gain" in p:
        raise PolicyError("not yet calibrated in this version: minimum_gain for an ordinal outcome (no validated interval)")
    if outcome == "ordinal" and "operating_points" in p:
        raise PolicyError("operating points are not defined for an ordinal outcome")
    return p


def canonical(p):
    return json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(spec):
    return hashlib.sha256(canonical(spec).encode()).hexdigest()


def load_policy(path):
    p = validate(_read(path))
    return p, spec_hash(p)


def spec_from_args(kind, **kw):
    """The analysis specification of a run without a policy file (hashed the same way, so the ledger can count
    distinct specifications either way)."""
    return {"kind": kind, **{k: v for k, v in kw.items() if v not in (None, [], "")}}
