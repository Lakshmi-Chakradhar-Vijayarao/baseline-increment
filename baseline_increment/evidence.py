"""Evidence packages and the run ledger (1.0).

``--evidence-dir DIR`` writes, for every run, a self-contained folder DIR/<evaluation-id>/:

    report.html          the audit report (executive summary, engineer's numbers, auditor's record)
    results.json         the full machine-readable result
    specification.json   the policy file, or the specification taken from the command line (canonical JSON)
    provenance.json      input hashes, command, versions, time
    environment.json     Python, platform and installed versions of every runtime dependency
    command.txt          the command as run (paths reduced to file names)
    ledger.jsonl         a copy of the run ledger up to and including this run
    MANIFEST.sha256      SHA-256 of every file above (sha256sum format)

and appends one line to DIR/run-ledger.jsonl. The ledger is append-only and hash-chained: each entry carries the
SHA-256 of the previous line, so deleting or editing an earlier run breaks the chain (`baseline-increment
verify-evidence DIR` checks both the chain and every manifest).

Nothing here changes a statistic or a verdict. The ledger only lets a report say how many runs, and how many
distinct specifications, were made on the same input before this one.
"""
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

LEDGER = "run-ledger.jsonl"
MANIFEST = "MANIFEST.sha256"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def read_ledger(root, include_declarations=False):
    f = Path(root) / LEDGER
    if not f.exists():
        return []
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows if include_declarations else [r for r in rows if r.get("kind") != "declaration"]


def _append(root, entry):
    """Append one chained line to the ledger (fsync'd); returns the line."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lf = root / LEDGER
    text = lf.read_text() if lf.exists() else ""
    prev = text.splitlines()[-1] if text.strip() else ""
    entry = {**entry, "run": len([x for x in text.splitlines() if x.strip()]) + 1,
             "prev_entry_sha256": _sha(prev.encode()) if prev else None}
    line = json.dumps(entry, sort_keys=True)
    with open(lf, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def declare(root, policy, policy_sha256, policy_file, input_sha256=None):
    """Record a policy in the ledger before any run uses it. Later reports can then show that the specification was
    declared first, and when."""
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return _append(root, {"kind": "declaration", "time_utc": now, "policy_file": policy_file,
                          "policy_sha256": policy_sha256, "policy": policy, "input_sha256": input_sha256})


def declaration_status(root, policy_sha256, input_sha256):
    """Was this policy declared in the ledger before its first run? (None if no policy.)"""
    if not policy_sha256:
        return None
    rows = read_ledger(root, include_declarations=True)
    decl = next((r for r in rows if r.get("kind") == "declaration" and r.get("policy_sha256") == policy_sha256), None)
    first_run = next((r for r in rows if r.get("kind") != "declaration" and r.get("policy_sha256") == policy_sha256), None)
    if decl is None:
        return {"declared": False}
    before = first_run is None or decl["run"] < first_run["run"]
    return {"declared": True, "declared_utc": decl["time_utc"], "ledger_line": decl["run"], "before_first_run": before,
            "bound_to_this_input": decl.get("input_sha256") == input_sha256 if decl.get("input_sha256") else None}


def ledger_context(root, input_sha256, spec_sha256):
    """What the ledger says about earlier runs on the same input, before this run is added."""
    runs = read_ledger(root)
    same_input = [r for r in runs if r.get("input_sha256") == input_sha256]
    specs = []
    for r in same_input:
        if r["spec_sha256"] not in specs:
            specs.append(r["spec_sha256"])
    if spec_sha256 not in specs:
        specs.append(spec_sha256)
    same_spec = [r for r in same_input if r["spec_sha256"] == spec_sha256]
    return {"run_number": len(read_ledger(root, include_declarations=True)) + 1, "runs_on_this_input": len(same_input) + 1,
            "runs_with_this_specification": len(same_spec) + 1, "specifications_on_this_input": len(specs),
            "earlier_runs_used_other_specifications": any(r["spec_sha256"] != spec_sha256 for r in same_input),
            "first_use_of_specification_utc": same_spec[0]["time_utc"] if same_spec else None}


def record_rows(ctx, spec_sha256, policy_file, p_value=None, policy_sha256=None, decl=None):
    """Auditor rows describing the run record. The Bonferroni figure is reporting only; the verdict is unchanged."""
    rows = [("Analysis specification hash (SHA-256)", spec_sha256),
            ("Declared policy file", policy_file or "none: the analysis was given on the command line")]
    if policy_sha256:
        rows.append(("Policy file hash (SHA-256 of canonical JSON)", policy_sha256))
    if decl is not None:
        if not decl["declared"]:
            rows.append(("Declared in the ledger before the first run", "no (no `declare` entry for this policy)"))
        else:
            v = "yes" if decl["before_first_run"] else "no: first used before it was declared"
            v += f", at {decl['declared_utc']} (ledger line {decl['ledger_line']})"
            if decl["bound_to_this_input"] is not None:
                v += "; declared for this exact input" if decl["bound_to_this_input"] else "; declared for a different input file"
            rows.append(("Declared in the ledger before the first run", v))
    rows += [
            ("Run on this input", f"{ctx['runs_on_this_input']} of {ctx['runs_on_this_input']} recorded "
                                  f"(ledger run {ctx['run_number']})"),
            ("Distinct specifications tested on this input", str(ctx["specifications_on_this_input"]))]
    if ctx["first_use_of_specification_utc"]:
        rows.append(("This specification was first used", ctx["first_use_of_specification_utc"]))
    if ctx["earlier_runs_used_other_specifications"]:
        rows.append(("Note", "earlier runs on this input used a different specification; this is not a first, "
                             "pre-declared test"))
    if p_value is not None and ctx["specifications_on_this_input"] > 1:
        k = ctx["specifications_on_this_input"]
        rows.append(("Permutation p counted over every specification tested on this input (Bonferroni, reporting only)",
                     f"min(1, {k} x {p_value:.3f}) = {min(1.0, k * p_value):.3f}"))
    return rows


def environment():
    from importlib import metadata
    pk = {}
    for name in ("baseline-increment", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "statsmodels",
                 "inspect_ai", "pyyaml"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "implementation": platform.python_implementation(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": pk}


def write_package(root, *, kind, result, report_html, spec, spec_sha256, policy_file, provenance, verdict, policy=None,
                  policy_sha256=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc)
    eval_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + spec_sha256[:8]
    d = root / eval_id
    n = 1
    while d.exists():
        n += 1
        d = root / f"{eval_id}-{n}"
    d.mkdir()
    files = {
        "report.html": report_html,
        "results.json": json.dumps(result, indent=1, default=float, sort_keys=True),
        "specification.json": json.dumps({"analysis_specification": spec, "analysis_spec_sha256": spec_sha256,
                                          "policy_file": policy_file, "policy": policy, "policy_sha256": policy_sha256},
                                         indent=1, sort_keys=True),
        "provenance.json": json.dumps(provenance, indent=1, default=str),
        "environment.json": json.dumps(environment(), indent=1),
        "command.txt": provenance.get("command", "") + "\n",
    }
    for name, text in files.items():
        (d / name).write_text(text)
    entry = _append(root, {"time_utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"), "evaluation_id": d.name, "kind": kind,
                           "input_sha256": provenance.get("input_sha256"), "spec_sha256": spec_sha256,
                           "policy_file": policy_file, "policy_sha256": policy_sha256, "verdict": verdict,
                           "tool": provenance.get("tool")})
    lf = root / LEDGER
    (d / "evidence-record.json").write_text(json.dumps(
        evidence_record(entry, kind, result, spec, spec_sha256, policy_sha256, provenance), indent=1, default=float))
    (d / "ledger.jsonl").write_text(lf.read_text())
    names = sorted(p.name for p in d.iterdir() if p.name != MANIFEST)
    (d / MANIFEST).write_text("".join(f"{_sha((d / x).read_bytes())}  {x}\n" for x in names))
    return d


def head(root):
    lf = Path(root) / LEDGER
    lines = lf.read_text().splitlines() if lf.exists() else []
    return _sha(lines[-1].encode()) if lines else None


DOES_NOT_ESTABLISH = [
    "causal validity: the result is predictive, not causal",
    "absence of omitted baselines: it holds relative to the baselines used, not to every cheap variable",
    "generalisation: other tasks, models, prompts, populations or time periods",
    "label validity: the labels are taken as given",
    "probability calibration of the raw score (reported separately, descriptive)",
    "usefulness after deployment: cost, latency or downstream outcomes",
]
SCHEMA = "https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment/blob/main/baseline_increment/data/evidence_record.schema.json"


def evidence_record(entry, kind, result, spec, spec_sha256, policy_sha256, provenance):
    """The one machine-readable record of a run (schema: data/evidence_record.schema.json, version 1)."""
    rec = {"schema": SCHEMA, "schema_version": 1, "evaluation_id": entry["evaluation_id"], "ledger_line": entry["run"],
           "time_utc": entry["time_utc"], "kind": kind, "tool": provenance.get("tool"),
           "input_sha256": provenance.get("input_sha256"), "input_rows": provenance.get("input_rows"),
           "analysis_spec_sha256": spec_sha256, "policy_sha256": policy_sha256, "specification": spec,
           "command": provenance.get("command"), "does_not_establish": DOES_NOT_ESTABLISH}
    if kind == "compare":
        rec["claim"] = {"improvement_detected": result["improvement"]["detected"],
                        "regression_detected": result["regression"]["detected"],
                        "improvement_p_holm": result["improvement"]["p_holm"], "regression_p_holm": result["regression"]["p_holm"],
                        "standalone_auc_difference": result["standalone"]["difference"]}
    else:
        ip = result.get("incremental_prediction", {})
        rec["claim"] = {"statement": "the score improves held-out prediction beyond the declared baselines, under the declared "
                                     "model families (primary); conditional information is reported separately and is experimental",
                        "verdict": ip.get("verdict", "DESCRIPTIVE"), "minimum_gain": ip.get("minimum_gain"),
                        "gain": result["increment"], "gain_ci": result.get("gain_ci"),
                        "conditional_information": result.get("conditional_information",
                                                              {"adds_information": result["adds_information"],
                                                               "p": result["increment_perm_p"], "status": "experimental"}),
                        "statistical_preset": result.get("statistical_preset"),
                        "adds_information": result["adds_information"], "increment": result["increment"],
                        "permutation_p": result["increment_perm_p"], "headline_family": result["headline_family"],
                        "n_items": result["n_items"], "n_groups": result["n_groups"],
                        "settings": result.get("settings"),
                        "validity_status": (result.get("validity") or {}).get("status"),
                        "validity_reasons": (result.get("validity") or {}).get("reasons")}
    return rec


def verify(root, expect_head=None):
    """Check the ledger chain and every evidence package's manifest. Returns (ok, list of messages).

    A consistent chain shows that no entry was edited, removed or reordered *relative to the others*; someone with
    write access could still regenerate the whole directory. Comparing the head hash with a copy kept elsewhere
    (``expect_head``) closes that gap."""
    root = Path(root)
    msgs, ok = [], True
    lf = root / LEDGER
    lines = lf.read_text().splitlines() if lf.exists() else []
    prev = None
    for i, line in enumerate(lines, 1):
        e = json.loads(line)
        want = _sha(prev.encode()) if prev is not None else None
        if e.get("prev_entry_sha256") != want or e.get("run") != i:
            ok = False
            msgs.append(f"ledger line {i}: chain broken (an earlier entry was edited, removed or reordered)")
        prev = line
    msgs.append(f"ledger: {len(lines)} runs checked; head {head(root)}")
    if expect_head:
        if head(root) != expect_head:
            ok = False
            msgs.append("ledger head does NOT match the expected head: entries were added, removed or the ledger was rebuilt")
        else:
            msgs.append("ledger head matches the expected head")
    else:
        msgs.append("note: without --expect-head this shows internal consistency only; record the head hash outside "
                    "this directory to detect a rebuilt ledger")
    for e in (json.loads(x) for x in lines):
        if e.get("kind") == "declaration":
            msgs.append(f"line {e['run']}: declaration of policy {e['policy_sha256'][:12]}... ({e['time_utc']})")
            continue
        d = root / e["evaluation_id"]
        m = d / MANIFEST
        if not m.exists():
            ok = False
            msgs.append(f"{e['evaluation_id']}: missing")
            continue
        bad = []
        for row in m.read_text().splitlines():
            h, name = row.split("  ", 1)
            if not (d / name).exists() or _sha((d / name).read_bytes()) != h:
                bad.append(name)
        if bad:
            ok = False
        msgs.append(f"{e['evaluation_id']}: " + ("intact" if not bad else f"MODIFIED {bad}"))
    return ok, msgs
