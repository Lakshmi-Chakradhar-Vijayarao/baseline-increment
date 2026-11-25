"""Command line: baseline-increment data.csv --score s --label y --group g --baselines a,b --strata c"""
import argparse
import json
import sys


from .core import FAMILIES, audit

DEFAULT_INTERVAL = "delong"      # adopted under docs/prereg/PREREG_0.12.md Part B (coverage 0.905-0.965 in all 6 scenarios)
from .tables import read_table


POLICY_FIXED = ("--score", "--label", "--group", "--baselines", "--rank-within", "--outcome", "--families", "--null",
                "--raw-score", "--seeds", "--perms", "--boot", "--perm-stop", "--fast", "--perm-strata", "--min-gain",
                "--interval")


def _flag_given(argv, flag):
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def _apply_policy(a, argv):
    """--policy declares the analysis; flags that would change it are refused rather than merged."""
    from .policy import load_policy
    pol, h = load_policy(a.policy)
    clash = [f for f in POLICY_FIXED if _flag_given(argv, f)]
    if clash:
        raise SystemExit(f"--policy declares the analysis; remove {', '.join(clash)} from the command (or edit the policy)")
    a.score, a.label, a.group = pol["detector"]["score"], pol["label"], pol.get("group")
    a.baselines, a.rank_within = ",".join(pol.get("baselines", [])), pol.get("rank_within")
    a.outcome = pol.get("outcome", "binary")
    a.fast = pol.get("preset", "fast") == "fast"
    if pol.get("preset") == "full":
        a.seeds = 5
    a.min_gain = float(pol.get("minimum_gain", 0.0))
    a.ops = tuple(pol.get("operating_points", [0.05, 0.10]))
    return pol, h


def _spec(a, kind, result):
    """Normalised analysis specification: only what changes the analysis (columns, baselines as a set, resolved
    settings). Cosmetic policy fields and how the run was invoked (policy file or flags) do not change it, so the
    ledger counts specifications correctly. The policy file's own hash is recorded separately."""
    from .core import resolve_null
    from .policy import spec_hash
    split = lambda s: sorted(c for c in (s or "").split(",") if c)
    st = result.get("settings", {})
    if kind == "compare":
        spec = {"kind": "compare", "old": a.old, "new": a.new, "label": a.label, "group": a.group,
                "baselines": split(a.baselines), "rank_within": a.rank_within,
                "settings": {"families": sorted(st.get("families", [])), "seeds": st.get("seeds"), "n_perm": st.get("n_perm"),
                             "perm_stop": st.get("perm_stop"), "null": resolve_null(a.null)}}
    else:
        keep = ("n_seeds", "n_perm", "perm_stop", "n_boot", "null", "score_transform", "risk_bins")
        s = {k: st[k] for k in keep if k in st}
        s["families"] = sorted(st.get("families", []))
        s["perm_strata"] = sorted(st.get("perm_strata", []) or [])
        s["interval"] = st.get("interval")
        s["minimum_gain"] = float(getattr(a, "min_gain", 0.0) or 0.0)
        spec = {"kind": "audit", "outcome": getattr(a, "outcome", "binary"), "score": a.score, "label": a.label,
                "group": a.group, "baselines": split(a.baselines), "rank_within": a.rank_within, "settings": s}
    return spec, spec_hash(spec)


def _validity(rep, df, group):
    from .envelope import prefix, status
    rep["validity"] = status(rep, df, group)
    msg = prefix(rep["validity"])
    if msg:
        print(("NOT TESTABLE: " if rep["validity"]["status"] == "not_testable" else "OUTSIDE VALIDATED RANGE: ") + msg)
    for n in rep["validity"]["notes"]:
        print("NOTE: " + n)


def _finish(kind, a, argv, df, doc, p_value, verdict, result):
    """Write the report and/or the evidence package. Neither changes the result."""
    v = result.get("validity")
    if v:
        from .envelope import prefix
        if prefix(v):
            doc["summary"] = prefix(v) + " " + doc.get("summary", "")
        doc.setdefault("record", []).insert(0, ("Validity status", v["status"].replace("_", " ") +
                                                (": " + "; ".join(v["reasons"]) if v["reasons"] else "")))
        for n in v["notes"]:
            doc["record"].insert(1, ("Note", n))
    from .report import provenance, render_html, write_report
    prov = provenance(argv if kind != "compare" else ["compare"] + list(argv), a.csv, a.map, len(df))
    doc["provenance"] = prov
    spec, h = _spec(a, kind, result)
    policy_name = __import__("pathlib").Path(a.policy).name if getattr(a, "policy", None) else None
    policy_doc, policy_h = (getattr(a, "policy_obj", None), getattr(a, "policy_hash", None))
    if not policy_name and doc.get("summary"):
        undeclared = lambda s: s.replace("the declared baselines (", "the baselines (not declared in a policy file: ")
        same = doc["verdict"][0] == doc["summary"]
        doc["summary"] = undeclared(doc["summary"])
        if same:
            doc["verdict"][0] = doc["summary"]
    if getattr(a, "evidence_dir", None):
        from .evidence import declaration_status, ledger_context, record_rows, write_package
        ctx = ledger_context(a.evidence_dir, prov["input_sha256"], h)
        decl = declaration_status(a.evidence_dir, policy_h, prov["input_sha256"])
        doc["record"] = record_rows(ctx, h, policy_name, p_value, policy_h, decl) + doc.get("record", [])
    elif policy_name:
        doc["record"] = [("Analysis specification hash (SHA-256)", h), ("Declared policy file", policy_name),
                         ("Policy file hash (SHA-256 of canonical JSON)", policy_h)] + doc.get("record", [])
    if a.report:
        print(f"report -> {write_report(doc, a.report)}")
    if getattr(a, "evidence_dir", None):
        d = write_package(a.evidence_dir, kind=kind, result=result, report_html=render_html(doc), spec=spec,
                          spec_sha256=h, policy_file=policy_name, provenance=prov, verdict=verdict,
                          policy=policy_doc, policy_sha256=policy_h)
        print(f"evidence -> {d}")


def plan_main(argv):
    from .planner import fmt_plan, from_pilot, plan
    p = argparse.ArgumentParser(prog="baseline-increment plan",
                                description="How many items to label: simulated power of the --fast audit for a target "
                                            "population increment (docs/prereg/PREREG_0.13.md, Part 2).")
    p.add_argument("--increment", type=float, required=True, help="population increment in AUROC to detect, e.g. 0.03")
    p.add_argument("--prevalence", type=float, help="share of positive labels")
    p.add_argument("--baseline-auroc", type=float, help="AUROC of the baselines alone")
    p.add_argument("--group-size", type=int, help="items per group (1 if ungrouped)")
    p.add_argument("--pilot", help="pilot CSV/Parquet/JSON to estimate prevalence, baselines-only AUROC and group size")
    p.add_argument("--label"); p.add_argument("--group"); p.add_argument("--baselines", default="")
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--max-items", type=int, default=20_000)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--json")
    a = p.parse_args(argv)
    est = {}
    if a.pilot:
        if not a.label:
            p.error("--pilot needs --label (and usually --group and --baselines)")
        est = from_pilot(read_table(a.pilot), a.label, a.group, [c for c in a.baselines.split(",") if c])
        print(f"from pilot: {est}")
    pi = a.prevalence if a.prevalence is not None else est.get("prevalence")
    a0 = a.baseline_auroc if a.baseline_auroc is not None else est.get("baselines_auroc")
    m = a.group_size if a.group_size is not None else est.get("group_size", 1)
    if pi is None or a0 is None:
        p.error("give --prevalence and --baseline-auroc, or --pilot")
    from .planner import default_grid
    res = plan(pi, a0, m, a.increment, power=a.power, grid=default_grid(pi, m, a.max_items), reps=a.reps, n_jobs=a.jobs)
    print(fmt_plan(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


def declare_main(argv):
    from pathlib import Path
    from .evidence import declare
    from .policy import load_policy
    from .report import _sha256
    p = argparse.ArgumentParser(prog="baseline-increment declare",
                                description="Record a policy in the run ledger before any run uses it, so reports can "
                                            "show it was declared first (docs/guide/assurance.md).")
    p.add_argument("policy")
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--data", help="bind the declaration to this input file (its SHA-256 is recorded)")
    a = p.parse_args(argv)
    pol, h = load_policy(a.policy)
    e = declare(a.evidence_dir, pol, h, Path(a.policy).name, _sha256(a.data) if a.data else None)
    print(f"declared policy {h} at {e['time_utc']} (ledger line {e['run']})")
    return 0


def verify_main(argv):
    from .evidence import verify
    p = argparse.ArgumentParser(prog="baseline-increment verify-evidence",
                                description="Check an evidence directory: the run ledger's hash chain and every manifest.")
    p.add_argument("evidence_dir")
    p.add_argument("--expect-head", help="SHA-256 of the last ledger line, recorded somewhere outside this directory; "
                                         "without it, 'intact' only means the ledger is internally consistent")
    a = p.parse_args(argv)
    ok, msgs = verify(a.evidence_dir, a.expect_head)
    print("\n".join(msgs))
    print("OK: ledger chain intact and every package matches its manifest" if ok else "FAILED: see above")
    return 0 if ok else 1


def fmt_primary(rep):
    ip, g = rep["incremental_prediction"], rep.get("gain_ci")
    ci = f"[{g['lower']:+.3f}, {g['upper']:+.3f}]" if g else "(no interval)"
    ci_info = rep["conditional_information"]
    return [f"PRIMARY  held-out gain beyond the baselines: {rep['gain']:+.3f} AUROC, 95% CI {ci}  ->  {ip['verdict']}",
            f"         ({ip['reason']})",
            f"SECONDARY (experimental)  conditional-information test: p = {ci_info['p']:.3f}, "
            f"adds_information = {str(ci_info['adds_information']).lower()}  [not used for any verdict]"]


def fmt(rep):
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "(no bootstrap)"
    lines = [f"baseline-increment {rep['version']}"] + fmt_primary(rep) + [
             f"items {rep['n_items']}  groups {rep['n_groups']}  positive rate {rep['positive_rate']:.3f}",
             f"detector AUROC            {rep['detector_auc']:.3f} {ci(rep['detector_auc_ci'])}"]
    for k, v in rep["baseline_auc"].items():
        lines.append(f"baseline {k:<17} {v:.3f}  (linear, alone)")
    for fam, v in rep["families"].items():
        star = "  <- headline (best baselines-only AUROC)" if fam == rep["headline_family"] else ""
        lines.append(f"family {fam:<7} baselines {v['baseline_auc']:.3f}  + detector {v['plus_detector_auc']:.3f}  "
                     f"increment {v['increment']:+.3f}{star}")
    lines.append(f"details: increment ({rep['headline_family']}) {rep['increment']:+.3f}  perm p={rep['increment_perm_p']:.3f}"
                 f"  adds_information={str(rep['adds_information']).lower()}")
    if rep.get("increment_ci"):
        note = rep.get("increment_ci_note", "not used for the decision")
        lines.append(f"  interval {ci(rep['increment_ci'])}  ({note})")
    for col, st in rep["strata"].items():
        lines.append(f"stratum '{col}': base rate alone {st['base_rate_only_auc']:.3f}")
        for v, e in st["within"].items():
            a = f"{e['detector_auc']:.3f} {ci(e['detector_auc_ci'])}" if "detector_auc" in e else "n/a"
            lines.append(f"  {v:<20} n={e['n']:<5} pos={e['positive_rate']:.2f}  detector AUROC {a}")
    lines += [f"WARNING: {w}" for w in rep["warnings"]]
    return "\n".join(lines)


def check_main(argv):
    from .checks import BENCHMARKS, check, to_html, to_text
    p = argparse.ArgumentParser(prog="baseline-increment check",
                                description="Try to break a detector's claim with cheap alternative explanations.")
    p.add_argument("csv")
    p.add_argument("--score", required=True, help="detector score, higher = more likely the flagged event")
    p.add_argument("--label", required=True, help="1 = the event the detector should flag (e.g. hallucinated)")
    p.add_argument("--question", help="question / source id; rows sharing it never straddle folds")
    p.add_argument("--model", help="which model produced the row (enables the same-question test)")
    p.add_argument("--response", help="response text (length is derived from it)")
    p.add_argument("--length", help="a length column, if there is no response text")
    p.add_argument("--task", help="task / dataset / source column")
    p.add_argument("--difficulty", default="", help="comma-separated difficulty columns you already have")
    p.add_argument("--benchmark", choices=BENCHMARKS, help="join the shipped difficulty table for this benchmark")
    p.add_argument("--question-text", help="question text column, used to join --benchmark")
    p.add_argument("--min-increment", type=float, default=0.0)
    p.add_argument("--boot", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--families", default="linear,spline", help="baseline families (gbm is slow)")
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    p.add_argument("--samples", default="", help="comma-separated 0/1 'wrong' labels of repeated sampled answers (>= 4) for the "
                                                 "error-propensity decomposition; needs --model and --question")
    p.add_argument("--panel-meta", help="JSON with the reference panel behind --difficulty (models, n_models, label_source)")
    p.add_argument("--json", help="write the full report as JSON")
    p.add_argument("--html", help="write a self-contained HTML report")
    p.add_argument("--map", help="JSON mapping file: rename / derive columns from an export (see tables.py)")
    a = p.parse_args(argv)
    df = read_table(a.csv, a.map)
    rep = check(df, a.score, a.label, question=a.question, model=a.model, response=a.response, length=a.length, task=a.task,
                difficulty=[c for c in a.difficulty.split(",") if c], benchmark=a.benchmark, question_text=a.question_text,
                min_increment=a.min_increment, n_boot=a.boot, n_perm=a.perms, seeds=a.seeds,
                families=[c for c in a.families.split(",") if c], n_jobs=a.jobs,
                samples=[c for c in a.samples.split(",") if c], panel_meta=json.load(open(a.panel_meta)) if a.panel_meta else None)
    rep["settings"]["command"] = " ".join(["baseline-increment", "check"] + list(argv))
    import hashlib
    rep["input_sha256"] = hashlib.sha256(open(a.csv, "rb").read()).hexdigest()
    print(to_text(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)
    if a.html:
        open(a.html, "w").write(to_html(rep))


def ordinal_main(a, argv):
    from .ordinal import audit_ordinal, fmt_ordinal
    split = lambda s: [c for c in s.split(",") if c]
    bad = [f for f, v in (("--strata", a.strata), ("--perm-strata", a.perm_strata), ("--calibration", a.calibration),
                          ("--interval", a.interval not in (None, "bootstrap")),
                          ("--null permute", a.null == "permute")) if v]
    if bad:
        raise SystemExit(f"--outcome ordinal does not support {', '.join(bad)} (see docs/prereg/PREREG_0.11.md)")
    df = read_table(a.csv, a.map)
    if a.fast:
        a.seeds, a.perms, a.perm_stop = 2, 200, 10
    rep = audit_ordinal(df, a.score, a.label, a.group, split(a.baselines), families=split(a.families), seeds=range(a.seeds),
                        n_perm=a.perms, perm_stop=a.perm_stop, score_transform="raw" if a.raw_score else "rank",
                        rank_within=a.rank_within, n_jobs=a.jobs, null=a.null)
    print(fmt_ordinal(rep))
    _validity(rep, df, a.group)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.report or a.evidence_dir:
        from .report import ordinal_document
        _finish("audit", a, argv, df, ordinal_document(rep, {}), rep["increment_perm_p"], "DESCRIPTIVE", rep)
    if a.fail_if_no_increment:
        print("error: --fail-if-no-increment needs a validated interval for the gain, which graded (ordinal) labels do "
              "not have yet; the ordinal result is descriptive", file=sys.stderr)
        return 2
    return 0


def compare_main(argv):
    from .compare import compare, fmt_compare
    p = argparse.ArgumentParser(prog="baseline-increment compare",
                                description="Compare two versions of a detector score on the same items: improvement, "
                                            "regression and standalone AUROC (docs/prereg/PREREG_0.8.md).")
    p.add_argument("csv")
    p.add_argument("--old", required=True, help="score column of the current version (v1)")
    p.add_argument("--new", required=True, help="score column of the candidate version (v2)")
    p.add_argument("--label", required=True, help="1 = the event the detector should flag")
    p.add_argument("--group", help="items sharing a group never straddle train/test")
    p.add_argument("--baselines", default="", help="comma-separated baseline columns")
    p.add_argument("--rank-within", help="rank both scores within this column (e.g. model)")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--perm-stop", type=int, default=10)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--map", help="JSON mapping file: rename / derive columns from an export (see tables.py)")
    p.add_argument("--null", choices=["residual", "residual_sl", "residual_gbm", "permute"], default="residual")
    p.add_argument("--fail-on-regression", action="store_true",
                   help="exit with status 1 when v1 keeps information that v2 lacks")
    p.add_argument("--json", help="write the full report as JSON")
    p.add_argument("--report", help="write a comparison report: .html (printable to PDF) or .md")
    p.add_argument("--evidence-dir", help="write an evidence package and append to the run ledger in this directory")
    a = p.parse_args(argv)
    df = read_table(a.csv, a.map)
    rep = compare(df, a.old, a.new, a.label, a.group, [c for c in a.baselines.split(",") if c], rank_within=a.rank_within,
                  families=[c for c in a.families.split(",") if c], seeds=range(a.seeds), n_perm=a.perms,
                  perm_stop=a.perm_stop or None, n_jobs=a.jobs, null=a.null)
    print(fmt_compare(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)
    if a.report or a.evidence_dir:
        import numpy as np
        from .core import estimate, split_design, tpr_at_fpr, transform_score
        from .report import compare_document
        y = np.asarray(df[a.label], int)
        s1 = transform_score(df, a.old, "rank", a.rank_within)
        s2 = transform_score(df, a.new, "rank", a.rank_within)
        ops = [{"fpr": f, "tpr_old": tpr_at_fpr(y, s1, f), "tpr_new": tpr_at_fpr(y, s2, f)} for f in (0.05, 0.10)]
        groups = np.asarray(df[a.group]).astype(str) if a.group else np.arange(len(y)).astype(str)
        num, lin = split_design(df, [c for c in a.baselines.split(",") if c])
        fams = [c for c in a.families.split(",") if c]
        versions = {}
        for k, sv in (("v1", s1), ("v2", s2)):
            est, head = estimate(num, lin, sv, y, groups, fams, range(a.seeds))
            versions[k] = {"family": head, "baselines_auc": est[head]["baseline_auc"], "increment": est[head]["increment"]}
        doc = compare_document(rep, ops, {}, versions=versions)
        verdict = ("improvement" if rep["improvement"]["detected"] else "no_improvement") + \
                  ("+regression" if rep["regression"]["detected"] else "")
        _finish("compare", a, argv, df, doc, None, verdict, rep)
    if a.fail_on_regression and rep["regression"]["detected"]:
        print("FAIL: v1 keeps information that v2 lacks (regression detected)")
        return 1
    return 0


def adapter_main(kind, argv):
    from .adapters import from_inspect, from_lm_eval, list_inspect
    p = argparse.ArgumentParser(prog=f"baseline-increment {kind}",
                                description="Convert harness logs into the per-item CSV the audit reads.")
    p.add_argument("logs", nargs="+", help="lm-eval samples_*.jsonl files, or Inspect logs / directories of logs")
    p.add_argument("-o", "--out", help="output CSV (or .parquet)")
    if kind == "from-lm-eval":
        p.add_argument("--metric", default="acc")
    else:
        p.add_argument("--list", action="store_true", help="list scorers (value types, dict keys) and metadata fields, then exit")
        p.add_argument("--label-scorer", help="scorer whose value says correct / incorrect ('name' or 'name:key')")
        p.add_argument("--score-scorer", help="numeric scorer used as the detector score ('name' or 'name:key')")
        p.add_argument("--score-metadata", help="numeric sample metadata field used as the detector score")
        p.add_argument("--group-by", choices=["item", "file"], default="item",
                       help="item (default): one group per task:sample id across models and epochs; file: pre-0.9 behaviour")
        p.add_argument("--partial-as", choices=["drop", "wrong", "correct"], default="drop", help="label for 'P' (partial) scores")
        p.add_argument("--noanswer-as", choices=["wrong", "correct", "drop"], default="wrong", help="label for 'N' (no answer) scores")
    p.add_argument("--lomo", action="store_true",
                   help="logs from several models on the same task: add leave-one-model-out difficulty "
                        "(lomo_share_wrong) and write <out>.panel.json for check --panel-meta")
    a = p.parse_args(argv)
    if kind == "from-inspect" and a.list:
        print(json.dumps(list_inspect(a.logs), indent=1))
        return 0
    if not a.out:
        p.error("-o/--out is required")
    if kind == "from-inspect" and not a.label_scorer:
        p.error("--label-scorer is required (use --list to see the scorers)")
    df = (from_lm_eval(a.logs, a.metric) if kind == "from-lm-eval"
          else from_inspect(a.logs, a.label_scorer, a.score_scorer, a.score_metadata, group_by=a.group_by,
                            partial_as=a.partial_as, noanswer_as=a.noanswer_as))
    dropped = df.attrs.get("dropped", {})
    for reason, n in dropped.items():
        print(f"dropped {n} rows: {reason}")
    if df.empty:
        raise SystemExit("no usable rows")
    if a.lomo:
        from .adapters import add_lomo_difficulty
        df, meta = add_lomo_difficulty(df)
        if df.empty:
            raise SystemExit("--lomo: no item was answered by two or more models; pass logs from several models on one task")
        meta["dropped_by_reason"] = dropped
        json.dump(meta, open(a.out + ".panel.json", "w"), indent=1)
        print(f"own panel: {len(meta['models'])} models, {meta['n_models']} reference models per row, "
              f"{meta['dropped_rows']} rows dropped (item answered by one model only) -> {a.out}.panel.json")
        if meta["n_models"] < 5:
            print(f"note: {meta['n_models']} reference models; increments over difficulty shrink as the panel grows, "
                  "so treat the increment as an upper bound")
    if str(a.out).endswith(".parquet"):
        df.to_parquet(a.out, index=False)
    else:
        df.to_csv(a.out, index=False)
    print(f"{len(df)} rows, {df['group'].nunique()} groups, error rate {df['wrong'].mean():.3f} -> {a.out}")
    return 0


def main(argv=None):
    """Exit codes: 0 success; 1 a gate failed (--fail-if-no-increment, --fail-on-regression); 2 a usage or input
    error (bad flags, missing columns, labels of the wrong kind, unreadable files)."""
    try:
        return _main(argv)
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            raise
        print(f"error: {e.code}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "check":
        return check_main(argv[1:])
    if argv and argv[0] == "compare":
        return compare_main(argv[1:])
    if argv and argv[0] in ("from-lm-eval", "from-inspect"):
        return adapter_main(argv[0], argv[1:])
    if argv and argv[0] == "verify-evidence":
        return verify_main(argv[1:])
    if argv and argv[0] == "declare":
        return declare_main(argv[1:])
    if argv and argv[0] == "plan":
        return plan_main(argv[1:])
    p = argparse.ArgumentParser(prog="baseline-increment", description=__doc__)
    p.add_argument("csv")
    p.add_argument("--score", help="detector score column (required unless --policy declares it)")
    p.add_argument("--label", help="1 = the event the detector should flag (required unless --policy declares it)")
    p.add_argument("--policy", help="declared evaluation policy (JSON, or YAML with PyYAML): fixes score, label, group, "
                                    "baselines, outcome and preset; see docs/guide/assurance.md")
    p.add_argument("--evidence-dir", help="write an evidence package and append to the run ledger in this directory")
    p.add_argument("--group", help="items sharing a group never straddle train/test")
    p.add_argument("--baselines", default="", help="comma-separated columns")
    p.add_argument("--strata", default="", help="comma-separated columns to break the AUROC down by")
    p.add_argument("--perm-strata", default="", help="comma-separated columns; permute only within their levels")
    p.add_argument("--families", default=",".join(FAMILIES), help="baseline families: linear,spline,gbm")
    p.add_argument("--rank-within", help="rank the score within this column (e.g. model)")
    p.add_argument("--raw-score", action="store_true", help="use the score as given instead of its rank")
    p.add_argument("--seeds", type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument("--perms", type=int, default=200, help=argparse.SUPPRESS)
    p.add_argument("--boot", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    p.add_argument("--min-gain", type=float, default=0.0,
                   help="minimum gain in AUROC that the lower 95%% bound must exceed for PASS (default 0)")
    p.add_argument("--perm-stop", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--interval", choices=["delong", "bootstrap", "half"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--null", choices=["residual", "residual_sl", "residual_gbm", "permute"], default="residual",
                   help=argparse.SUPPRESS)
    p.add_argument("--fast", action="store_true",
                   help="the validated default preset (kept for compatibility; it is now the default)")
    p.add_argument("--fail-if-no-increment", action="store_true",
                   help="CI gate: exit 1 unless the lower 95%% bound of the gain exceeds --min-gain (verdict PASS), "
                        "or when the data are not testable")
    p.add_argument("--json", help="also write the full report here")
    p.add_argument("--map", help="JSON mapping file: rename / derive columns from an export (see tables.py)")
    p.add_argument("--report", help="write an audit report: .html (printable to PDF) or .md")
    p.add_argument("--outcome", choices=["binary", "ordinal"], default="binary",
                   help="binary (0/1 label, default) or ordinal (graded label with 3-10 integer levels, e.g. severity 0-3)")
    p.add_argument("--ladder", action="store_true",
                   help="report the increment as the baselines are added one at a time, in the order given (descriptive)")
    p.add_argument("--text", help="text column (question or response) to check for exact duplicates across groups")
    p.add_argument("--sensitivity", action="store_true",
                   help="omitted-baseline sensitivity analysis when the detector adds information (docs/prereg/PREREG_0.13.md)")
    p.add_argument("--calibration", action="store_true",
                   help="also describe how well the raw score works as a probability (binary labels, score in [0, 1])")
    a = p.parse_args(argv)
    a.ops, a.policy_obj, a.policy_hash = (0.05, 0.10), None, None
    if a.policy:
        a.policy_obj, a.policy_hash = _apply_policy(a, argv)
    if not a.score or not a.label:
        p.error("--score and --label are required (or declare them in --policy)")
    if a.outcome == "ordinal":
        return ordinal_main(a, argv)
    if a.fast:
        a.seeds, a.boot, a.perms, a.perm_stop = 2, 0, 200, 10
    if a.interval is None:
        a.interval = DEFAULT_INTERVAL
    df = read_table(a.csv, a.map)
    split = lambda s: [c for c in s.split(",") if c]
    rep = audit(df, a.score, a.label, a.group, split(a.baselines), split(a.strata), families=split(a.families),
                seeds=range(a.seeds), n_perm=a.perms, n_boot=a.boot, perm_strata=split(a.perm_strata),
                score_transform="raw" if a.raw_score else "rank", rank_within=a.rank_within, n_jobs=a.jobs,
                perm_stop=a.perm_stop, interval=a.interval, null=a.null, min_gain=a.min_gain)
    print(fmt(rep))
    _validity(rep, df, a.group)
    if a.calibration:
        from .calibration import fmt_calibration, score_calibration
        rep["calibration"] = score_calibration(df, a.score, a.label, a.group)
        print(fmt_calibration(rep["calibration"]))
    if a.sensitivity and rep["incremental_prediction"]["verdict"] == "PASS":
        from .sensitivity import sensitivity, sentence
        rep["sensitivity"] = sensitivity(df, a.score, a.label, a.group, split(a.baselines), families=split(a.families),
                                         seeds=range(a.seeds), null=a.null,
                                         score_transform="raw" if a.raw_score else "rank", rank_within=a.rank_within,
                                         headline=rep["headline_family"])
        print(sentence(rep["sensitivity"]))
    if a.report or a.evidence_dir or a.ladder:
        from .descriptive import ladder, precision_recall
        rep["precision_recall"] = precision_recall(df, a.score, a.label, a.group, split(a.baselines), rep["headline_family"],
                                                   range(a.seeds), "raw" if a.raw_score else "rank", a.rank_within)
        if a.ladder:
            rep["ladder"] = ladder(df, a.score, a.label, a.group, split(a.baselines), split(a.families), range(a.seeds),
                                   "raw" if a.raw_score else "rank", a.rank_within)
            for r in rep["ladder"]:
                print(f"ladder  baselines [{', '.join(r['baselines']) or 'none'}]  increment {r['increment']:+.3f} ({r['family']})")
    if a.text:
        from .descriptive import duplicates
        rep["duplicates"] = duplicates(df, a.text, a.group)
        d = rep["duplicates"]
        if d.get("duplicate_texts_across_groups"):
            msg = (f"{d['duplicate_texts_across_groups']} texts in '{a.text}' appear in more than one group "
                   f"({d['rows_affected']} rows): grouped CV cannot keep them apart")
            rep["warnings"].append(msg)
            print("WARNING: " + msg)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.report or a.evidence_dir:
        from .core import operating_point
        from .report import audit_document
        ops = [operating_point(df, a.score, a.label, a.group, split(a.baselines), family=rep["headline_family"], fpr=f,
                               seeds=range(a.seeds), score_transform="raw" if a.raw_score else "rank", rank_within=a.rank_within)
               for f in a.ops]
        _finish("audit", a, argv, df, audit_document(rep, ops, {}), rep["increment_perm_p"],
                rep["incremental_prediction"]["verdict"], rep)
    if a.fail_if_no_increment and rep["validity"]["status"] == "not_testable":
        print("FAIL: not testable on this data (" + "; ".join(rep["validity"]["reasons"]) + ")")
        return 1
    if a.fail_if_no_increment and rep["incremental_prediction"]["verdict"] != "PASS":
        print(f"FAIL: {rep['incremental_prediction']['verdict']}: {rep['incremental_prediction']['reason']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
