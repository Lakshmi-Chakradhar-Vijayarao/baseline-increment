"""Command line: baseline-increment data.csv --score s --label y --group g --baselines a,b --strata c"""
import argparse
import json
import sys


from .core import FAMILIES, audit
from .tables import read_table


def fmt(rep):
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "(no bootstrap)"
    lines = [f"baseline-increment {rep['version']}",
             f"items {rep['n_items']}  groups {rep['n_groups']}  positive rate {rep['positive_rate']:.3f}",
             f"detector AUROC            {rep['detector_auc']:.3f} {ci(rep['detector_auc_ci'])}"]
    for k, v in rep["baseline_auc"].items():
        lines.append(f"baseline {k:<17} {v:.3f}  (linear, alone)")
    for fam, v in rep["families"].items():
        star = "  <- headline (best baselines-only AUROC)" if fam == rep["headline_family"] else ""
        lines.append(f"family {fam:<7} baselines {v['baseline_auc']:.3f}  + detector {v['plus_detector_auc']:.3f}  "
                     f"increment {v['increment']:+.3f}{star}")
    lines.append(f"increment ({rep['headline_family']})      {rep['increment']:+.3f}  perm p={rep['increment_perm_p']:.3f}"
                 f"  adds_information={str(rep['adds_information']).lower()}")
    if rep.get("increment_ci"):
        lines.append(f"  interval {ci(rep['increment_ci'])}  (conservative; not used for the decision)")
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
    p.add_argument("--min-increment", type=float, default=0.02)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
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
                          ("--interval", a.interval != "bootstrap")) if v]
    if bad:
        raise SystemExit(f"--outcome ordinal does not support {', '.join(bad)} (see docs/PREREG_0.11.md)")
    df = read_table(a.csv, a.map)
    if a.fast:
        a.seeds, a.perms, a.perm_stop = 2, 200, 10
    rep = audit_ordinal(df, a.score, a.label, a.group, split(a.baselines), families=split(a.families), seeds=range(a.seeds),
                        n_perm=a.perms, perm_stop=a.perm_stop, score_transform="raw" if a.raw_score else "rank",
                        rank_within=a.rank_within, n_jobs=a.jobs)
    print(fmt_ordinal(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.report:
        from .report import ordinal_document, provenance, write_report
        print(f"report -> {write_report(ordinal_document(rep, provenance(argv, a.csv, a.map, len(df))), a.report)}")
    if a.fail_if_no_increment and not rep["adds_information"]:
        print("FAIL: the detector does not add information beyond the baselines (adds_information = false)")
        return 1
    return 0


def compare_main(argv):
    from .compare import compare, fmt_compare
    p = argparse.ArgumentParser(prog="baseline-increment compare",
                                description="Compare two versions of a detector score on the same items: improvement, "
                                            "regression and standalone AUROC (docs/PREREG_0.8.md).")
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
    p.add_argument("--fail-on-regression", action="store_true",
                   help="exit with status 1 when v1 keeps information that v2 lacks")
    p.add_argument("--json", help="write the full report as JSON")
    p.add_argument("--report", help="write a comparison report: .html (printable to PDF) or .md")
    a = p.parse_args(argv)
    df = read_table(a.csv, a.map)
    rep = compare(df, a.old, a.new, a.label, a.group, [c for c in a.baselines.split(",") if c], rank_within=a.rank_within,
                  families=[c for c in a.families.split(",") if c], seeds=range(a.seeds), n_perm=a.perms,
                  perm_stop=a.perm_stop or None, n_jobs=a.jobs)
    print(fmt_compare(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)
    if a.report:
        import numpy as np
        from .core import tpr_at_fpr, transform_score
        from .report import compare_document, provenance, write_report
        y = np.asarray(df[a.label], int)
        s1 = transform_score(df, a.old, "rank", a.rank_within)
        s2 = transform_score(df, a.new, "rank", a.rank_within)
        ops = [{"fpr": f, "tpr_old": tpr_at_fpr(y, s1, f), "tpr_new": tpr_at_fpr(y, s2, f)} for f in (0.05, 0.10)]
        doc = compare_document(rep, ops, provenance(["compare"] + list(argv), a.csv, a.map, len(df)))
        print(f"report -> {write_report(doc, a.report)}")
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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "check":
        return check_main(argv[1:])
    if argv and argv[0] == "compare":
        return compare_main(argv[1:])
    if argv and argv[0] in ("from-lm-eval", "from-inspect"):
        return adapter_main(argv[0], argv[1:])
    p = argparse.ArgumentParser(prog="baseline-increment", description=__doc__)
    p.add_argument("csv")
    p.add_argument("--score", required=True)
    p.add_argument("--label", required=True, help="1 = the event the detector should flag")
    p.add_argument("--group", help="items sharing a group never straddle train/test")
    p.add_argument("--baselines", default="", help="comma-separated columns")
    p.add_argument("--strata", default="", help="comma-separated columns to break the AUROC down by")
    p.add_argument("--perm-strata", default="", help="comma-separated columns; permute only within their levels")
    p.add_argument("--families", default=",".join(FAMILIES), help="baseline families: linear,spline,gbm")
    p.add_argument("--rank-within", help="rank the score within this column (e.g. model)")
    p.add_argument("--raw-score", action="store_true", help="use the score as given instead of its rank")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--boot", type=int, default=200, help="refit bootstrap replicates (0 = skip)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--perm-stop", type=int, default=None,
                   help="stop permuting once this many null increments reach the observed one (Besag-Clifford)")
    p.add_argument("--interval", choices=["bootstrap", "half"], default="bootstrap",
                   help="increment interval: refit group bootstrap (conservative) or half-sample refits")
    p.add_argument("--fast", action="store_true",
                   help="CI preset: 2 seeds, no interval refits, up to 200 permutations with stopping at 10 "
                        "(overrides --seeds, --boot, --perms, --perm-stop; see CALIBRATION.md)")
    p.add_argument("--fail-if-no-increment", action="store_true",
                   help="exit with status 1 when adds_information is false (for CI gates)")
    p.add_argument("--json", help="also write the full report here")
    p.add_argument("--map", help="JSON mapping file: rename / derive columns from an export (see tables.py)")
    p.add_argument("--report", help="write an audit report: .html (printable to PDF) or .md")
    p.add_argument("--outcome", choices=["binary", "ordinal"], default="binary",
                   help="binary (0/1 label, default) or ordinal (graded label with 3-10 integer levels, e.g. severity 0-3)")
    p.add_argument("--calibration", action="store_true",
                   help="also describe how well the raw score works as a probability (binary labels, score in [0, 1])")
    a = p.parse_args(argv)
    if a.outcome == "ordinal":
        return ordinal_main(a, argv)
    if a.fast:
        a.seeds, a.boot, a.perms, a.perm_stop = 2, 0, 200, 10
    df = read_table(a.csv, a.map)
    split = lambda s: [c for c in s.split(",") if c]
    rep = audit(df, a.score, a.label, a.group, split(a.baselines), split(a.strata), families=split(a.families),
                seeds=range(a.seeds), n_perm=a.perms, n_boot=a.boot, perm_strata=split(a.perm_strata),
                score_transform="raw" if a.raw_score else "rank", rank_within=a.rank_within, n_jobs=a.jobs,
                perm_stop=a.perm_stop, interval=a.interval)
    print(fmt(rep))
    if a.calibration:
        from .calibration import fmt_calibration, score_calibration
        rep["calibration"] = score_calibration(df, a.score, a.label, a.group)
        print(fmt_calibration(rep["calibration"]))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.report:
        from .core import operating_point
        from .report import audit_document, provenance, write_report
        ops = [operating_point(df, a.score, a.label, a.group, split(a.baselines), family=rep["headline_family"], fpr=f,
                               seeds=range(a.seeds), score_transform="raw" if a.raw_score else "rank", rank_within=a.rank_within)
               for f in (0.05, 0.10)]
        doc = audit_document(rep, ops, provenance(argv, a.csv, a.map, len(df)))
        print(f"report -> {write_report(doc, a.report)}")
    if a.fail_if_no_increment and not rep["adds_information"]:
        print("FAIL: the detector does not add information beyond the baselines (adds_information = false)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
