"""Command line: baseline-increment data.csv --score s --label y --group g --baselines a,b --strata c"""
import argparse
import json
import sys

import pandas as pd

from .core import FAMILIES, audit


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
    a = p.parse_args(argv)
    df = pd.read_csv(a.csv)
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
    p.add_argument("--fail-on-regression", action="store_true",
                   help="exit with status 1 when v1 keeps information that v2 lacks")
    p.add_argument("--json", help="write the full report as JSON")
    a = p.parse_args(argv)
    df = pd.read_csv(a.csv)
    rep = compare(df, a.old, a.new, a.label, a.group, [c for c in a.baselines.split(",") if c], rank_within=a.rank_within,
                  families=[c for c in a.families.split(",") if c], seeds=range(a.seeds), n_perm=a.perms,
                  perm_stop=a.perm_stop or None, n_jobs=a.jobs)
    print(fmt_compare(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)
    if a.fail_on_regression and rep["regression"]["detected"]:
        print("FAIL: v1 keeps information that v2 lacks (regression detected)")
        return 1
    return 0


def adapter_main(kind, argv):
    from .adapters import from_inspect, from_lm_eval
    p = argparse.ArgumentParser(prog=f"baseline-increment {kind}",
                                description="Convert harness logs into the per-item CSV the audit reads.")
    p.add_argument("logs", nargs="+", help="lm-eval samples_*.jsonl files, or Inspect logs")
    p.add_argument("-o", "--out", required=True)
    if kind == "from-lm-eval":
        p.add_argument("--metric", default="acc")
    else:
        p.add_argument("--label-scorer", required=True, help="scorer whose value says correct / incorrect")
        p.add_argument("--score-scorer", help="numeric scorer used as the detector score")
        p.add_argument("--score-metadata", help="numeric sample metadata field used as the detector score")
    p.add_argument("--lomo", action="store_true",
                   help="logs from several models on the same task: add leave-one-model-out difficulty "
                        "(lomo_share_wrong) and write <out>.panel.json for check --panel-meta")
    a = p.parse_args(argv)
    df = (from_lm_eval(a.logs, a.metric) if kind == "from-lm-eval"
          else from_inspect(a.logs, a.label_scorer, a.score_scorer, a.score_metadata))
    if a.lomo:
        from .adapters import add_lomo_difficulty
        df, meta = add_lomo_difficulty(df)
        if df.empty:
            raise SystemExit("--lomo: no item was answered by two or more models; pass logs from several models on one task")
        json.dump(meta, open(a.out + ".panel.json", "w"), indent=1)
        print(f"own panel: {len(meta['models'])} models, {meta['n_models']} reference models per row, "
              f"{meta['dropped_rows']} rows dropped (item answered by one model only) -> {a.out}.panel.json")
        if meta["n_models"] < 5:
            print(f"note: {meta['n_models']} reference models; increments over difficulty shrink as the panel grows, "
                  "so treat the increment as an upper bound")
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
    a = p.parse_args(argv)
    if a.fast:
        a.seeds, a.boot, a.perms, a.perm_stop = 2, 0, 200, 10
    df = pd.read_csv(a.csv)
    split = lambda s: [c for c in s.split(",") if c]
    rep = audit(df, a.score, a.label, a.group, split(a.baselines), split(a.strata), families=split(a.families),
                seeds=range(a.seeds), n_perm=a.perms, n_boot=a.boot, perm_strata=split(a.perm_strata),
                score_transform="raw" if a.raw_score else "rank", rank_within=a.rank_within, n_jobs=a.jobs,
                perm_stop=a.perm_stop, interval=a.interval)
    print(fmt(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.fail_if_no_increment and not rep["adds_information"]:
        print("FAIL: the detector does not add information beyond the baselines (adds_information = false)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
