"""baseline-increment judge: does an LLM judge add anything beyond cheap baselines?

    baseline-increment judge alignment.csv --judge judge_score --label human_label \
        --baselines task,difficulty --response response_text --group item_id

Input is the labelled set a team already builds to align a judge with human labels: one row per judged item, the judge's
score or verdict, the human label (0/1), and cheap columns that need no judge (task, length, difficulty proxy, model).
Output puts the number teams usually look at (judge-human agreement) next to the one this tool adds: the held-out gain
of the judge over the baselines alone, with the validated 95% interval and the PASS / INCONCLUSIVE / FAIL verdict.

This is a thin wrapper over ``audit``: the gain, interval and verdict are the production primary result (validated
preset: 2 seeds, three families, clustered DeLong). The secondary conditional-information test is not run here.
"""
import argparse
import json

import numpy as np

from .core import FAMILIES, audit
from .tables import read_table

LENGTH_COL = "response_length_words"


def agreement(judge, label):
    """Raw agreement and Cohen's kappa when the judge gives 0/1 verdicts; None otherwise."""
    j = np.asarray(judge, float)
    if not set(np.unique(j)) <= {0.0, 1.0}:
        return None
    y = np.asarray(label, float)
    po = float((j == y).mean())
    pe = float(j.mean() * y.mean() + (1 - j.mean()) * (1 - y.mean()))
    return {"agreement": po, "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}


def sentence(rep, baselines):
    v = rep["incremental_prediction"]["verdict"]
    lo, hi = rep["gain_ci"]["lower"], rep["gain_ci"]["upper"]
    mg = rep["incremental_prediction"]["minimum_gain"]
    b = ", ".join(baselines)
    if v == "PASS":
        return (f"The judge improves held-out prediction of the human label beyond [{b}] by {rep['gain']:+.3f} AUROC "
                f"(95% interval {lo:+.3f} to {hi:+.3f}).")
    if v == "FAIL":
        return (f"The judge does not add {mg:+.3f} AUROC beyond [{b}]: the whole 95% interval ({lo:+.3f} to {hi:+.3f}) "
                f"lies below it.")
    return (f"This data cannot show whether the judge adds anything beyond [{b}]: the 95% interval ({lo:+.3f} to "
            f"{hi:+.3f}) includes {mg:+.3f}.")


def judge_report(df, judge, label, baselines, group=None, response=None, min_gain=0.0, n_jobs=-1):
    baselines = list(baselines)
    if response:
        df = df.copy()
        df[LENGTH_COL] = df[response].astype(str).str.split().str.len()
        baselines.append(LENGTH_COL)
    if not baselines:
        raise ValueError("declare at least one baseline (--baselines) or a --response column to derive length from")
    rep = audit(df, judge, label, group, baselines, families=FAMILIES, seeds=range(2), n_perm=1, perm_stop=None,
                n_boot=0, interval="delong", null="residual_gbm", min_gain=min_gain, n_jobs=n_jobs)
    rep["conditional_information"] = {"status": "not run in judge mode"}
    rep["judge"] = {"column": judge, "label": label, "baselines": baselines,
                    "agreement": agreement(df[judge], df[label]),
                    "judge_alone_auroc": rep["detector_auc"], "judge_alone_auroc_ci": rep["detector_auc_ci"],
                    "baselines_alone_auroc": rep["baselines_combined_auc"],
                    "baselines_plus_judge_auroc": rep["baselines_plus_detector_auc"],
                    "sentence": sentence(rep, baselines)}
    return rep


def fmt_judge(rep):
    j, ip = rep["judge"], rep["incremental_prediction"]
    out = [f"Items {rep['n_items']}  groups {rep['n_groups']}  human-label positive rate {rep['positive_rate']:.3f}", ""]
    if j["agreement"]:
        out.append(f"Judge-human agreement      {j['agreement']['agreement']:.3f}   (Cohen's kappa {j['agreement']['cohen_kappa']:.3f})")
    lo, hi = j["judge_alone_auroc_ci"] if j["judge_alone_auroc_ci"] else (float("nan"), float("nan"))
    out += [f"Judge alone, AUROC         {j['judge_alone_auroc']:.3f}   (95% CI {lo:.3f} to {hi:.3f})",
            f"Baselines alone, AUROC     {j['baselines_alone_auroc']:.3f}   ({rep['headline_family']}, held out)",
            f"Baselines + judge, AUROC   {j['baselines_plus_judge_auroc']:.3f}",
            "",
            f"Judge's incremental gain   {rep['gain']:+.3f}   (95% CI {rep['gain_ci']['lower']:+.3f} to {rep['gain_ci']['upper']:+.3f})",
            f"Verdict                    {ip['verdict']}   ({ip['reason']})",
            "", j["sentence"]]
    if j["baselines_alone_auroc"] >= j["judge_alone_auroc"]:
        out.append(f"Note: the baselines alone ({j['baselines_alone_auroc']:.3f}) match or beat the judge alone "
                   f"({j['judge_alone_auroc']:.3f}).")
    out.append("Baselines declared: " + ", ".join(j["baselines"]) + ". The answer is relative to these columns.")
    return "\n".join(out)


def judge_main(argv):
    p = argparse.ArgumentParser(prog="baseline-increment judge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--judge", required=True, help="judge score or 0/1 verdict column")
    p.add_argument("--label", required=True, help="human label column (1 = the property the judge should flag)")
    p.add_argument("--baselines", default="", help="comma-separated cheap columns (task, difficulty, model, ...)")
    p.add_argument("--response", help="response text column: its length in words is added as a baseline")
    p.add_argument("--group", help="items sharing a group (e.g. the same prompt) never straddle train/test")
    p.add_argument("--min-gain", type=float, default=0.0, help="gain the lower 95%% bound must exceed for PASS")
    p.add_argument("--map", help="JSON mapping file to rename / derive columns (see tables.py)")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--json", help="also write the full result here")
    a = p.parse_args(argv)
    df = read_table(a.csv, a.map)
    rep = judge_report(df, a.judge, a.label, [c for c in a.baselines.split(",") if c], a.group, a.response,
                       a.min_gain, a.jobs)
    from .envelope import prefix, status
    rep["validity"] = status(rep, df, a.group)
    msg = prefix(rep["validity"])
    if msg:
        print(("NOT TESTABLE: " if rep["validity"]["status"] == "not_testable" else "OUTSIDE VALIDATED RANGE: ") + msg)
    print(fmt_judge(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)
    return 0
