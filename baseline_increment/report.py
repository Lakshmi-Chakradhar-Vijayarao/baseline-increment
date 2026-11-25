"""Audit-grade reports for `audit` and `compare` (0.10.0).

A report is a document someone can file and sign off: a verdict in fixed sentences tied to the decision rules,
the key numbers, what the result means at fixed false-positive rates, the calibration evidence for the settings
actually used, provenance (input hash, command, versions, seeds, time), and what was not tested.

Every piece of text from the data or the command line is escaped. Nothing in a report is generated free-form.
"""
import datetime as _dt
import hashlib
import html
import json
import platform
import sys
from pathlib import Path

NOT_TESTED = [
    "Leakage or test-set tuning: whether the detector or its threshold was fitted on these items.",
    "Out-of-distribution behaviour: other tasks, models, prompts or time periods.",
    "Label quality: the labels are taken as given; grader errors pass straight into every number here.",
    "Downstream usefulness: cost, latency, or what happens after an item is flagged.",
    "Baselines not supplied: the result holds only relative to the baseline columns listed.",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _redact(argv):
    """Paths in the recorded command are reduced to their file name: reports get shared, home directories should not."""
    out = []
    for t in argv:
        t = str(t)
        out.append(Path(t).name if ("/" in t or "\\" in t) and not t.startswith("-") else t)
    return out


def provenance(argv, input_path, mapping_path=None, n_rows=None):
    import numpy, pandas, scipy, sklearn
    from .core import VERSION
    p = {"tool": f"baseline-increment {VERSION}",
         "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
         "command": "baseline-increment " + " ".join(_redact(argv)),
         "input_file": Path(input_path).name, "input_sha256": _sha256(input_path), "input_rows": n_rows,
         "python": sys.version.split()[0], "platform": platform.platform(),
         "packages": {"numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
                      "scikit-learn": sklearn.__version__}}
    if mapping_path:
        p["mapping_file"] = Path(mapping_path).name
        p["mapping_sha256"] = _sha256(mapping_path)
    return p


def _calibration():
    return json.loads((Path(__file__).parent / "data" / "calibration_summary.json").read_text())


def calibration_for_audit(settings):
    cal = _calibration()
    if settings.get("outcome") == "ordinal":
        return calibration_for_ordinal(settings)
    seeds, n_perm, stop = settings.get("n_seeds"), settings.get("n_perm"), settings.get("perm_stop")
    used_null = {"residual": "residual_gbm"}.get(settings.get("null"), settings.get("null"))
    standard = (used_null in ("residual_gbm", "residual_sl") and not settings.get("risk_bins")
                and settings.get("score_transform") in ("rank", "raw")
                and sorted(settings.get("families", [])) == ["gbm", "linear", "spline"])
    match = None
    for pr in cal["presets"]:
        s = pr["settings"]
        if (standard and seeds == s["seeds"] and n_perm == s["n_perm"] and (stop or None) == s["perm_stop"]
                and used_null == s.get("null", "residual_gbm")):
            match = pr
    fast = next(p for p in cal["presets"] if p["name"] == "fast")
    ref = match or fast
    lines = [f"Calibration run '{ref['name']}' ({ref['replicates']} simulated datasets per scenario, {ref['source']}, "
             f"commit {ref['commit']}): across {ref['null_scenarios']} scenarios where the detector adds nothing, the "
             f"highest false-flag rate was {ref['max_null_flag_rate']:.2f} ({ref['max_null_scenario']}; upper 95% bound "
             f"{ref['max_null_flag_rate_upper95']:.2f}); nominal rate 0.05.",
             f"Detection rate for planted signal: weak (0.3 SD) {ref['power']['weak_signal']:.2f}, moderate (0.6 SD) "
             f"{ref['power']['moderate_signal']:.2f}, strong (1.2 SD) {ref['power']['strong_signal']:.2f}."]
    if match:
        status = f"Your settings match calibration run '{match['name']}' exactly."
    else:
        status = (f"Your settings (seeds {seeds}, permutations {n_perm}, stopping {stop}, families "
                  f"{','.join(settings.get('families', []))}) do not match a calibrated run. "
                  "The procedure is the same, but the rates below were measured at the settings stated with them, "
                  "not at yours.")
        if not standard:
            status += (" You also changed the baseline families, null, risk banding or score transform from the "
                       "calibrated defaults (all three families, residual null).")
    lines.append(cal["interval_note"])
    return {"status": status, "matched": bool(match), "lines": lines}


def calibration_for_ordinal(settings):
    o = _calibration().get("ordinal")
    if not o:
        return {"status": "No calibration run exists for ordinal outcomes in this version.", "matched": False, "lines": []}
    s = o["settings"]
    matched = (settings.get("n_seeds") == s["seeds"] and settings.get("n_perm") == s["n_perm"]
               and settings.get("perm_stop") == s["perm_stop"] and sorted(settings.get("families", [])) == sorted(s["families"]))
    status = ("Your settings match the calibrated ordinal run exactly." if matched else
              f"Your settings differ from the calibrated ordinal run; its rates were measured at {json.dumps(s)}.")
    lines = [f"Calibrated ordinal run ({o['replicates']} simulated datasets per scenario, {o['source']}, commit {o['commit']}): "
             f"across {o['null_scenarios']} scenarios where the detector adds nothing, the highest false-flag rate was "
             f"{o['max_null_flag_rate']:.2f} ({o['max_null_scenario']}; upper 95% bound {o['max_null_flag_rate_upper95']:.2f}).",
             f"Detection rate for planted signal: weak {o['power']['weak_signal']:.2f}, moderate {o['power']['moderate_signal']:.2f}."]
    return {"status": status, "matched": matched, "lines": lines}


def calibration_section(c):
    """Rows for the raw-score calibration section of a report."""
    cov = _calibration().get("score_calibration_coverage")
    ci = lambda k: f" [{c[k + '_ci'][0]:.3f}, {c[k + '_ci'][1]:.3f}]" if k + "_ci" in c else ""
    rows = [("Calibration slope (1 ideal; < 1 overconfident)", f"{c['slope']:.3f}{ci('slope')}"),
            ("Calibration-in-the-large (0 ideal; > 0 under-predicts)", f"{c['citl']:+.3f}{ci('citl')}"),
            ("Brier score", f"{c['brier']:.4f}{ci('brier')}"),
            ("Mean predicted / observed rate", f"{c['mean_predicted']:.3f} / {c['observed_rate']:.3f}"),
            ("ECE (biased upward at finite n; descriptive)", f"{c['ece']:.3f}")]
    for i, b in enumerate(c["reliability"], 1):
        rows.append((f"  bin {i} (n {b['n']})", f"predicted {b['mean_predicted']:.3f}, observed {b['observed_rate']:.3f}"))
    if cov:
        rows.append(("Interval coverage (simulated calibrated scores, clustered)",
                     f"slope {cov['slope_coverage_of_1']:.2f}, calibration-in-the-large {cov['citl_coverage_of_0']:.2f} "
                     f"({cov['reps']} runs, {cov['source']})"))
    return rows


def calibration_for_compare(settings):
    c = _calibration()["compare"]
    s = c["settings"]
    matched = (settings.get("seeds") == s["seeds"] and settings.get("n_perm") == s["n_perm"]
               and settings.get("perm_stop") == s["perm_stop"] and sorted(settings.get("families", [])) == sorted(s["families"]))
    status = ("Your settings match the calibrated compare run exactly." if matched else
              "Your settings differ from the calibrated compare run; the rates below were measured at its settings "
              f"({json.dumps(s)}).")
    p = c["power"]
    lines = [f"Calibrated compare run ({c['replicates']} simulated datasets per scenario, {c['source']}, commit {c['commit']}): "
             f"highest false-flag rate across {c['registered_nulls']} registered null cases {c['max_null_flag_rate']:.2f} "
             f"({c['max_null']}; upper 95% bound {c['max_null_flag_rate_upper95']:.2f}).",
             f"Detection: improvement {min(p['improvement']):.2f}-{max(p['improvement']):.2f}, regression "
             f"{min(p['regression']):.2f}-{max(p['regression']):.2f}, standalone {p['standalone']:.2f}; standalone interval "
             f"coverage {min(c['standalone_coverage']):.2f}-{max(c['standalone_coverage']):.2f}.",
             f"Outside the registration: {c['unregistered_observation']} (a regression reported when v2 is a "
             "task-dependent rescaling of v1 may be a false alarm about 1 time in 8)."]
    return {"status": status, "matched": matched, "lines": lines}


# --------------------------------------------------------------------------------------------- documents

def _primary_summary(rep, baselines, what="the detector"):
    """0.13 summary: the held-out gain and its interval decide; nothing else."""
    b = ", ".join(baselines) if baselines else "none (intercept only)"
    ip, g = rep["incremental_prediction"], rep.get("gain_ci")
    ci = f"95% CI [{g['lower']:+.3f}, {g['upper']:+.3f}]" if g else "no interval"
    v = ip["verdict"]
    if v == "PASS":
        return (f"PASS: {what} improves held-out prediction beyond the declared baselines ({b}) by {rep['gain']:+.3f} AUROC "
                f"({ci}), under the declared model families.")
    if v == "FAIL":
        return (f"FAIL: {what}'s gain beyond the declared baselines ({b}) is below the minimum of {ip['minimum_gain']:+.3f} "
                f"AUROC ({rep['gain']:+.3f}, {ci}).")
    if v == "INCONCLUSIVE":
        return (f"INCONCLUSIVE: an improvement in held-out prediction beyond the declared baselines ({b}) was not shown "
                f"({rep['gain']:+.3f} AUROC, {ci}, {rep['n_items']} items). This is not evidence that the gain is zero.")
    return f"UNAVAILABLE: no validated interval was computed for the gain ({rep['gain']:+.3f} AUROC)."


def _secondary_line(rep):
    c = rep["conditional_information"]
    return (f"Secondary, experimental: the permutation test of conditional independence gives p = {c['p']:.3f} "
            f"(adds_information = {str(c['adds_information']).lower()}). It is not used for the verdict: its false-flag rate "
            "exceeds the nominal level when a baseline's effect varies across categories or the score is an XOR-like or "
            "higher-order function of the baselines.")


def _summary(adds, baselines, n_items, n_groups, what="the detector"):
    b = ", ".join(baselines) if baselines else "none (intercept only)"
    if adds:
        return (f"Adds information: {what} provides statistically significant incremental predictive information "
                f"beyond the declared baselines ({b}).")
    return (f"Not detected: this evaluation did not detect incremental predictive information from {what} beyond the "
            f"declared baselines ({b}) at this sample size ({n_items} items, {n_groups} groups). This is not evidence "
            "that the information is absent.")


def _settings_rows(s):
    rows = [("Outcome", s.get("outcome", "binary")), ("Baseline families", ", ".join(s.get("families", []))),
            ("CV seeds / grouped folds", f"{s.get('n_seeds', s.get('seeds'))} / 5"),
            ("Permutations (maximum) / stopping count", f"{s.get('n_perm')} / {s.get('perm_stop')}"),
            ("Null", str(s.get("null"))), ("Score transform", str(s.get("score_transform", "rank"))),
            ("Rank within", str(s.get("rank_within")))]
    if "interval" in s:
        none = s["interval"] in ("bootstrap", "half") and not s.get("n_boot")
        rows.append(("Increment interval method", "none computed (no refit replicates)" if none else str(s["interval"])))
    rows.append(("Definition of 'adds information'", "permutation p-value below 0.05 and increment above 0, under the "
                 "settings above; fixed before the run"))
    rows.append(("Random seeds", "fixed: CV folds use seeds 0.., permutations 50000 + draw index; the same input, "
                 "settings, package version and pinned dependencies give the same numbers"))
    return rows


def audit_document(rep, ops, prov, title="Detector audit"):
    head = rep["headline_family"]
    verdict = _primary_summary(rep, list(rep["baseline_auc"]))
    ip = rep["incremental_prediction"]
    rule = (f"Decision rule: PASS when the lower bound of the 95% interval for the held-out gain exceeds the minimum gain "
            f"({ip['minimum_gain']:+.3f}). The interval is a clustered DeLong interval whose coverage was validated in "
            "simulation (0.905-0.965). " + _secondary_line(rep))
    key = [("Items / groups / positive rate", f"{rep['n_items']} / {rep['n_groups']} / {rep['positive_rate']:.3f}"),
           ("Detector AUROC (alone)", f"{rep['detector_auc']:.3f} [{rep['detector_auc_ci'][0]:.3f}, {rep['detector_auc_ci'][1]:.3f}]"),
           (f"Baselines AUROC ({head})", f"{rep['baselines_combined_auc']:.3f}"),
           (f"Baselines + detector AUROC ({head})", f"{rep['baselines_plus_detector_auc']:.3f}"),
           ("Increment", f"{rep['increment']:+.3f}"),
           ("Permutation p-value", f"{rep['increment_perm_p']:.3f} ({rep.get('n_perm_drawn', rep['settings']['n_perm'])} null draws)")]
    if rep.get("increment_ci"):
        key.append((f"Increment 95% interval ({rep.get('increment_ci_note', 'not used for the decision')})",
                    f"[{rep['increment_ci'][0]:+.3f}, {rep['increment_ci'][1]:+.3f}]"))
    for c, v in rep["baseline_auc"].items():
        key.append((f"Baseline '{c}' alone (linear)", f"{v:.3f}"))
    use = []
    for o in ops:
        use.append((f"At {o['fpr']:.0%} false-positive rate",
                    f"baselines catch {o['tpr_baselines']:.1%} of positives; baselines + detector {o['tpr_plus_detector']:.1%} "
                    f"({o['tpr_gain'] * 100:+.1f} points); detector alone {o['tpr_detector_alone']:.1%}"))
        use.append((f"  per 1,000 positives, at {o['fpr']:.0%} false-positive rate",
                    f"baselines catch {o['tpr_baselines'] * 1000:.0f}, baselines + detector {o['tpr_plus_detector'] * 1000:.0f} "
                    f"(change {o['tpr_gain'] * 1000:+.0f})"))
    pr = rep.get("precision_recall")
    if pr:
        use.append(("Average precision (AUPRC; descriptive, not calibrated as a test)",
                    f"detector alone {pr['detector_ap']:.3f}; baselines {pr['baselines_ap']:.3f}, + detector "
                    f"{pr['baselines_plus_detector_ap']:.3f} ({pr['ap_gain']:+.3f}); prevalence {pr['prevalence']:.3f}"))
    lad = [(("  + " if i else "") + (", ".join(r["baselines"]) if i == 0 else r["baselines"][-1]) if r["baselines"] else "no baselines",
            f"increment {r['increment']:+.3f} over baselines AUROC {r['baselines_auc']:.3f} ({r['family']})")
           for i, r in enumerate(rep.get("ladder", []))]
    fam = [(f"Family {k}", f"baselines {v['baseline_auc']:.3f}, + detector {v['plus_detector_auc']:.3f}, increment "
                           f"{v['increment']:+.3f}" + ("  (headline)" if k == head else "")) for k, v in rep["families"].items()]
    strata = []
    for col, st in rep.get("strata", {}).items():
        strata.append((f"'{col}' base rate alone", f"{st['base_rate_only_auc']:.3f}"))
        for v, e in st["within"].items():
            a = f"{e['detector_auc']:.3f}" if "detector_auc" in e else "n/a"
            strata.append((f"  {col} = {v}", f"n {e['n']}, positive rate {e['positive_rate']:.2f}, detector AUROC {a}"))
    cal = calibration_for_audit(rep["settings"])
    calib = calibration_section(rep["calibration"]) if rep.get("calibration") else []
    sens = []
    if rep.get("sensitivity"):
        from .sensitivity import sentence
        sv = rep["sensitivity"]
        sens.append(("Summary", sentence(sv)))
        if sv.get("robustness_value") is not None:
            sens.append(("Residual correlation r (detector vs label, beyond the baselines)", f"{sv['residual_correlation']:.3f}"))
            sens.append(("Robustness value (equal-strength point)", f"{sv['robustness_value']:.3f}"))
            for c, v in sv["benchmarks"].items():
                sens.append((f"Declared baseline '{c}', if it had been left out",
                             f"erasure index {v['phi']:.2f} (linear {v['phi_linear']:.2f}, flexible {v['phi_flexible']:.2f})"))
        sens.append(("Status", sv["label"]))
    return {"title": title, "summary": verdict,
            "record": _settings_rows(rep["settings"]) + [("Statistical preset", json.dumps(rep.get("statistical_preset")))],
            "verdict": [verdict, rule], "sections": [
        ("Key numbers", key), ("What it means in use (out-of-fold, headline family)", use),
        ("Every baseline family (the headline is chosen by baselines alone, before looking at the detector)", fam),
        ("By stratum", strata),
        ("Baseline ladder (descriptive: baselines added one at a time, in the order given)", lad),
        ("Omitted-baseline sensitivity (descriptive; erasure index 1 = would make the detector redundant)", sens),
        ("The raw score as a probability (descriptive, no verdict)", calib)],
        "warnings": rep["warnings"], "calibration": cal, "not_tested": NOT_TESTED, "provenance": prov,
        "settings": rep["settings"]}


def ordinal_document(rep, prov, title="Detector audit (graded labels)"):
    head = rep["headline_family"]
    verdict = (f"DESCRIPTIVE: the gain in concordance beyond the baselines is {rep['increment']:+.3f}. Graded labels have no "
               "validated interval yet, so no PASS/FAIL verdict is given.")
    rule = (f"Secondary, experimental: permutation test of conditional independence p = {rep['increment_perm_p']:.3f}. "
            "Concordance: among pairs of items with different levels, the share where the more severe item has the higher score.")
    key = [("Items / groups", f"{rep['n_items']} / {rep['n_groups']}"),
           ("Items per level", ", ".join(f"{k}: {v}" for k, v in rep["levels"].items())),
           ("Detector concordance (alone)", f"{rep['detector_c']:.3f} [{rep['detector_c_ci'][0]:.3f}, {rep['detector_c_ci'][1]:.3f}]"),
           (f"Baselines concordance ({head})", f"{rep['baselines_combined_c']:.3f}"),
           (f"Baselines + detector concordance ({head})", f"{rep['baselines_plus_detector_c']:.3f}"),
           ("Increment", f"{rep['increment']:+.3f}"),
           ("Permutation p-value", f"{rep['increment_perm_p']:.3f} ({rep['n_perm_drawn']} null draws)")]
    key += [(f"Baseline '{c}' alone (linear)", f"{v:.3f}") for c, v in rep["baseline_c"].items()]
    fam = [(f"Family {k}", f"baselines {v['baseline_c']:.3f}, + detector {v['plus_detector_c']:.3f}, increment "
                           f"{v['increment']:+.3f}" + ("  (headline)" if k == head else "")) for k, v in rep["families"].items()]
    return {"title": title, "summary": verdict,
            "record": _settings_rows(rep["settings"]),
            "verdict": [verdict, rule], "sections": [("Key numbers", key), ("Every baseline family", fam)],
            "warnings": rep["warnings"], "calibration": calibration_for_audit(rep["settings"]),
            "not_tested": NOT_TESTED + ["Operating points (recall at a fixed false-positive rate) are not defined for graded labels."],
            "provenance": prov, "settings": rep["settings"]}


def compare_document(rep, ops, prov, title="Detector release comparison", versions=None):
    i, r, s = rep["improvement"], rep["regression"], rep["standalone"]
    side = [("Measure", "v1", "v2", "Change")]
    side.append(("AUROC alone", f"{s['auc_old']:.3f}", f"{s['auc_new']:.3f}", f"{s['difference']:+.3f}"))
    if versions:
        a, b = versions["v1"], versions["v2"]
        side.append((f"Baselines alone ({a['family']})", f"{a['baselines_auc']:.3f}", f"{b['baselines_auc']:.3f}", "same baselines"))
        side.append(("Increment beyond the baselines (each version alone, descriptive)", f"{a['increment']:+.3f}",
                     f"{b['increment']:+.3f}", f"{b['increment'] - a['increment']:+.3f}"))
    for o in ops:
        side.append((f"Recall at {o['fpr']:.0%} false-positive rate (alone)", f"{o['tpr_old']:.1%}", f"{o['tpr_new']:.1%}",
                     f"{(o['tpr_new'] - o['tpr_old']) * 100:+.1f} points"))
    q = [("Does v2 add information beyond v1 and the baselines?", "yes (detected)" if i["detected"] else "not detected"),
         ("Does v1 keep information that v2 lacks (regression)?", "yes (detected)" if r["detected"] else "not detected"),
         ("Is v2's AUROC alone different from v1's?", s["verdict"])]
    key = [("Items / groups", f"{rep['n_items']} / {rep['n_groups']}"),
           ("Old (v1) / new (v2)", f"{rep['old']} / {rep['new']}"),
           ("Standalone AUROC, v1 / v2", f"{s['auc_old']:.3f} / {s['auc_new']:.3f}"),
           ("Standalone difference v2 - v1", f"{s['difference']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]: {s['verdict']}"),
           ("Improvement: v2 beyond v1 + baselines", f"increment {i['increment']:+.3f}, p {i['p']:.3f}, Holm p {i['p_holm']:.3f}: "
                                                     f"{'detected' if i['detected'] else 'not detected'}"),
           ("Regression: v1 beyond v2 + baselines", f"increment {r['increment']:+.3f}, p {r['p']:.3f}, Holm p {r['p_holm']:.3f}: "
                                                    f"{'detected' if r['detected'] else 'not detected'}")]
    use = [(f"At {o['fpr']:.0%} false-positive rate", f"v1 alone catches {o['tpr_old']:.1%} of positives, v2 alone {o['tpr_new']:.1%} "
                                                     f"({(o['tpr_new'] - o['tpr_old']) * 100:+.1f} points)") for o in ops]
    rules = ["Improvement and regression: detected if the Holm-adjusted permutation p-value over the pair is below 0.05 "
             "and the increment is above 0. Standalone: detected if the 95% paired group-bootstrap interval excludes 0.",
             "Not detected means the data did not show it at this sample size; it does not mean the effect is absent."]
    summ = ("Improvement detected: v2 adds information beyond v1 and the baselines." if i["detected"] else
            "Improvement not detected: v2 was not shown to add information beyond v1 and the baselines.")
    summ += (" Regression detected: v1 keeps information that v2 lacks." if r["detected"] else
             " No regression detected.")
    st = rep["settings"]
    rec = _settings_rows({"families": st.get("families", []), "seeds": st.get("seeds"), "n_perm": st.get("n_perm"),
                          "perm_stop": st.get("perm_stop"), "null": st.get("null", "residual")})
    return {"title": title, "summary": summ, "record": rec, "verdict": rep["summary"] + rules,
            "sections": [("v1 against v2", side), ("The three questions, answered separately", q), ("Key numbers", key),
            ("What it means in use (standalone scores, no fitting)", use)],
            "warnings": [], "calibration": calibration_for_compare(rep["settings"]),
            "not_tested": NOT_TESTED, "provenance": prov, "settings": rep["settings"]}


# --------------------------------------------------------------------------------------------- rendering

def _md_table(rows, e):
    if len(rows[0]) > 2:
        head, body = rows[0], rows[1:]
        return (["| " + " | ".join(e(x) for x in head) + " |", "|" + "---|" * len(head)]
                + ["| " + " | ".join(e(x) for x in r) + " |" for r in body])
    return ["| | |", "|---|---|"] + [f"| {e(a)} | {e(b)} |" for a, b in rows]


def _dne():
    from .evidence import DOES_NOT_ESTABLISH
    return DOES_NOT_ESTABLISH


def render_markdown(doc):
    e = lambda x: str(x).replace("|", "\\|")
    out = [f"# {doc['title']}", ""]
    if doc.get("summary"):
        out += ["## Summary", "", f"**{doc['summary']}**", ""]
    out += [f"**{doc['verdict'][0]}**", ""] + [f"- {v}" for v in doc["verdict"][1:]] + [""]
    out += ["# For engineers", ""]
    for name, rows in doc["sections"]:
        if not rows:
            continue
        out += [f"## {name}", ""] + _md_table(rows, e) + [""]
    if doc["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in doc["warnings"]] + [""]
    out += ["# For auditors", ""]
    if doc.get("record"):
        out += ["## Audit record", ""] + _md_table(doc["record"], e) + [""]
    out += ["## Calibration evidence for these settings", "", doc["calibration"]["status"], ""]
    out += [f"- {x}" for x in doc["calibration"]["lines"]] + [""]
    out += ["## Not tested by this report", ""] + [f"- {x}" for x in doc["not_tested"]] + [""]
    out += ["## What this result does not establish", ""] + [f"- {x}" for x in _dne()] + [""]
    out += ["## Provenance", "", "| | |", "|---|---|"]
    for k, v in doc["provenance"].items():
        out.append(f"| {e(k)} | `{e(json.dumps(v) if isinstance(v, dict) else v)}` |")
    out += ["", f"Settings: `{e(json.dumps(doc['settings'], default=str))}`", "",
            "Sign-off: ____________________   Date: ____________", ""]
    return "\n".join(out)


def render_html(doc):
    e = html.escape
    def table(rows):
        if rows and len(rows[0]) > 2:
            head = "<tr>" + "".join(f"<th scope=col>{e(str(x))}</th>" for x in rows[0]) + "</tr>"
            body = "".join("<tr>" + f"<th>{e(str(r[0]))}</th>" + "".join(f"<td>{e(str(x))}</td>" for x in r[1:]) + "</tr>"
                           for r in rows[1:])
            return f"<table class=wide>{head}{body}</table>"
        return "<table>" + "".join(f"<tr><th>{e(str(a))}</th><td>{e(str(b))}</td></tr>" for a, b in rows) + "</table>"
    ok = doc["verdict"][0].startswith(("PASS", "v2 improves")) and "not shown" not in doc["verdict"][0]
    secs = "".join(f"<h2>{e(n)}</h2>{table(r)}" for n, r in doc["sections"] if r)
    summary = f"<p class=summary>{e(doc['summary'])}</p>" if doc.get("summary") else ""
    record = f"<h2>Audit record</h2>{table(doc['record'])}" if doc.get("record") else ""
    warn = ("<h2>Warnings</h2><ul>" + "".join(f"<li>{e(w)}</li>" for w in doc["warnings"]) + "</ul>") if doc["warnings"] else ""
    cal = (f"<h2>Calibration evidence for these settings</h2><p class=status>{e(doc['calibration']['status'])}</p><ul>"
           + "".join(f"<li>{e(x)}</li>" for x in doc["calibration"]["lines"]) + "</ul>")
    nt = "<h2>Not tested by this report</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in doc["not_tested"]) + "</ul>"
    nt += "<h2>What this result does not establish</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in _dne()) + "</ul>"
    prov = "<h2>Provenance</h2>" + table([(k, json.dumps(v) if isinstance(v, dict) else v) for k, v in doc["provenance"].items()])
    prov += f"<p class=small>Settings: <code>{e(json.dumps(doc['settings'], default=str))}</code></p>"
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{e(doc['title'])}</title><style>
:root{{--fg:#1f2328;--muted:#57606a;--line:#d0d7de;--bg:#fff;--ok:#1a7f37;--no:#9a6700;--panel:#f6f8fa}}
@media (prefers-color-scheme:dark){{:root{{--fg:#e6edf3;--muted:#8b949e;--line:#30363d;--bg:#0d1117;--ok:#3fb950;--no:#d29922;--panel:#161b22}}}}
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 16px;color:var(--fg);background:var(--bg)}}
h1{{font-size:1.6rem;margin-bottom:.2rem}}h2{{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid var(--line);padding-bottom:.2rem}}
.verdict{{border-left:4px solid {'var(--ok)' if ok else 'var(--no)'};background:var(--panel);padding:.6rem 1rem;margin:1rem 0}}
.verdict p:first-child{{font-weight:600;font-size:1.05rem}}.summary{{font-size:1.1rem}}
.part{{margin-top:2.4rem;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
table.wide th{{width:auto}}table.wide th[scope=col]{{color:var(--muted)}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;vertical-align:top;padding:.35rem .5rem;border-top:1px solid var(--line)}}th{{width:38%;font-weight:500;color:var(--muted)}}
td{{overflow-wrap:anywhere}}.small,.status{{color:var(--muted);font-size:13px}}code{{font-size:12px;overflow-wrap:anywhere}}
.sign{{margin-top:2.5rem;display:flex;gap:3rem;color:var(--muted)}}.sign span{{border-top:1px solid var(--line);padding-top:.3rem;min-width:14rem}}
@media print{{:root{{--fg:#000;--bg:#fff;--panel:#f3f3f3}}body{{margin:0;max-width:none;font-size:11pt}}h2{{break-after:avoid}}table,ul{{break-inside:avoid}}}}
</style></head><body><h1>{e(doc['title'])}</h1><p class=small>{e(doc['provenance']['tool'])} &middot; {e(doc['provenance']['generated_utc'])}
&middot; input {e(doc['provenance']['input_file'])} (sha256 {e(doc['provenance']['input_sha256'][:12])}&hellip;)</p>
<div class=verdict>{summary}{''.join(f'<p>{e(v)}</p>' for v in doc['verdict'])}</div>
<p class=part>For engineers</p>{secs}{warn}
<p class=part>For auditors</p>{record}{cal}{nt}{prov}
<div class=sign><span>Reviewed by</span><span>Date</span></div></body></html>"""


def write_report(doc, path):
    path = str(path)
    text = render_markdown(doc) if path.endswith(".md") else render_html(doc)
    Path(path).write_text(text)
    return path
