"""Validity envelope: the conditions under which the decision rule's error rates were measured (CALIBRATION.md).

The audit's results are unchanged by this module. It adds a *status* that says whether the run
lies inside the conditions the calibration suites covered, so a result is never presented with more authority than
its evidence:

  not_testable              too little data for the statistic to mean anything (fewer than 10 items in the rarer
                            class, or fewer than 10 groups); the verdict is reported but must not be relied on
  outside_validated_range   the procedure ran, but outside the calibrated conditions listed below; error rates
                            there are not measured
  within_validated_range    inside the envelope; the verdict itself is the primary result (PASS / INCONCLUSIVE / FAIL)

The envelope is taken from the calibration designs, not chosen to flatter: the smallest calibrated designs are
`few_groups` (60 groups) and `imbalance` (about 33 positives in 600 items); the largest are the stress tests
(10 baseline columns, groups of up to 40 items, 10,000 rows).

One data structure lies outside the interval's validated scope whatever the sizes above: a score that is a
three-way interaction of baselines and a category (R1 structure D4, coverage 0.845 / 0.695; docs/prereg/PREREG_R1.md).
It cannot be detected from the data, so it is stated in the documentation rather than as a run-time status.
"""
import numpy as np

ENVELOPE = {
    "min_groups": 60,
    "min_minority_class": 30,
    "max_baseline_columns": 10,
    "max_group_size": 40,
    "largest_calibrated_rows": 10_000,
    "outcomes": ["binary", "ordinal (3-10 integer levels)"],
}
NOT_TESTABLE = {"min_groups": 10, "min_minority_class": 10}


def status(rep, df=None, group=None):
    """Status and reasons for an audit report (binary or ordinal)."""
    n, g = rep["n_items"], rep["n_groups"]
    if "levels" in rep:
        minority = min(rep["levels"].values())
    else:
        pos = int(round(rep["positive_rate"] * n))
        minority = min(pos, n - pos)
    nb = len(rep.get("baseline_auc", rep.get("baseline_c", {})))
    max_gs = None
    if df is not None and group:
        max_gs = int(df.groupby(group).size().max())
    hard, soft, notes = [], [], []
    if g < NOT_TESTABLE["min_groups"]:
        hard.append(f"only {g} groups")
    if minority < NOT_TESTABLE["min_minority_class"]:
        hard.append(f"only {minority} items in the rarer class")
    if not hard:
        if g < ENVELOPE["min_groups"]:
            soft.append(f"{g} groups (calibrated from {ENVELOPE['min_groups']})")
        if minority < ENVELOPE["min_minority_class"]:
            soft.append(f"{minority} items in the rarer class (calibrated from {ENVELOPE['min_minority_class']})")
        if nb > ENVELOPE["max_baseline_columns"]:
            soft.append(f"{nb} baseline columns (calibrated up to {ENVELOPE['max_baseline_columns']})")
        if max_gs is not None and max_gs > ENVELOPE["max_group_size"]:
            soft.append(f"groups of up to {max_gs} items (calibrated up to {ENVELOPE['max_group_size']})")
    if n > ENVELOPE["largest_calibrated_rows"]:
        notes.append(f"{n} rows: larger than the largest calibrated design ({ENVELOPE['largest_calibrated_rows']} rows); "
                     "no failure mode is known at larger sizes, but rates there are not measured")
    if hard:
        st = "not_testable"
    elif soft:
        st = "outside_validated_range"
    else:
        st = "within_validated_range"
    return {"status": st, "reasons": hard or soft, "notes": notes, "envelope": ENVELOPE}


def prefix(st):
    """Sentence put before the summary when the run is not inside the envelope."""
    if st["status"] == "not_testable":
        return ("Not testable: " + "; ".join(st["reasons"]) + ". The verdict below is shown for completeness and "
                "should not be relied on.")
    if st["status"] == "outside_validated_range":
        return ("Outside the validated range: " + "; ".join(st["reasons"]) + ". The procedure ran, but its error "
                "rates were not measured under these conditions.")
    return ""
