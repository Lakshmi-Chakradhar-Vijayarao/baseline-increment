"""Build baseline_increment/data/calibration_summary.json from the committed simulation outputs, so reports can cite
the measured error rates for the settings a user ran. Run from the repo root: python scripts/build_calibration_summary.py"""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulations"
NULLS = ["noise", "metadata_copy", "nonlinear_of_metadata", "metadata_plus_noise", "random_labels", "few_groups",
         "unequal_groups", "imbalance", "misspecified_baseline", "interaction_meta", "difficulty_by_model", "scale_by_model", "simpson"]


def wilson_hi(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    return min(1.0, c + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)


def audit_preset(fname, commit, name, settings):
    r = json.load(open(SIM / fname))
    sc = r["scenarios"]
    worst = max(NULLS, key=lambda n: sc[n]["reject_rate"])
    return {"name": name, "source": f"simulations/{fname}", "commit": commit, "replicates": r["reps"], "settings": settings,
            "null_scenarios": len(NULLS), "max_null_flag_rate": sc[worst]["reject_rate"], "max_null_scenario": worst,
            "max_null_flag_rate_upper95": round(wilson_hi(sc[worst]["reject_rate"], r["reps"]), 3),
            "power": {k: sc[k]["reject_rate"] for k in ("weak_signal", "moderate_signal", "strong_signal",
                                                         "signal_with_difficulty", "hard_only_signal")},
            "all_criteria_met": bool(r.get("all_pass", True))}


def compare_preset():
    r = json.load(open(SIM / "compare_v0.8.0.json"))
    sc = r["scenarios"]
    nulls = [("reseed", "T1"), ("monotone", "T1"), ("interaction", "T1"), ("coarsened", "T1"),
             ("strict_upgrade", "T2"), ("monotone", "T2"), ("complementary", "T3"), ("monotone", "T3")]
    worst = max(nulls, key=lambda t: sc[t[0]][t[1]])
    return {"name": "compare", "source": "simulations/compare_v0.8.0.json", "commit": "164fa2a", "replicates": r["reps"],
            "settings": {"seeds": 2, "n_perm": 200, "perm_stop": 10, "families": ["linear", "spline", "gbm"]},
            "registered_nulls": len(nulls), "max_null_flag_rate": sc[worst[0]][worst[1]], "max_null": f"{worst[0]} {worst[1]}",
            "max_null_flag_rate_upper95": round(wilson_hi(sc[worst[0]][worst[1]], r["reps"]), 3),
            "power": {"improvement": [sc["strict_upgrade"]["T1"], sc["complementary"]["T1"]],
                      "regression": [sc["reseed"]["T2"], sc["coarsened"]["T2"], sc["complementary"]["T2"]],
                      "standalone": sc["strict_upgrade"]["T3"]},
            "standalone_coverage": [sc["complementary"]["T3_coverage_of_mean_diff"], sc["strict_upgrade"]["T3_coverage_of_mean_diff"]],
            "unregistered_observation": f"interaction scenario, regression test: {sc['interaction']['T2']:.2f}"}


def ordinal_preset(commit):
    r = json.load(open(SIM / "ordinal_v0.11.0.json"))
    sc = r["scenarios"]
    nulls = ["noise", "metadata_copy", "nonlinear_of_metadata", "interaction_meta", "few_groups"]
    worst = max(nulls, key=lambda n: sc[n]["flag_rate"])
    return {"name": "ordinal", "source": "simulations/ordinal_v0.11.0.json", "commit": commit, "replicates": r["reps"],
            "settings": {k: r["settings"][k] for k in ("seeds", "n_perm", "perm_stop", "families")},
            "null_scenarios": len(nulls), "max_null_flag_rate": sc[worst]["flag_rate"], "max_null_scenario": worst,
            "max_null_flag_rate_upper95": round(wilson_hi(sc[worst]["flag_rate"], r["reps"]), 3),
            "power": {"weak_signal": sc["weak_signal"]["flag_rate"], "moderate_signal": sc["moderate_signal"]["flag_rate"]}}


def calcov(commit):
    r = json.load(open(SIM / "calcov_v0.11.0.json"))
    return {"source": "simulations/calcov_v0.11.0.json", "commit": commit, "reps": r["reps"],
            "slope_coverage_of_1": r["calibrated"]["slope_coverage_of_1"], "citl_coverage_of_0": r["calibrated"]["citl_coverage_of_0"],
            "miscalibrated_detected": r["miscalibrated_p_squared"]["slope_ci_excludes_1"], "pass": r["pass"]}


out = {"note": "Measured on simulated data with known truth; see CALIBRATION.md. Rates apply to the settings listed.",
       "presets": [audit_preset("results_v0.2.2.json", "e5ea27e", "suite", {"seeds": 2, "n_perm": 100, "perm_stop": None}),
                   audit_preset("results_v0.7.0_fast.json", "d87836b", "fast", {"seeds": 2, "n_perm": 200, "perm_stop": 10})],
       "compare": compare_preset(),
       "ordinal": ordinal_preset("924167c"),
       "score_calibration_coverage": calcov("924167c"),
       "interval_note": "The refit-bootstrap increment interval is conservative under the null (coverage 1.00 vs 0.85-0.99 "
                        "band, v0.2.2 and v0.7.0 half-sample) and is not used for the decision."}
(ROOT / "baseline_increment" / "data" / "calibration_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({p["name"]: (p["max_null_flag_rate"], p["power"]["weak_signal"]) for p in out["presets"]}))
