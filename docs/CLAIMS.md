# Claim ledger

Generated from `claims/claims.json` by `scripts/build_claims_md.py`; do not edit by hand.

Every public claim of the README, with its evidence and the test that keeps it honest. Status: established (evidence recorded and tested), under_validation (pre-registered, runs in progress), not_established (not claimed). tests/test_claims.py fails if recorded evidence stops supporting an established claim.

| ID | Claim | Evidence | Test | Status |
|---|---|---|---|---|
| CLAIM-001 | adds_information is true exactly when the permutation p-value is below 0.05 and the increment is above 0 | baseline_increment/core.py (audit) | `tests/test_claims.py::test_claim_001_decision_rule` | Established |
| CLAIM-002 | Items sharing a group never sit in both the training and the test fold | baseline_increment/core.py (group_folds) | `tests/test_claims.py::test_claim_002_groups_never_straddle` | Established |
| CLAIM-003 | The verdict does not change under monotone rescaling of the score | rank transform (core.transform_score) | `tests/test_claims.py::test_claim_003_rank_invariance` | Established |
| CLAIM-004 | --fast keeps the false-flag rate at or below 0.10 on every registered no-signal scenario (nominal 0.05) | simulations/results/results_v0.7.0_fast.json | `tests/test_claims.py::test_claim_004_fast_null_rates` | Established |
| CLAIM-005 | --fast detects moderate and strong planted signal in at least 80% of runs | simulations/results/results_v0.7.0_fast.json | `tests/test_claims.py::test_claim_005_fast_power` | Established |
| CLAIM-006 | On real detectors the tool reproduces seven independent prior analyses (verdict, magnitude within the prior interval widened by 0.02, ordering) | validation/results.json (agreement) | `tests/test_claims.py::test_claim_006_real_data_agreement` | Established |
| CLAIM-007 | On real data structures with the detector replaced by a no-information one, every false-flag rate is at most 0.10 | validation/results.json (nulls) | `tests/test_claims.py::test_claim_007_real_data_nulls` | Established |
| CLAIM-008 | Editing any file in an evidence package, or removing or reordering a ledger entry, is detected by verify-evidence | baseline_increment/evidence.py | `tests/test_assurance.py::test_evidence_package_ledger_and_tamper_detection` | Established |
| CLAIM-009 | The package opens no network connection | tests/test_offline.py | `tests/test_offline.py::test_cli_audit_and_report_offline` | Established |
| CLAIM-010 | The binary engine's output is fixed to 1e-12 for a given version and pinned dependencies | tests/fixtures/golden_audit_synthetic.json | `tests/test_golden.py::test_binary_engine_is_unchanged` | Established |
| CLAIM-011 | A policy cannot invoke an uncalibrated rule (minimum gain, other alpha, other metric) | baseline_increment/policy.py | `tests/test_assurance.py::test_policy_refuses_uncalibrated_rules` | Established |
| CLAIM-012 | The 0.12 null keeps interaction-only detectors at or below 0.10 false flags (upper 95% bound) | simulations/results/stress_v0.12.0_sl.json | `tests/test_claims.py::test_claim_012_interaction_null` | Under validation |
| CLAIM-013 | The increment has a calibrated 95% interval | docs/prereg/PREREG_0.12.md Part B | — | Under validation |
| CLAIM-014 | The omitted-baseline sensitivity analysis gives no false comfort against simulated omitted baselines | docs/prereg/PREREG_0.13.md Part 1 | — | Under validation |
| CLAIM-015 | The sample-size planner's power predictions hold under its own and a misspecified generator | docs/prereg/PREREG_0.13.md Part 2 | — | Under validation |
| CLAIM-016 | A minimum-gain acceptance gate | — | — | Not established |
| CLAIM-017 | Increments generalise across datasets or models (formal heterogeneity test) | — | — | Not established |
| CLAIM-018 | Construct validity: that the label measures what the evaluator is claimed to measure | — | — | Not established |
