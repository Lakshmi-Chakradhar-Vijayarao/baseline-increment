# Calibration record

**What this is:** every measurement of whether baseline-increment's tests behave as claimed, made on simulated data where the truth is known. Results are appended per release and never rewritten. The raw outputs are in [`simulations/results/`](simulations/results/). The frozen pre-registrations are in [`docs/prereg/`](docs/prereg/).

| Section | What was measured | Engine / code | Verdict |
|---|---|---|---|
| [Permutation test and decision rule](#permutation-test-and-decision-rule-adds_information--p--005-and-increment--0) | false-flag rate on 13 nulls, power on 5 signals | v0.2.2 (`e7c8ee2`) | all criteria met |
| [Refit-bootstrap CI](#refit-bootstrap-ci-40-replicates-100-refit-bootstraps-each-run-on-v022) | interval coverage | v0.2.2 | conservative (too wide); not used for decisions |
| [`check` battery](#check-battery-40-replicates-v030-settings-n_boot-50-n_perm-99-seeds-2) | each cheap-explanation check | v0.3.0 | all criteria met |
| [v0.4 validity estimands](#v04-validity-estimands-simulationsvalidity_suitepy-code-040) | decomposition, R²_v, panel sensitivity | 0.4.0 (`6b784ea`) | met at the second run |
| [Misspecified-baseline re-check](#re-check-of-the-misspecified-baseline-null-19-nov-2026-version-050-engine-unchanged-since-022) | the borderline null at 400 runs | 0.5.0 | 0.065, consistent with 0.05 |
| [0.7.0 additions](#070-additions-pre-registered-in-docspreregprereg_07md-commit-412317b-engine-at-86df237) | `--fast` preset; half-sample interval | 86df237 | fast adopted; half-sample rejected |
| [0.8.0 `compare`](#080-compare-detector-release-comparison-pre-registered-in-docspreregprereg_08md-commit-3b69b76) | improvement, regression, standalone | 164fa2a | adopted |
| [0.11.0 ordinal and calibration](#0110-ordinal-outcome-and-raw-score-calibration-pre-registered-in-docspreregprereg_011md-commit-c522f40-code-at-924167c) | ordinal nulls and power; calibration-interval coverage | 924167c | both adopted |
| [Stress test after 0.11.1](#stress-test-after-0111-pre-registered-in-docspreregprereg_validationmd-part-b-commit-01826ec) | 10 baselines, 10,000 rows, uneven groups, clustering; 400-run borderline re-checks | 0.11.1 | 10 pass, 1 borderline (interaction, 0.095) |
| [Real-data validation](#real-data-validation-after-0111-pre-registered-in-docspreregprereg_validationmd-part-a--addendum-1) | agreement with independent analyses; nulls on real data | 0.11.1 | agrees; real-data nulls 0.053 pooled |

**Two rules apply throughout:**
- The ≤ 0.10 flag-rate limits are development acceptance criteria at 100 replicates, not proofs of exact calibration.
- The binary engine has not changed since v0.2.2. `tests/test_golden.py` enforces this from 0.11.0 onwards.

---

*The sections below were first written at 0.4.0 by `simulations/legacy_report_v0.4.py` (now guarded so that it cannot overwrite this file). Later sections were appended by hand at each release.*

*Original provenance:*
- *The permutation and coverage runs used the audit engine at code commit `e7c8ee2` (v0.2.2 / v0.3.0).*
- *The v0.4 validity suite was run at `6b784ea`.*
- *0.4.0 added a k < 5 reference-panel warning; 0.4.1 changed packaging only.*

## Permutation test and decision rule (`adds_information` = p < 0.05 and increment > 0)

100 replicates per scenario. Rates have Wilson 95% intervals; the standard error at a true 5% rate is about 0.022.

| Group | Scenario | Flag rate [95% CI] | p < 0.05 alone | Mean increment (headline) | Mean increment (linear) | Headline linear/spline/gbm | Acceptance | Result |
|---|---|---|---|---|---|---|---|---|
| null | `noise` | 0.06 [0.03, 0.12] | 0.06 | -0.0010 | -0.0010 | 99/1/0 | ≤ 0.10 | pass |
| null | `metadata_copy` | 0.05 [0.02, 0.11] | 0.05 | -0.0008 | -0.0008 | 99/1/0 | ≤ 0.10 | pass |
| null | `nonlinear_of_metadata` | 0.01 [0.00, 0.05] | 0.02 | -0.0019 | -0.0018 | 96/2/2 | ≤ 0.10 | pass |
| null | `metadata_plus_noise` | 0.03 [0.01, 0.08] | 0.03 | -0.0010 | -0.0011 | 96/4/0 | ≤ 0.10 | pass |
| null | `random_labels` | 0.03 [0.01, 0.08] | 0.03 | -0.0025 | -0.0004 | 22/33/45 | ≤ 0.10 | pass |
| null | `few_groups` | 0.07 [0.03, 0.14] | 0.07 | -0.0043 | -0.0031 | 85/7/8 | ≤ 0.10 | pass |
| null | `unequal_groups` | 0.03 [0.01, 0.08] | 0.03 | -0.0006 | -0.0005 | 97/3/0 | ≤ 0.10 | pass |
| null | `imbalance` | 0.05 [0.02, 0.11] | 0.05 | -0.0074 | -0.0047 | 81/13/6 | ≤ 0.10 | pass |
| adversarial null | `misspecified_baseline` | 0.10 [0.06, 0.17] | 0.10 | -0.0003 | +0.1484 | 0/99/1 | ≤ 0.10 | pass |
| adversarial null | `interaction_meta` | 0.09 [0.05, 0.16] | 0.09 | -0.0055 | +0.2050 | 0/0/100 | ≤ 0.10 | pass |
| adversarial null | `difficulty_by_model` | 0.04 [0.02, 0.10] | 0.04 | -0.0008 | -0.0007 | 99/0/1 | ≤ 0.10 | pass |
| adversarial null | `scale_by_model` | 0.04 [0.02, 0.10] | 0.05 | -0.0007 | -0.0007 | 98/2/0 | ≤ 0.10 | pass |
| adversarial null | `simpson` | 0.05 [0.02, 0.11] | 0.05 | -0.0009 | -0.0009 | 100/0/0 | ≤ 0.10 | pass |
| planted signal | `weak_signal` | 0.82 [0.73, 0.88] | 0.82 | +0.0087 | +0.0087 | 97/3/0 | report only | pass |
| planted signal | `moderate_signal` | 1.00 [0.96, 1.00] | 1.00 | +0.0330 | +0.0331 | 98/2/0 | ≥ 0.80 | pass |
| planted signal | `strong_signal` | 1.00 [0.96, 1.00] | 1.00 | +0.1011 | +0.1011 | 100/0/0 | ≥ 0.80 | pass |
| planted signal | `signal_with_difficulty` | 0.99 [0.95, 1.00] | 0.99 | +0.0294 | +0.0296 | 94/5/1 | ≥ 0.80 | pass |
| planted signal | `hard_only_signal` | 1.00 [0.96, 1.00] | 1.00 | +0.0288 | +0.0289 | 96/4/0 | ≥ 0.50 | pass |

Overall: **all acceptance criteria met**.

## Refit-bootstrap CI (40 replicates, 100 refit bootstraps each; run on v0.2.2)

| Scenario | Coverage of the mean point estimate [95% CI] | CI excludes 0 | Mean width | Acceptance 0.85–0.99 |
|---|---|---|---|---|
| `noise` | 1.000 [0.91, 1.00] | 0.00 | 0.017 | **FAIL** |
| `moderate_signal` | 0.925 [0.80, 0.97] | 0.95 | 0.043 | pass |
| `misspecified_baseline` | 1.000 [0.91, 1.00] | 0.00 | 0.022 | **FAIL** |

## What this shows and what it does not

- **The ≤ 0.10 flag-rate threshold is a development acceptance criterion, not proof of calibration.** With 100 replicates, a test whose true rate is 0.05 passes about 98% of the time; one whose true rate is 0.08 also usually passes.
- **Supported:**
  - no material anti-conservative behaviour under the tested nulls, including detectors that copy the metadata, are nonlinear or interaction functions of it, or differ in scale or base rate by model;
  - planted signal of 0.6 SD or more is detected.
- **Bootstrap CIs are conservative under pure noise and under a misspecified baseline** (coverage 1.00 and 1.00 at n = 40, above the 0.99 band, so those rows show FAIL); coverage under a moderate signal is 0.925. Exact 95% calibration is not claimed. The coverage run used code version 0.2.2.
- **Borderline:** the misspecified-baseline null is flagged at exactly the 0.10 acceptance limit (10/100, Wilson CI shown above). With a true rate of 0.05, 10/100 is unusual (about 3%), so a mild anti-conservative tendency there cannot be ruled out.
- **"Coverage" is measured against the mean of the point estimates across replicates,** i.e. the estimator's own expectation at that sample size, not a population-level true increment.
- **Not tested:**
  - more than 2 baseline columns;
  - leakage planted inside the detector;
  - fewer than 60 groups;
  - real-data dependence structures beyond groups.

## `check` battery (40 replicates; v0.3.0; settings {'n_boot': 50, 'n_perm': 99, 'seeds': 2})

Rate at which each check returned SUPPORTED. Data: 300 questions x 4 models (see `simulations/check_suite.py`).

| Scenario | SUPPORTED rate per check | Acceptance | Result |
|---|---|---|---|
| `pure_difficulty` | your difficulty columns: 0.00; response length: 1.00; model identity: 1.00; all of the above together: 0.00; same question, across models: 0.00 | your difficulty columns <= 0.10 | pass |
| `duplicated` | your difficulty columns: 0.00; response length: 1.00; model identity: 1.00; all of the above together: 0.00; same question, across models: 0.00 | your difficulty columns <= 0.10 | pass |
| `nonlinear` | your difficulty columns: 0.00; response length: 1.00; model identity: 1.00; all of the above together: 0.00; same question, across models: 0.00 | your difficulty columns <= 0.10 | pass |
| `pure_length` | your difficulty columns: 1.00; response length: 0.00; model identity: 1.00; all of the above together: 0.00; same question, across models: 1.00 | response length <= 0.10; all of the above together <= 0.10 | pass |
| `pure_model` | your difficulty columns: 0.03; response length: 0.07; model identity: 0.00; all of the above together: 0.00; same question, across models: 0.03 | same question, across models <= 0.10; model identity <= 0.10 | pass |
| `planted` | your difficulty columns: 1.00; response length: 1.00; model identity: 1.00; all of the above together: 1.00; same question, across models: 1.00 | your difficulty columns >= 0.80; same question, across models >= 0.80 | pass |
| `planted_noisy` | your difficulty columns: 1.00; response length: 1.00; model identity: 1.00; all of the above together: 0.80; same question, across models: 1.00 | report only | pass |

Overall: **all acceptance criteria met**.

## History

- **v0.1** (linear baseline only): a detector that was a nonlinear function of the metadata got +0.088, flagged in 100/100 runs. The CI came from a single CV seed.
- **v0.2.0:** added flexible baselines, the headline chosen without the detector, and a refit bootstrap. Its battery flagged an interaction-only detector in 17% of runs: an unconfined permutation makes the null "pure noise", which a costless metadata function beats.
- **v0.2.1:** permutations confined to 10 bands of the headline baseline's own out-of-fold risk; a flag requires p < 0.05 and increment > 0. Its battery made the interaction null *worse* (27% flagged, p < 0.05 in 66%). A detector that is an exact metadata function beats any baseline that only approximates that function at finite n, and within-band shuffling destroys exactly the fine ordering where it wins.
- **v0.3.0:** `check` command (status per cheap explanation, Holm over checks, same-question test, shipped difficulty tables joinable by question-text hash, HTML/JSON report). The `audit` engine is unchanged from 0.2.2.
- **v0.2.2:** conditional-randomisation null. Fit S ~ baselines out of fold (gradient-boosting regressor); null scores = fitted S + group-permuted residuals. Every null detector carries the same metadata function as S. Interaction null 0.27 → 0.09; all 18 scenarios within their limits. The misspecified-baseline null sits exactly at the 0.10 limit: a pass, but a marginal one.

## v0.4 validity estimands (`simulations/validity_suite.py`, code 0.4.0)

Simulation: logit p = θ_q + a_m + γ_mq, 10 sampled labels per (model, question), split 5/5. Acceptance criteria were fixed in the script and committed (`068951a`) before the first run.

| Criterion | Rule | Run 1 (40 reps, failed) | Run 2 (100 reps) |
|---|---|---|---|
| R2_v CI lower bound > 0 for a pure-noise detector | <= 0.10 | 1.000–1.000 (**FAIL**) | 0.000–0.010 (pass) |
| R2_v CI lower bound > 0 for a planted detector (signal 1.0) | >= 0.80 | 1.000–1.000 (pass) | 1.000–1.000 (pass) |
| question-bootstrap 95% CI of the share covers the noise-free share | 0.85-0.99 | 1.000–1.000 (**FAIL**) | 0.910–0.970 (pass) |
| |mean excess share| with no interaction and ability spread up to 2.5 logit sd | <= 0.05 | max 0.0035 (pass) | max 0.0033 (pass) |
| mean excess share with interaction sd 1.0 | >= 0.10 | 0.322–0.339 (pass) | 0.326–0.339 (pass) |
| |mean delta(k)| for a pure-noise detector, every k | <= 0.01 | max 0.0060 (pass) | max 0.0060 (pass) |

**Run 1 failures and fixes (both post hoc, recorded here; run 1 outputs kept as `validity_v0.4.0_run1_failed.*`):**
1. **R²_v false recovery 1.00.**
   - The detector slope was fitted on the same data it scored, so R²_v ≥ 0 by construction and its bootstrap CI can never include 0. A pure-noise detector always "excluded 0" (estimate about 0.0004).
   - **Fix (tool):** cross-half R²_v. The slope is fitted on one sample half and scored on the other, as in the DP study. Noise detectors are now centred on 0.
   - **Real-data consequence:** the H-2V R²_v values used the in-sample slope. With one slope over more than 10,000 entries the bias is about 1/N (about 0.0001), so they are unchanged in practice.
2. **Coverage 1.00.**
   - The suite's truth was the share on the *same* simulated questions. A question bootstrap measures variability over question sampling, so that target was wrong.
   - **Fix (suite):** the population share from 200,000 questions with the same parameters.
   - The tool's CI was not changed.

**Not tested:** fewer than 5 samples per half; label noise in the sample labels; latent-scale shares (the tool reports the probability scale plus the additive null, not a latent fit).


## Re-check of the misspecified-baseline null (19 Nov 2026, version 0.5.0 engine, unchanged since 0.2.2)

At 100 runs this null flagged 0.10 [0.06, 0.17], the only interval above the nominal α of 0.05. It was re-run with **400 runs** (`simulations/results/results_v0.5.0_misspecified_400.json`):

- **flag rate 0.065 (26/400), Wilson 95% [0.045, 0.094]**;
- the headline family was spline in 384 of 400 runs.

**Reading:**
- The interval includes 0.05, so the earlier 0.10 was mostly Monte Carlo error.
- The point estimate stays slightly above nominal. Under a misspecified baseline (the label depends on difficulty², and the baselines approximate it), the test may be mildly liberal.
- This is documented as a limitation. When a baseline's relationship to the label may be strongly nonlinear, use the spline or gbm family (the headline already chooses spline here).

**Disclosure:** this re-run finished before its interpretation was written down, so the reading above was fixed after seeing the result.

## 0.7.0 additions (pre-registered in `docs/prereg/PREREG_0.7.md`, commit 412317b; engine at 86df237)

### Fast preset (`--fast`): ADOPTED

**Settings:** 2 seeds, no interval refits, and up to 200 permutations. Permuting stops (Besag & Clifford 1991) once 10 null increments reach the observed one.

**Run:** 100 replicates per scenario. Output: `simulations/results/results_v0.7.0_fast.json`.

**Result:** every pre-registered criterion is met.

| Scenario | Flag rate, fast [95% CI] | Flag rate, v0.2.2 suite | Mean permutations drawn | Result |
|---|---|---|---|---|
| `noise` | 0.06 [0.03, 0.12] | 0.06 | 40 | pass |
| `metadata_copy` | 0.03 [0.01, 0.08] | 0.05 | 45 | pass |
| `nonlinear_of_metadata` | 0.01 [0.00, 0.05] | 0.01 | 34 | pass |
| `metadata_plus_noise` | 0.04 [0.02, 0.10] | 0.03 | 43 | pass |
| `random_labels` | 0.01 [0.00, 0.05] | 0.03 | 32 | pass |
| `few_groups` | 0.08 [0.04, 0.15] | 0.07 | 48 | pass |
| `unequal_groups` | 0.03 [0.01, 0.08] | 0.03 | 37 | pass |
| `imbalance` | 0.06 [0.03, 0.12] | 0.05 | 37 | pass |
| `misspecified_baseline` | 0.10 [0.06, 0.17] | 0.10 | 47 | pass |
| `interaction_meta` | 0.08 [0.04, 0.15] | 0.09 | 51 | pass |
| `difficulty_by_model` | 0.04 [0.02, 0.10] | 0.04 | 42 | pass |
| `scale_by_model` | 0.05 [0.02, 0.11] | 0.04 | 47 | pass |
| `simpson` | 0.05 [0.02, 0.11] | 0.05 | 38 | pass |
| `weak_signal` | 0.81 [0.72, 0.87] | 0.82 | 177 | pass |
| `moderate_signal` | 1.00 [0.96, 1.00] | 1.00 | 200 | pass |
| `strong_signal` | 1.00 [0.96, 1.00] | 1.00 | 200 | pass |
| `signal_with_difficulty` | 0.98 [0.93, 0.99] | 0.99 | 198 | pass |
| `hard_only_signal` | 1.00 [0.96, 1.00] | 1.00 | 200 | pass |

**What the stopping rule changes:**
- On the no-signal scenarios, it stops after about 33–51 of the 200 permutations.
- With real signal, nearly all 200 are drawn.
- It can never create a flag: an early stop always gives p ≥ 10/200 = 0.05.
- `misspecified_baseline` sits at the 0.10 limit, the same as in v0.2.2. Its 400-run re-check gave 0.065.

### Half-sample interval (`interval="half"`): REJECTED, interval demoted

**Settings:** 40 replicates, 100 half-sample refits each. Output: `simulations/results/coverage_v0.7.0_half.json`.

| Scenario | Coverage of the mean estimate [95% CI] | Mean width | Acceptance 0.85–0.99 |
|---|---|---|---|
| `noise` | 1.000 [0.91, 1.00] | 0.016 | **FAIL** |
| `moderate_signal` | 0.900 [0.77, 0.96] | 0.045 | pass |
| `misspecified_baseline` | 1.000 [0.91, 1.00] | 0.018 | **FAIL** |

**What the half-sample interval does:**
- It is about as wide under the null as the refit bootstrap: 0.016 vs 0.017 on `noise`.
- It fails the same two scenarios, `noise` and `misspecified_baseline`.

**Diagnosis:** the over-coverage is not caused by duplicated groups in the bootstrap. The registration allowed no second candidate.

**What changes, as pre-registered:**
- The refit bootstrap stays the default interval.
- Reports label it as conservative and not used for the decision.
- The README no longer presents it as a calibrated confidence interval.
- `interval="half"` remains available, documented as failing this check.

## 0.8.0: `compare` (detector release comparison), pre-registered in `docs/prereg/PREREG_0.8.md` (commit 3b69b76)

**Settings:** 100 replicates per scenario, `--fast`-equivalent (2 seeds, permutation stopping at 10 of up to 200), all three families. The baselines are task and L. Output: `simulations/results/compare_v0.8.0.json`.

**What each cell shows:** the flag rate with its 95% interval. The italic label is the scenario's role for that test.

| Scenario | T1 improvement | T2 regression | T3 standalone | T3 coverage |
|---|---|---|---|---|
| `reseed` | 0.05 [0.02, 0.11] *null* | 1.00 [0.96, 1.00] *positive* | 0.90 [0.83, 0.94] *—* | — |
| `monotone` | 0.00 [0.00, 0.04] *null* | 0.00 [0.00, 0.04] *null* | 0.00 [0.00, 0.04] *null* | — |
| `interaction` | 0.01 [0.00, 0.05] *adversarial null* | 0.12 [0.07, 0.20] *(null, not registered)* | 0.40 [0.31, 0.50] *—* | — |
| `coarsened` | 0.02 [0.01, 0.07] *null* | 0.82 [0.73, 0.88] *positive* | 0.99 [0.95, 1.00] *—* | — |
| `strict_upgrade` | 1.00 [0.96, 1.00] *positive* | 0.06 [0.03, 0.12] *null* | 0.91 [0.84, 0.95] *positive* | 0.96 |
| `complementary` | 1.00 [0.96, 1.00] *positive* | 1.00 [0.96, 1.00] *positive* | 0.05 [0.02, 0.11] *null* | 0.95 |

**Result:**
- **Ship-blocking criteria:** all met.
  - T1 is ≤ 0.10 on every null.
  - T2 is ≤ 0.10 on `strict_upgrade` and `monotone`.
  - T3 is ≤ 0.10 on `complementary` and `monotone`.
- **Power targets:** all met.
  - T1 is 1.00 on `strict_upgrade` and `complementary`.
  - T2 is 1.00, 0.82 and 1.00 on `reseed`, `coarsened` and `complementary`.
  - T3 is 0.91 on `strict_upgrade`.
- **T3 interval coverage:** 0.96 and 0.95. T3 is therefore a decision, not only a description.

**One observation outside the registration:**
- **Where:** in `interaction`, v2 = v1 × a task-specific scale, so v1 is recoverable from v2 and the task. That makes T2 a null there too, but the registration gave it no rule.
- **What happened:** T2 flagged 0.12 [0.07, 0.20]. That is slightly above the 0.10 used for the registered nulls.
- **How to read it:** when one version is a task-dependent rescaling of the other, a reported regression may be a false alarm about 1 time in 8.
- **What was done:** it is stated in the README. Nothing was changed.

## 0.11.0: ordinal outcome and raw-score calibration, pre-registered in `docs/prereg/PREREG_0.11.md` (commit c522f40; code at 924167c)

### `--outcome ordinal`: ADOPTED

**Settings:** 100 replicates per scenario, `--fast` settings, all three families. Output: `simulations/results/ordinal_v0.11.0.json`.

| Scenario | Flag rate [95% CI] | Mean increment (C) | Headline linear/spline/gbm | Rule | Result |
|---|---|---|---|---|---|
| `noise` | 0.07 [0.03, 0.14] | -0.0005 | 100/0/0 | <= 0.10 (ship-blocking) | pass |
| `metadata_copy` | 0.08 [0.04, 0.15] | -0.0006 | 98/2/0 | <= 0.10 (ship-blocking) | pass |
| `nonlinear_of_metadata` | 0.06 [0.03, 0.12] | -0.0008 | 0/95/5 | <= 0.10 (ship-blocking) | pass |
| `interaction_meta` | 0.06 [0.03, 0.12] | -0.0044 | 0/0/100 | <= 0.10 (ship-blocking) | pass |
| `few_groups` | 0.05 [0.02, 0.11] | -0.0039 | 89/6/5 | <= 0.10 (ship-blocking) | pass |
| `weak_signal` | 1.00 [0.96, 1.00] | +0.0215 | 98/1/1 | reported | pass |
| `moderate_signal` | 1.00 [0.96, 1.00] | +0.0710 | 98/2/0 | >= 0.80 (power target) | pass |

**Result:** every ship-blocking null is ≤ 0.10, the highest being 0.08, and the power target is met.
- **Adversarial nulls:** the nonlinear and interaction scenarios pick the spline and gbm headline as designed.
- **Weak signal:** detection is 1.00. Its 0.3 factor multiplies a 0–3 severity, so it is not the same effect size as the binary suite's weak signal.

### Raw-score calibration intervals (`--calibration`): ADOPTED

**Settings:** 200 replicates. Each has 300 groups of 2 items sharing p ~ Beta(2, 3), and the score is p itself, so it is calibrated by construction. Each interval uses 2,000 group-bootstrap draws. Output: `simulations/results/calcov_v0.11.0.json`.

| Check | Result | Acceptance |
|---|---|---|
| Slope interval contains 1 | 0.955 | 0.85–0.99: pass |
| Calibration-in-the-large interval contains 0 | 0.945 | 0.85–0.99: pass |
| Score p² (miscalibrated): slope interval excludes 1 | 0.91 (mean slope 0.71) | reported |

## Stress test after 0.11.1, pre-registered in `docs/prereg/PREREG_VALIDATION.md` Part B (commit 01826ec)

**Settings:** `--fast` (2 seeds, permutation stopping at 10 of up to 200), all three families. Output: `simulations/results/stress_v0.11.1.json`.

| Scenario | Role | Replicates | Rows / groups | Flag rate [Wilson 95%] | Seconds per replicate (1 core) | Verdict |
|---|---|---|---|---|---|---|
| `many_baselines_noise` | S1a null | 100 | 600 / 600 | 0.030 [0.010, 0.085] | 2.6 | pass |
| `many_baselines_copy` | S1b null | 100 | 600 / 600 | 0.030 [0.010, 0.085] | 2.3 | pass |
| `large_noise` | S2a null | 100 | 10000 / 5000 | 0.010 [0.002, 0.054] | 12.4 | pass |
| `large_copy` | S2b null | 100 | 10000 / 5000 | 0.090 [0.048, 0.162] | 19.3 | pass |
| `uneven_noise` | S3a null | 100 | 4004 / 1000 | 0.050 [0.022, 0.112] | 4.3 | pass |
| `uneven_copy` | S3b null | 100 | 4001 / 1000 | 0.060 [0.028, 0.125] | 4.4 | pass |
| `clustered_noise` | S4a null | 100 | 1800 / 600 | 0.070 [0.034, 0.137] | 2.7 | pass |
| `clustered_shared_noise` | S4b null | 100 | 1800 / 600 | 0.020 [0.006, 0.070] | 2.9 | pass |
| `many_baselines_signal` | S1c power | 100 | 600 / 600 | 1.000 [0.963, 1.000] | 6.7 | reported |
| `large_weak` | S2c power (0.15·y) | 100 | 10000 / 5000 | 1.000 [0.963, 1.000] | 62.6 | reported |
| `misspecified_baseline` | S5 borderline re-run | 400 | 600 / 300 | 0.068 [0.047, 0.096] | 4.3 | pass |
| `interaction_meta` | S5 borderline re-run | 400 | 600 / 300 | 0.095 [0.070, 0.128] | 22.0 | borderline |
| `few_groups` | S5 borderline re-run | 400 | 120 / 60 | 0.048 [0.031, 0.073] | 2.2 | pass |

**Result:**
- **S1–S4:** every null is within the 0.10 limit. That covers 10 baselines (7 of them noise), 10,000 rows, groups of 1 to 40 items, and clustered detector noise.
  - **Largest:** `large_copy`, at 0.09. Its interval reaches 0.16, so 100 replicates cannot rule out a true rate above 0.10 there.
- **Power:** 1.00 with 10 baselines, and 1.00 for a 0.15·y signal at 10,000 rows.
- **Runtime:** about 12–20 s per replicate on one core at 10,000 rows, or 63 s when the signal makes all 200 permutations run.
- **S5 at 400 replicates:**
  - `misspecified_baseline` passes, at 0.0675 [0.047, 0.096]. This confirms that the 0.10 seen at 100 replicates was noise.
  - `few_groups` passes, at 0.0475.
- **`interaction_meta` is BORDERLINE, at 0.095 [0.070, 0.128]:**
  - **The case:** the detector is an exact function of an interaction between two baselines (task × L).
  - **What the interval shows:** its lower bound is above 0.05. So in this case the test is anti-conservative, flagging at roughly twice the nominal rate, although it stays at the 0.10 development limit.
  - **What this means for users:** when a detector could be an interaction of the supplied baselines, add that interaction as an explicit baseline column, or read p-values near 0.05 with care.
  - **Fix:** none here. A fix needs a new registration.

## Real-data validation after 0.11.1, pre-registered in `docs/prereg/PREREG_VALIDATION.md` Part A + addendum 1

The full write-up is [`docs/validation.md`](docs/validation.md); the raw output is `validation/results.json`.

- **Agreement with independent analyses: AGREES.** All seven detectors match the prior verdict and fall within 0.006 of its increment, and the ordering check passes.
- **Real-data nulls:** six of six are within 0.10. Pooled, the rate is 32/600 = 0.053 [0.038, 0.074]. Planted-signal power is 1.00 on all three tables.

## 0.12 and 0.13: null, interval, judges, diagnostic (pre-registered: `PREREG_0.12.md`, `PREREG_DIAG.md`, `PREREG_JUDGES.md`, `PREREG_TREC.md`, `PREREG_0.13.md`)

Run on Kaggle CPU with pinned dependencies. The commit and environment for each run are in `simulations/results/kaggle_run_info_*.json`.

**Super-learner null (`residual_sl`): not adopted.**
- **Where it helped:** it fixed the development case (interaction 0.0475, Wilson [0.031, 0.073]). On fresh seeds it passed the full suite, `compare`, ordinal, the stress test and the real-data agreement and nulls.
- **Where it failed:** held-out exact-function detectors (400 runs each) — product 0.1475, threshold 0.1125, three-way 1.00.
- **The old null on the same cases:** 0.1475, 0.1025 and 1.00. Both fail alike (`stress_v0.12.0_sl.json`, `stress_v0.12.0_heldout_gbm.json`).

**Interval for the gain (clustered DeLong): adopted.**

| Scenario | Coverage |
|---|---|
| noise | 0.965 |
| moderate | 0.910 |
| misspecified | 0.965 |
| weak | 0.905 |
| strong | 0.950 |
| signal + difficulty | 0.930 |

The band is 0.85–0.99 (`coverage_v0.12.0_delong.json`). From 0.13 this interval is the primary verdict.

**Deterministic-function diagnostic** (200 runs per cell; `diag_v0.12_*.json`). False-flag rate of the conditional-information test, default null, at 600 / 3,000 items:

| Score s = f(b) + noise | 600 | 3,000 |
|---|---|---|
| D1 linear | 0.07 | 0.035 |
| D2 sum | 0.085 | 0.075 |
| D3 product | 0.10 | 0.07 |
| D4 three-way | 1.00 | 1.00 |
| D5 threshold | 0.045 | 0.085 |
| D6 threshold × L | 0.09 | 0.04 |
| D7 exponential | 0.065 | 0.07 |
| D8 XOR | 0.745 | 0.37 |
| D9 sin + square | 0.12 | 0.09 |
| D10 task-specific slope | 0.575 | 0.325 |
| N0 noise | 0.05 | 0.04 |

The super-learner null differs only on D10 (0.265 / 0.28) and D7. Power on genuine signal (P1) is 0.96–0.965.

**LLM judges on human labels** (`docs/judges.md`).
- **The calibration control failed:** the TREC no-information judge was flagged in 33% of runs.
- **Stratifying by query did not fix it** (`PREREG_TREC.md`): 0.29. Query × length deciles gave 0.07, but that is descriptive only.
- **WQA judges with rule labels:** the null controls passed (0.04–0.06).

**Omitted-baseline sensitivity: validated** (`sensitivity_v0.13.0.json`).
- **V1:** 0.980 pooled (per family: linear 1.00, nonlinear 1.00, categorical 1.00, interaction 0.938).
- **V2:** 0.97. **V3:** ρ = 0.89.

**Sample-size planner: under validation** (`planner_v0.13.0_S3.json`). S3 passed: P1 and P2 hold in all three cells, and independent power at the recommended size is 0.88. S1 and S2 ran out of time and are being re-run in stages (`simulations/planner_staged.py`).
