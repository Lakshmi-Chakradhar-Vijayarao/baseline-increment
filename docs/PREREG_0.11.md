# Pre-registration for 0.11.0: graded (ordinal) labels and raw-score calibration

Written and committed before any implementation or calibration run. The binary engine is frozen by `tests/test_golden.py` (commit f9e466f). Nothing below changes it.

## 0. Binary label validation (a fix to all earlier versions)
- **What changes:** the binary path (`audit`, `check`, `compare`, the default CLI) rejects any label outside {0, 1}. The error names `--outcome ordinal`.
- **Why:** until now, a 0–3 severity column was silently mishandled. AUROC treated every level above 0 as positive, while the model fitted four classes and read P(level 1).

## 1. Ordinal labels (`--outcome ordinal`)

**Scope:**
- **Labels:** integer levels with 3 to 10 distinct values. Anything else is an error.
- **Where it works:** the main audit command and `audit_ordinal()` only.
- **Where it doesn't:** `check`, `compare`, `operating_point`, and the report's recall-at-FPR section error out when given ordinal labels.
- **Not covered:** continuous labels.

**Method:** a separate module (`ordinal.py`). It reuses the binary engine's folds, design, score transform and conditional-randomisation null, without modifying them.

**Models,** one per family, each predicting the level as a number:
- `linear`: Ridge regression;
- `spline`: Ridge on the same B-spline features;
- `gbm`: a histogram gradient-boosting regressor with the same shallow settings as the classifier.

**Metric: concordance (Harrell's C for an ordinal outcome).**
- **Definition:** among pairs of items with different levels, the share where the higher-level item has the higher prediction. Prediction ties count ½.
- **Equivalences:** it equals AUROC on 0/1 labels, and must equal a brute-force O(n²) pair count, ties included.

**Statistic:** increment = mean over seeds of [C(baselines + detector) − C(baselines)], out of fold. The headline family is the one with the best baselines-only C.

**Decision:** "adds information" if the Besag–Clifford permutation p < 0.05 and the increment > 0. This is the same rule as the binary engine. There is no refit interval.

**Descriptive output:**
- the detector's own C, with a 2,000-draw group-bootstrap interval;
- counts per level.

**Calibration run,** 100 replicates per scenario:
- **Settings:** the `--fast` settings (2 seeds, at most 200 permutations, stopping at 10, all three families).
- **Data:** 300 groups × 2 items, tasks A, B and C, metadata L. Severity is the latent η + logistic noise, cut at (−1, 0.5, 2) into levels 0–3. e ~ N(0, 1).

| Scenario | η (label) | Detector score | Role |
|---|---|---|---|
| `noise` | task + 0.8L | e | null |
| `metadata_copy` | task + 0.8L | η + 0.3e | null |
| `nonlinear_of_metadata` | task + 1.2(L² − 1) | L² + 0.3e | adversarial null |
| `interaction_meta` | 1.5·L·(+1 if A, else −1) | L·(±1) + 0.3e | adversarial null |
| `few_groups` (60 groups) | task + 0.8L | e | null |
| `weak_signal` | task + 0.8L | e + 0.3·severity | positive |
| `moderate_signal` | task + 0.8L | e + 0.6·severity | positive |

**Acceptance:**
- **Ship-blocking:** every null flags in ≤ 0.10 of runs.
- **Power target:** `moderate_signal` ≥ 0.80. If missed, ordinal still ships, with the measured power stated.
- **Reported only:** `weak_signal`.
- **If any null fails,** ordinal does not ship.

## 2. Raw-score calibration (`--calibration`, binary labels only)

**Inputs:**
- **The score:** read raw, never rank-transformed. Any value outside [0, 1] is an error.
- **Clipping:** scores are clipped to [1e-6, 1 − 1e-6] before taking logits.

**What is reported (all descriptive; no verdict):**
- **Calibration slope:** b in a logistic fit of y ~ a + b·logit(p). 1 is ideal; below 1 means overconfident.
- **Calibration-in-the-large:** the intercept of y ~ offset(logit(p)) + c, with the slope fixed at 1. 0 is ideal; positive means it under-predicts.
- **Accuracy:** Brier score and mean predicted vs observed rate.
- **Reliability table:** 10 equal-count bins.
- **ECE:** shown, explicitly labelled as biased upward at finite n.

**Intervals:** a 2,000-draw group bootstrap (resampling groups) for the slope, calibration-in-the-large and Brier.

**Coverage check,** with data clustered like the intervals assume:
- **Data:** 300 groups × 2 items. The two items in a group share p ~ Beta(2, 3), and each y ~ Bernoulli(p). The score is p itself, so it is calibrated by construction.
- **Run:** 200 replicates.
- **Acceptance:** the slope interval contains 1 in 0.85–0.99 of replicates, and the calibration-in-the-large interval contains 0 in 0.85–0.99.
- **If either fails,** the intervals are not shown; only point estimates are reported, labelled descriptive.
- **Reported only:** with the score p² (miscalibrated), the share of runs whose slope interval excludes 1.

## 3. Calibration matching in reports
- **Match key:** `calibration_for_audit` adds outcome (binary or ordinal) to its match key.
- **Ordinal preset:** ordinal runs match only the new ordinal preset in `calibration_summary.json`.
- **Raw-score calibration:** its report section cites its own coverage run.

## No second attempt
No scenario, threshold or setting changes after the first run under this registration.
