# R1b pre-registration: a refit-aware interval for the held-out gain

Written and committed before any R1b run and before the R1-a results were seen. R1 itself is unchanged; its rule and its result stand whatever R1b finds. This experiment asks whether one fully specified alternative interval has acceptable coverage. It is not a search for an interval that makes a failing cell pass.

## 1. Why

**What R1 found.** In R1 (`PREREG_R1.md`), the production clustered-DeLong interval under-covered in D4 at 3,000 items, the three-way interaction s = L1·L2·sign(task) + noise:
- coverage was 0.695 [0.628, 0.755];
- misses fell on both sides (38 below, 23 above);
- the real spread of the gain across datasets was 1.79 times the interval's mean standard error, against 0.79–1.23 in the 12 passing cells known at the time of writing.

**Diagnosis (a hypothesis, not a result).** The DeLong standard error treats the out-of-fold predictions as fixed. It captures test-sampling variation given the fitted models. It does not capture variation from re-estimating the models on a new training sample. When the gain comes from a flexible model partly learning an interaction, that fitting variation is large.

**Science question.** What uncertainty must an interval for a held-out incremental-prediction estimator include when the prediction functions are themselves estimated? This is the question Bates, Hastie & Tibshirani (2024, *JASA*) raise for cross-validated prediction error.

## 2. Estimands, fixed now

**Primary: unconditional Δ_d.** For cell d this is E[Δ̂], the expected value of the production estimator `increment` for the whole procedure at the cell's size:
- generate data;
- draw grouped folds;
- fit all families with and without s;
- pick the headline family;
- take the mean over the 2 CV seeds of the out-of-fold AUROC gain.

This is the R1 estimand. **R1's reference values are reused unchanged**: 1,000 replicates per cell, seed salt 32, in `simulations/results/r1_bi-r1-*.json`.

**Secondary, diagnostic only: conditional Δ_cond(D).** For one dataset D, this is the mean over the 2 seeds × 5 folds of AUROC(fold model with s) − AUROC(fold model without s), in the headline family chosen on D. Each fold model is fitted on its training folds of D and scored on an independent sample of 20,000 items from the same generator (seed salt 33).
- It is computed only for the D-series and control cells, whose generator can produce a fresh sample.
- It approximates the "given these fitted models" target: it averages per-fold AUROCs instead of pooling out-of-fold predictions.
- **It is reported, and drives no decision.** It separates two readings of the D4 failure: "DeLong is right for a different estimand" versus "DeLong is wrong for both".

## 3. The candidate interval (one candidate only)

**Refit group bootstrap, normal form:** Δ̂ ± 1.96 · SD(Δ*₁ … Δ*_B), with B = 100. Each draw Δ*_b is the production function `core._boot_once` with seed 1000 + b. It performs these steps:
1. Resample the groups with replacement. The resampling unit is the group, the unit the CV already keeps intact.
2. Copies of a group keep that group's id, so they always share a fold. Duplicated items can never sit in both train and test.
3. Draw new grouped folds with the same 2 CV seeds.
4. Refit every family for the baselines-only model and for baselines + s.
5. Choose the headline family again.
6. Compute the gain.

Every stage that varies in the real procedure varies in the bootstrap.

**Reported alongside, descriptively, from the same draws (neither can be adopted):**
- the percentile form [q2.5, q97.5];
- the production DeLong interval on the same datasets.

**Risk stated in advance.** The percentile refit bootstrap was **over-conservative under the null** in v0.2.2: coverage 1.00 at pure noise (`CALIBRATION.md`). Each bootstrap training sample holds only about 63% unique groups, so its fitting variation can exceed the real procedure's. The normal form may therefore over-cover in the control cells. That outcome is allowed for in §6.

## 4. Cells (fixed now)

| Cells | Why |
|---|---|
| D4@600, D4@3000 | the failing structure |
| N0@600, N0@3000 | the null control |
| P1@600, P1@3000, P2@600, P2@3000 | the genuine-signal controls |
| D10@3000, D6@3000 | the passing cells with the largest spread ratio (1.16, 1.23) |
| **every other cell that the merged R1 result rules as failing** | determined mechanically from `r1_coverage.json` |

## 5. Replicates and computation

- **Datasets.** The same 200 coverage datasets per cell as R1 (`r1_coverage.make(cell, rep, salt 31)`), so DeLong and the candidate are compared on identical data.
- **Settings.** The audit is called with the frozen R1 settings. B = 100 bootstrap draws per replicate.
- **Runs.** `simulations/r1b_refit.py` on Kaggle CPU, one cell per kernel for 100 replicates (two kernels per cell), checkpointed every 8 replicates.
- **Timing.** Measured locally: about 3 s per draw at both sizes. A replicate takes about 5 minutes.
- **Failures.** A replicate that errors is rerun once with the same seed. If it fails again, it is reported as a technical failure and excluded, with the count shown. No replicate is dropped for any statistical reason.

## 6. Decision (fixed now)

**Cell rule** (as in R1): the candidate *passes* a cell if 0.85 ≤ coverage ≤ 0.99. It *under-covers* below 0.85 and is *over-conservative* above 0.99.

Outcomes:
1. **Eligible for default.** It passes every R1b cell.
   - It becomes the default only after the remaining R1 cells pass under the same protocol, as a separately registered R1c with the same rule.
   - Until then it ships as an opt-in, `--interval refit`.
2. **Opt-in, labelled conservative.** It passes every R1-failing cell, and in each control cell (N0, P1, P2) its coverage is at least 0.85, but at least one cell is over-conservative.
   - It is offered as `--interval refit` and described as conservative.
   - It is recommended where the report already warns that the gain over the linear baseline differs from the headline gain, the sign of a nonlinear gain.
   - The default stays DeLong, with R1's scope restriction.
3. **Not adopted.** The candidate under-covers in any R1-failing cell or in any control cell.
   - The R1 scope restriction stays.
   - The result is reported as an open methodological limitation.
   - No second candidate is tried under this registration.

The conditional-coverage diagnostic (§2) and the percentile form are reported in every case and drive nothing.

**Cost.** Mean runtime per audit is reported. It is not part of the decision, but if the candidate is adopted its cost is stated next to the option.

## 7. No tuning

None of the following changes after the first R1b result is seen:
- B, or the normal form;
- the resampling unit or the fold handling;
- the seeds, cells, replicates or references;
- the acceptance band.

Any deviation is recorded with the original text, the change, the reason, and whether it could affect the result.

## 8. Reporting

Results go to:
- `simulations/results/r1b.json`, with per-cell coverage, Wilson intervals, misses below and above, mean widths, the spread of the gain against the mean bootstrap SD, and conditional coverage;
- an R1b section in `CALIBRATION.md`;
- the validity envelope and README, updated only by the rule in §6.

Whatever the outcome, the paper reports R1's D4 failure and the R1b outcome together.

## 7. Result (observed 2026-11-27)

All 10 cells completed. 200 replicates each. No technical failures.

| Cell | Refit coverage [95% Wilson] | Status | DeLong coverage | DeLong status |
|---|---|---|---|---|
| D4@600 | **0.985** [0.957, 0.995] | pass | 0.845 | under-covers |
| D4@3000 | **0.985** [0.957, 0.995] | pass | 0.695 | under-covers |
| N0@600 | 1.000 [0.981, 1.000] | over-conservative | 0.960 | pass |
| N0@3000 | 1.000 [0.981, 1.000] | over-conservative | 0.935 | pass |
| P1@600 | **0.980** [0.950, 0.992] | pass | 0.955 | pass |
| P1@3000 | **0.985** [0.957, 0.995] | pass | 0.940 | pass |
| P2@600 | **0.980** [0.950, 0.992] | pass | 0.870 | pass |
| P2@3000 | **0.975** [0.943, 0.989] | pass | 0.950 | pass |
| D10@3000 | 1.000 [0.981, 1.000] | over-conservative | 0.930 | pass |
| D6@3000 | **0.990** [0.964, 0.997] | pass | 0.910 | pass |

**Verdict: Outcome 2 — opt-in, labelled conservative.**

Both D4 cells pass (0.985 each). Every control cell (N0, P1, P2) has refit coverage ≥ 0.85. Three cells are over-conservative: N0@600, N0@3000 and D10@3000, all at exactly 1.000. This matches the §6 risk stated in advance: the refit bootstrap draws only ~63% unique groups per bootstrap training set, so its fitting variation can exceed the real procedure's under the null.

The refit-group-bootstrap interval ships as `--interval refit` and is described as conservative. It is recommended when the report's nonlinear-gain warning fires (the case where the linear-baseline gain exceeds the headline gain by more than 0.02), which in simulation fired in 9/10 and 10/10 D4 runs. The default interval stays clustered DeLong, with R1's scope restriction on D4 documented.

Result file: `simulations/results/r1b.json`.

## References

- Bates, S., Hastie, T. & Tibshirani, R. (2024). Cross-validation: what does it estimate and how well does it do it? *JASA* 119(546).
- Efron, B. & Tibshirani, R. (1997). Improvements on cross-validation: the .632+ bootstrap method. *JASA* 92(438).
