# R1 pre-registration: coverage of the primary gain interval across hard structures

Written and committed before any R1 run. The engine is v0.13.0 (tag `v0.13.0`, commit f8eaa9a). No estimator, interval, model, fold, seed or confidence-level change is permitted during R1.

## 1. Question

Does the production 95% interval for the held-out gain (clustered DeLong, PREREG_0.12 Part B) achieve its intended coverage across the data structures the tool may meet? That includes the structures where the secondary conditional-information test fails.

R1 tests only the interval attached to the primary estimand. It does not test whether a detector holds information beyond the baselines.

## 2. Hypotheses

- **Primary.** For every structure × size cell d, the empirical coverage Ĉ_d of the true expected gain lies within [0.85, 0.99].
- **Secondary.** The same holds for uneven group sizes, clustered noise, rare positives and very few groups.

The band is the one used for every earlier interval check (0.2.2, 0.7.0, 0.12 Part B). It allows for simulation error while ruling out material under-coverage and implausible conservatism.

## 3. Why new data are needed

Coverage was established on six simple scenarios only (`coverage_v0.12.0_delong.json`). The structures where the conditional-information test fails were never checked for the interval. They include:
- a task-specific slope;
- XOR;
- a three-way interaction, which produces a real model-class gain that shrinks with n.

Those failures say nothing either way about the interval, so the interval has to be tested on them directly.

## 4. Frozen implementation

Each replicate calls the production `baseline_increment.audit` at the validated preset:
- 2 seeds and all three families;
- `interval="delong"` and `null="residual_gbm"`;
- `n_boot=0`, `n_boot_desc=10`;
- `n_perm=1`, `perm_stop=None`.

A single permutation is used because the gain and its interval do not depend on the permutation draws. This runs the identical code path for `increment` and `increment_ci` at a fraction of the cost.

R1 uses no permutation p-values, no `adds_information`, no bootstrap interval, and no alternative interval chosen after the fact.

## 5. Structures (frozen generators, unchanged)

| Source | Cells |
|---|---|
| `simulations/diag_suite.py` (`PREREG_DIAG`, commit 49536db) | D1–D10, N0, P1, P2, at n = 600 and n = 3,000 (26 cells) |
| `simulations/stress_suite.py` (`PREREG_VALIDATION` Part B) | `uneven_noise`, `uneven_copy` (uneven groups of 1–40, about 4,000 rows); `clustered_noise`, `clustered_shared_noise` (clustered noise, 1,800 rows) |
| `simulations/suite.py` (0.2 suite) | `imbalance` (rare positives, about 5.5%); `few_groups` (60 groups) |

That is 32 cells, with no new structure.

## 6. True expected gain: the reference procedure, fixed now

- **Estimand.** For cell d, Δ_d = E[Δ_r], the expected value of the production estimator `increment`: the mean over CV seeds of the out-of-fold AUROC gain of the headline family. It is taken under repeated sampling from the cell's generator at its stated size. This is the same target as every earlier coverage run, now estimated independently of the replicates being scored.
- **Reference.** R_ref = 1,000 independent datasets per cell, on seeds disjoint from the coverage replicates (seed salt 32). For each, compute `increment` with the same frozen call. The reference is Δ̂_d = mean(increment), with Monte Carlo SE = SD / √1000.
- **Precision rule.** If the reference SE exceeds 10% of the cell's mean interval width, the cell is **not estimable**: reported, but neither passed nor failed.

## 7. Coverage replicates

R = 200 independent datasets per cell, seed salt 31. For each: generate, run the frozen call, and record whether `increment_ci` contains Δ̂_d.

Coverage is Ĉ_d = the share that do, reported with its Wilson 95% interval.

Also reported per cell:
- the reference gain and its SE;
- the mean observed gain and the mean interval width;
- the shares of misses above and below.

No seed is dropped for any statistical reason. A replicate that raises an error is rerun once with the same seed. If it fails again, it is reported as a technical failure and excluded, with the count shown.

## 8. Decision (fixed now)

- **Cell rule:** a cell **passes** if 0.85 ≤ Ĉ_d ≤ 0.99. It **under-covers** if Ĉ_d < 0.85, and is **over-conservative** if Ĉ_d > 0.99.
- **PASS:** every estimable cell passes.
- **PASS WITH SCOPE RESTRICTION:** at most 25% of the estimable cells fail, and N0, P1 and P2 all pass at both sizes. Each failing structure is written into the validity envelope as "interval not validated for this structure", and the primary claim is restricted accordingly.
- **FAIL:** more than 25% of the estimable cells fail, or any of N0, P1 or P2 fails at either size. The primary interval then cannot carry the current production scope. Its scope is restricted to the six originally validated scenarios, pending a new pre-registration.

The secondary conditional-information results are reported separately and are not part of this decision.

## 9. No tuning

No confidence level, interval method, model family, fold count, seed count, sample size, generator parameter, acceptance band, replicate count or reference procedure changes after the first R1 result is seen. Every deviation is recorded with the original text, the change, the reason, and whether it could affect the result.

## 10. Reporting

Results go to:
- `simulations/results/r1_coverage.json`, with one row per cell;
- a new R1 section in `CALIBRATION.md`;
- the validity envelope and the README, updated by the decision rule above.

No cell is silently excluded.
