# PLAN2 pre-registration: a sample-size planner for the primary verdict

Written and committed before any PLAN2 run. It replaces `PREREG_0.13.md` Part 2; see that file's addendum 3 for why.

## 1. Question answered for the user

How many groups must be labelled so that the audit's **primary verdict** is PASS with probability ≥ 0.80? PASS means the lower 95% bound of the held-out gain exceeds 0, for a population increment δ.

## 2. Model (unchanged from PREREG_0.13 Part 2)

| Element | Specification |
|---|---|
| Groups | each shares a latent L ~ N(0, 1) |
| Label | Y ~ Bernoulli(σ(a + b·L)), with a and b solved for prevalence π and population AUROC A₀ of L |
| Detector | S = ε + θ·Y, ε ~ N(0, 1) |
| θ | solved so the population increment (200,000 items, linear family, one seed) equals δ |

## 3. What changes

**Detection criterion.** Each simulated dataset counts as detected when the production audit returns verdict **PASS**:
- clustered DeLong interval, 2 seeds, three families;
- `null="residual_gbm"`, `n_perm=1`, no bootstrap;
- `baseline_increment.planner.detected(r, "pass")`.

The permutation count does not affect the gain or its interval (`PREREG_R1.md` §4).

**Engine.** Commit `feb21f5`. The only code change is the `criterion` argument (`planner._one`, `planner.detected`, `simulations/planner_suite.py --criterion pass`).

## 4. Validation (scenarios, sizes and arms unchanged)

**Scenarios:** S1 (π 0.3, A₀ 0.70, m 2, δ 0.03), S2 (0.1, 0.80, 1, 0.05) and S3 (0.5, 0.65, 4, 0.02).

**Recommended N.** The smallest grid N with point power ≥ 0.80. The grid is the planner default: 8 geometric values from the smallest N with ≥ 20 positives up to 20,000 items, with R = 200 and seed 1.

**Cells.** For each scenario, three sizes (N_rec/2, N_rec, 2·N_rec) times three arms:

| Arm | Runs | Seeds |
|---|---|---|
| Predicted | R = 200 | seed 2 |
| Fresh, same generator | R = 400 | seed 1000 + salt 13 |
| Independent, misspecified generator | R = 400 | the `PREREG_0.13` alternative generator: task with 3 levels plus numeric difficulty, matched π, A₀ and δ |

**Runner.** `python simulations/planner_suite.py --criterion pass --null residual_gbm --seed-salt 13 --out simulations/results/planner_plan2.json`.

## 5. Acceptance (fixed now)

**P1, corrected.** For each cell, |predicted − fresh| ≤ 1.96 · √( p̄(1 − p̄)(1/200 + 1/400) ), where p̄ is the pooled proportion. This is the 95% band for the difference of two independent Monte Carlo estimates. A correct planner passes each cell with probability 0.95, and at least 8 of 9 cells with probability about 0.92 (simulation in `PREREG_0.13.md` addendum 3). **Pass:** at least 8 of 9 cells.

**P2 (unchanged).** |predicted − independent| ≤ 0.10 in at least 8 of 9 cells.

**P3 (unchanged).** The independent power at the recommended N is ≥ 0.70 in all 3 scenarios.

**Outcome:**
- **Ship** `plan` as validated if P1, P2 and P3 all pass. CLAIM-015 then becomes *established*, restated for the PASS verdict.
- **Otherwise** `plan` stays experimental and the result is documented. No second attempt under this registration.

## 6. No tuning

None of the following changes after the first PLAN2 result:
- the criterion, model, scenarios, grid, R or seeds;
- the bands and thresholds.

Every deviation is recorded.
