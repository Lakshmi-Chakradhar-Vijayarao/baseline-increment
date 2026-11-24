# Pre-registration for 0.13: omitted-baseline sensitivity analysis and sample-size planner

Written and committed before any code for either feature exists and before any run. Acceptance runs start only after every PREREG_0.12 run has finished, and use the engine and null adopted there. Fresh seeds: `--seed-salt 13`.

## Prior work, checked before writing

- **Omitted-variable sensitivity in regression:**
  - Cinelli & Hazlett (2020), the robustness value and benchmarking against observed covariates (`sensemakr`);
  - VanderWeele & Ding (2017), the E-value;
  - Rosenbaum-style bounds.
- **Testing whether a feature adds information beyond covariates:**
  - Watson & Wright (2021), conditional predictive impact;
  - leave-one-covariate-out tests;
  - the conditional randomisation test, which this package uses.
- **A recent preprint on embedding bias in conditional independence tests** (arXiv 2610.11584) uses partial-R² sensitivity parameters for a CI test.

No source found defines a sensitivity analysis for an **out-of-fold AUROC increment over baselines**. What follows adapts Cinelli and Hazlett's construction to that quantity. We call it a *sensitivity analysis*, not a bound, because no worst-case guarantee is proved for AUROC with flexible models. Part 1's validation checks the property that matters in practice, empirically.

## Part 1: omitted-baseline sensitivity (`--sensitivity`)

**Question answered for the user.** How strong would a cheap variable missing from the baselines have to be to erase the detector's increment? And how does that compare with the baselines that were declared?

**Construction.** Everything is out of fold, under the audit's own grouped folds, seed 0, headline family.
- **The two residuals:**
  - r_S = rank(S) − Ŝ(B), where Ŝ is the adopted null's score model;
  - r_Y = Y − p̂(B), where p̂ is the headline family's baselines-only prediction.

  Both are standardised to z_S and z_Y.
- **A hypothetical omitted baseline U(α, β):** a unit-variance combination a·z_S + b·z_Y + c·ε, where ε is fresh N(0, 1) noise. The weights a, b, c are solved so that corr(U, z_S) = α and corr(U, z_Y) = β, given the observed corr(z_S, z_Y). Combinations that are infeasible are marked infeasible.
- **Remaining increment:** the headline family's increment of S over B + U, which is mean[AUROC(B + U + S) − AUROC(B + U)] over the audit's CV seeds and 3 draws of ε.
- **Robustness value (RV):** the smallest α = β at which the remaining increment is ≤ 0, found by bisection on [0, 0.99] to a tolerance of 0.01. If the increment is still > 0 at 0.99, the result is reported as "RV > 0.99".
- **Contour:** remaining increment on a 5 × 5 grid with α and β in {0.1, 0.3, 0.5, 0.7, 0.9}, reported only.
- **Benchmarks:** for each declared baseline b_j, its own strengths if it had been omitted. That is α_j = corr(b_j residualised on B₋ⱼ, r_S given B₋ⱼ), and β_j likewise with r_Y given B₋ⱼ. For a categorical b_j, the out-of-fold prediction of b_j's indicator columns from B₋ⱼ is used, combined through its first canonical direction.
- **Report sentence** (fixed): "The increment would fall to zero only if an omitted baseline were correlated at least RV with both the detector's and the label's residuals; the strongest declared baseline, b*, has (α*, β*)."
- **Scope:** only for audits with `adds_information = true`. Descriptive, never part of the verdict.

**Validation, by simulation.** Each data-generating process (DGP) has a real omitted variable Z with known strengths. The tool runs without Z, then Z is added as a baseline and the actual increment is measured.
- **Grid:**
  - 4 DGP families: linear-Gaussian, nonlinear (Z enters through tanh), categorical Z with 4 levels, and an interaction (Z × task);
  - 3 regimes for how much of S's signal runs through Z: none, half, all;
  - 2 sample sizes: 600 and 2,000 items in groups of 2.

  That is 24 cells, 50 replicates each.
- **Prediction:** the erasure is predicted if Z's measured (α_Z, β_Z), computed with Z treated as omitted, lies on or beyond the zero contour. Concretely, the prediction uses the remaining increment interpolated at (α_Z, β_Z).
- **Outcome:** Z erases the increment if, with Z added, the increment is ≤ 0.

**Acceptance (the property that matters):**
- **V1, no false comfort.** Among replicates where erasure is not predicted, Z actually erases the increment in ≤ 5%, pooled. This must also hold within each DGP family at ≤ 10%.
- **V2, informativeness, reported.** Among replicates where erasure is predicted, the share where it actually happens. A low value means the analysis is conservative: it is reported, but it doesn't block.
- **V3, calibration of the RV, reported.** The correlation between the RV and the actual remaining increment across cells.

**Decision.** If V1 passes, `--sensitivity` ships, labelled "sensitivity analysis (validated against simulated omitted baselines; no worst-case guarantee)". If V1 fails, it doesn't ship and the failure is documented.

## Part 2: sample-size planner (`baseline-increment plan`)

**Question answered for the user.** How many items or groups do I need to label so that the audit detects a population increment of δ with power ≥ 0.80?

**Inputs:**
- the label prevalence π;
- the baselines-only AUROC A₀;
- the group size m (1 if ungrouped);
- the target population increment δ;
- the target power (default 0.80);
- optionally, `--pilot data.csv` with the usual columns. π, A₀ (headline family, out of fold) and the mean group size m are then estimated from the pilot. Any value given on the command line overrides the pilot's.

**Model (parametric, fixed).**
- **Groups:** each group shares a latent L ~ N(0, 1).
- **Label and baseline:** Y ~ Bernoulli(σ(a + b·L)). The observed baseline is L itself. a and b are solved so that the prevalence equals π and the population AUROC of L equals A₀.
- **Detector:** S = ε + θ·Y, with ε ~ N(0, 1).
- **θ:** solved by bisection so that the **population** increment equals δ. Population means one dataset of 200,000 items scored with the audit's estimator, linear family, a single seed.

**Power.** For N in a grid of group counts (default: 8 values spaced geometrically, from the smallest N with at least 20 positives up to 20,000 items), R = 200 datasets are generated, each run with the `--fast` audit (all three families). The planner reports:
- power(N), with Wilson 95% intervals;
- the smallest grid N whose point power is ≥ the target;
- its Monte Carlo uncertainty.

**Validation.**
- **Scenarios:** 3:
  - π = 0.3, A₀ = 0.70, m = 2, δ = 0.03;
  - π = 0.1, A₀ = 0.80, m = 1, δ = 0.05;
  - π = 0.5, A₀ = 0.65, m = 4, δ = 0.02.
- **N:** 3 values each, the planner's recommended N and ×0.5 and ×2 of it.
- **Independent check, under misspecification:** a different generator with the same π, A₀, m and δ. Its baselines are task (3 levels) plus a numeric difficulty, a and b are found by search to match π and A₀, and θ is calibrated the same way. R = 400, fresh seeds.

**Acceptance:**
- **P1, at the planner's own generator:** predicted power lies inside the fresh simulation's Wilson 95% interval in at least 8 of 9 cells.
- **P2, under misspecification:** |predicted − independent power| ≤ 0.10 in at least 8 of 9 cells.
- **P3, at the recommended N:** the independent power is ≥ 0.70 in all 3 scenarios.

**Decision.** Ship if P1, P2 and P3 pass. Otherwise it doesn't ship, and the result is documented.

## No second attempt

No construction, grid, threshold or setting changes after the first acceptance run under this registration.

## Addendum 1 (25 Nov, before any acceptance run): Part 1 construction replaced

**Flaw found in development** (dev seeds, `simulations/dev/`, not acceptance). The registered construction is the mixture U = a·z_S + b·z_Y + c·ε. A model given U and S together can recover b·z_Y + c·ε = U − a·z_S, a cleaner label signal. So S keeps "adding information" through U however strong U is.

On a test where the true omitted variable Z erases the increment (with Z added: +0.000, adds = false), the mixture never brought the increment to zero, even at the largest feasible strength (0.82). The registered construction cannot answer the question, and it is withdrawn.

**Replacement: the partial-correlation condition.** Let r̂ = corr(z_S, z_Y), the out-of-fold residual correlation of detector and label given the baselines. For an omitted variable U, let α and β be its multiple correlations with the two residuals, as in the benchmarks above.

In the linear-Gaussian approximation, the partial covariance of S and Y given (B, U) is r − α·β. So U makes the detector redundant when α·β reaches r. The analysis reports:
- **The erasure index of a variable:** φ = α·β / r̂, where φ ≥ 1 means "would erase the increment".
- **The robustness value:** RV = √r̂, the equal-strength point (α = β = RV).
- **Each declared baseline's own φ_j,** computed as if it had been left out, and the strongest of them.
- **The fixed sentence:** "In the linear approximation, an omitted baseline erases the increment when the product of its correlations with the detector's and the label's residuals reaches r̂ (both at least RV if equal); the strongest declared baseline, b*, reaches φ* of that."

**Development evidence.** Development used 9 runs (3 seeds × 3 regimes, 1,200 items, linear family):

| How much of S's signal runs through Z | φ_Z | Increment with Z added | adds |
|---|---|---|---|
| all | 0.91–1.09 | +0.000 to −0.002 | false |
| half | 0.44–0.46 | +0.037 to +0.041 | true |
| none | 0.13–0.19 | +0.064 to +0.079 | true |

**Validation (replaces V1–V3; same 24 cells, 50 replicates, seed salt 13).** Only replicates where the audit without Z gives `adds_information = true` count.
- **V1, no false comfort.** Among replicates with φ_Z ≤ 0.5 (Z clearly weaker than needed), the increment with Z added stays > 0 in ≥ 95% pooled, and in ≥ 90% within each DGP family.
- **V2, reported.** Among replicates with φ_Z ≥ 1, the share where `adds_information` is false once Z is added.
- **V3, reported.** The Spearman correlation between φ_Z and the relative drop, 1 − increment(with Z) / increment(without Z).

The decision rule is unchanged: ship if V1 passes. Nothing else in this registration changes.
