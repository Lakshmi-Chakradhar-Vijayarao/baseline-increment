# Pre-registration for 0.7.0: fast preset and half-sample interval

Written and committed before any run of the two checks below. The results go into `simulations/` and `CALIBRATION.md`, with any change to this file noted there.

## 1. Fast preset (`--fast`)

**Why.** The README sells the tool as a CI gate, but the defaults take tens of minutes on 16,000 rows. Almost all of that time is the refit bootstrap. The decision (`adds_information`) uses only the permutation p-value and the sign of the increment.

**Settings:**
- 2 CV seeds;
- no refit bootstrap (`n_boot = 0`);
- up to 200 permutations, with sequential stopping (Besag & Clifford 1991): stop as soon as 10 null increments reach the observed one, and report p = 10 / (number drawn). Otherwise p = (1 + count) / (1 + 200).

**Why the stopping rule keeps the test valid.** The Besag–Clifford p-value is valid for any stopping count. With 10 exceedances and at most 200 draws, an early stop always gives p ≥ 0.05. So it can never produce a flag that the full 200-draw run would not.

**Hypothesis.** The fast preset keeps the false-flag rates within the existing limits, keeps power close to the defaults, and is much faster.

**Why existing data can't answer this.** The calibration suite ran with 100 permutations and no stopping rule. It never ran the preset's own permutation settings.

**Run.** `simulations/suite.py` with all 18 scenarios, 100 replicates each, 2 seeds, 200 permutations and stopping at 10, using the suite's existing data generators and seeds.

**Ship the preset if all of these hold:**
- every null and adversarial-null scenario has a flag rate ≤ 0.10;
- `moderate_signal`, `strong_signal` and `signal_with_difficulty` each have ≥ 0.80;
- `hard_only_signal` has ≥ 0.50;
- `weak_signal` has ≥ 0.72, which is the default suite's 0.82 minus 0.10.

**Drop the preset if any of these fails.** It does not ship.

## 2. Half-sample interval for the increment

**Problem.** The refit-bootstrap interval covers the mean estimate 1.000 of the time under the `noise` and `misspecified_baseline` nulls. The acceptance band is 0.85–0.99, so the interval is too wide there.

**Candidate.**
- **Draw:** half the groups, without replacement, so no group is duplicated. Rerun the whole estimator on that half (all seeds, all families, the headline choice). Repeat B times.
- **Deviations:** let d_b = θ*_b − mean(θ*).
- **Interval:** [θ̂ + q_0.025(d), θ̂ + q_0.975(d)].
- **Why half:** for a mean, a half-sample drawn without replacement has variance σ²/n, which matches the full-sample estimator. Centring removes the bias that comes from fitting on smaller training sets.

**Hypothesis.** The half-sample interval falls inside the 0.85–0.99 band on all three coverage scenarios.

**Run.** The suite's existing coverage run: `noise`, `moderate_signal` and `misspecified_baseline`, 40 replicates, B = 100, 2 seeds.

**Adopt it if** coverage of the mean point estimate is within 0.85–0.99 on all three scenarios. It then becomes the reported interval (`interval="half"`), and the refit bootstrap remains available as `interval="bootstrap"`.

**Demote the interval if it fails any scenario.** It stays as the refit bootstrap, the report labels it "conservative, not used for the decision", and the README stops placing it beside the headline increment as a confidence interval. No second candidate gets tried under this registration.
