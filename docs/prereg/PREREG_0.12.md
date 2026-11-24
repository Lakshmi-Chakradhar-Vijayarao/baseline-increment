# Pre-registration for 0.12.0: null score model, effect-size interval, LLM-judge validation, speed

Written and committed before any acceptance run below. Engine code for both candidates is frozen in the same commit. Every acceptance run uses **fresh seeds** (`--seed-salt 12`), which no development or earlier run used. Results go to `simulations/results/*_v0.12.0*.json`, `validation/results_v0.12.0.json` and `CALIBRATION.md`.

## What development already used

Development is disclosed so the acceptance runs can be judged against it. It was done on separate development seeds (offset 900,000) in `simulations/dev/`:
- **`diag_null_cause.py`** (dev seeds, 200 runs per arm on `interaction_meta`, 100 on `large_copy`):

  | Arm | `interaction_meta` | `large_copy` (10,000 rows) |
  |---|---|---|
  | current null | 0.085 | 0.04 |
  | oracle score model: the true E[S \| baselines] | 0.03 | 0.04 |
  | oracle outcome baseline: the true η added as a column | 0.055 | 0.05 |

  - On the interaction case, both oracles reduce the excess, and the null's score model is the larger lever.
  - On `large_copy` nothing is in excess on dev seeds. The 0.09 in the 0.11.1 stress run is consistent with chance, since its Wilson interval was [0.048, 0.162].
- **`shat_candidates.py`:** label-free regressors for E[S | baselines], compared by mean squared error against the oracle.
  - A deeper gbm never helped, so it was dropped.
  - A ridge regression on an interaction basis (spline × one-hot and degree-2 products) came 4–8× closer to the oracle than the current gbm on `interaction_meta`.
- **Super-learner arm on dev seeds** (200 runs each):

  | Scenario | Super learner | Current null |
  |---|---|---|
  | `interaction_meta` | 0.05 | 0.085 |
  | `weak_signal` | 0.795 | 0.81 |
  | `misspecified_baseline` | 0.06 | 0.06 |

No other scenario, setting or threshold was looked at under the candidate.

## Part A: the null's score model (`null="residual_sl"`)

**Problem.**
- **The failure:** the 0.11.1 stress test found the interaction-only null borderline, at 0.095 [0.070, 0.128] over 400 runs.
- **Cause:** the conditional-randomisation null replaces the detector with S* = Ŝ + permuted residuals, where Ŝ = E[S | baselines] comes from one gradient-boosting regressor. When S is an exact function of the baselines, Ŝ's approximation error stays in the residual and is correlated with the label. The permuted copies lose that correlation and the real detector keeps it, so the test flags too often.

**Candidate (frozen in `baseline_increment/core.py`, `score_given_baselines(model="super_learner")`).**
- **Method:** a discrete super learner over two regressors of S on the baselines:
  1. the 0.11 gbm, unchanged;
  2. RidgeCV (α in 10^-3…10^3) on `interaction_basis`: one-hot columns, standardised numerics, their cubic B-spline basis, spline × one-hot products, and degree-2 numeric products (numeric × numeric and numeric × one-hot).
- **Selection:** both are fit out of fold under the same grouped folds. The one with the lower out-of-fold mean squared error of S is used.
- **No labels:** the choice never looks at the label, so it cannot be tuned to the outcome.

**Hypothesis.** The candidate removes the excess on interaction-type nulls without new failures elsewhere, and without losing power.

**Why existing data can't answer this.** Development used one interaction scenario on development seeds. Acceptance needs held-out scenarios, including a three-way interaction that the ridge basis cannot represent, as well as fresh seeds and every existing gate.

**Acceptance runs** (all with `--null residual_sl --seed-salt 12`, at the `--fast` settings: 2 seeds, at most 200 permutations, stopping at 10, all three families):

| Run | Rule |
|---|---|
| A1 `suite.py`, 18 scenarios, 100 runs | the 0.7.0 fast-preset rules, unchanged: every null and adversarial null ≤ 0.10; `moderate_signal`, `strong_signal` and `signal_with_difficulty` ≥ 0.80; `hard_only_signal` ≥ 0.50; `weak_signal` ≥ 0.72 |
| A2 `stress_suite.py`, S1–S5 | Part B rules of `PREREG_VALIDATION.md`, unchanged. **In addition, `interaction_meta` (400 runs) must PASS:** Wilson upper bound ≤ 0.10 |
| A3 held-out nulls (in `stress_suite.py`), 400 runs each | Wilson upper ≤ 0.10 is a pass; a point rate ≤ 0.10 is borderline; a rate above 0.10 is FAIL |
| A4 `compare_suite.py`, 100 runs | the 0.8.0 ship-blocking rules, unchanged |
| A5 `ordinal_suite.py`, 100 runs | the 0.11.0 rules, unchanged |
| A6 real data, `run_validation.py --null residual_sl` | A1–A3 agreement criteria and N1/N2 ≤ 0.10, as in `PREREG_VALIDATION.md` (same real-data seeds, for a like-for-like comparison) |

The A3 held-out nulls each use 300 groups × 2, with η = 1.2 f and S = f + 0.3 noise:

| Scenario | f | Baselines |
|---|---|---|
| `product_meta` | L1·L2 | L1, L2 |
| `threshold_meta` | 2·[L1 > 0.5 and task = A] − 0.7 | task, L1, L2 |
| `three_way_meta` | L1·L2·(+1 if task = A, else −1) | task, L1, L2 |
| `large_copy_400` | the stress `large_copy` (10,000 rows) | as in the stress suite |

**Decision.**
- **Adopt** if A1, A2, A4, A5 and A6 all pass and no A3 scenario is FAIL. `residual_sl` then becomes the default null (`DEFAULT_NULL`). The golden fixture is regenerated once, and the old engine stays available as `null="residual_gbm"`.
- **Otherwise** `residual_gbm` stays the default, and the result is documented. A borderline A3 scenario is adopted with a documented caveat.
- **Power:** reported in all runs. The decision uses only the rules above.

## Part B: an effect-size interval (`interval="delong"`)

**Problem.** The refit bootstrap was too wide (coverage 1.00), and the half-sample interval failed its coverage check (0.7.0). So reports carry no usable interval for the increment.

**Candidate (frozen in `core.py`):**
- **Predictions:** the headline family's out-of-fold predictions, baselines alone (pb) and baselines + detector (pf), each averaged over the CV seeds.
- **Interval:** increment ± 1.96 · SE. The SE is the clustered DeLong standard error of AUROC(pf) − AUROC(pb) (Obuchowski 1997): per-group sums of centred placement-value differences, scaled by I/(I − 1).
- **Cost and use:** no refits. Descriptive only; it never enters `adds_information`.

**Risk stated in advance.** The interval treats the fitted models as fixed, so it ignores fitting variance and may under-cover, particularly under the null.

**Run.** `suite.py --coverage --interval delong --seed-salt 12`, 200 runs per scenario:
- `noise`
- `moderate_signal`
- `misspecified_baseline`
- `weak_signal`
- `strong_signal`
- `signal_with_difficulty`

The run uses the fast settings and the adopted null (Part A's decision is made first).

**Truth.** As in every earlier coverage run, the truth is the mean point estimate over replicates.

**Adopt if** coverage is within 0.85–0.99 in all six scenarios. It then becomes the interval reported by `--fast` and by default in reports, labelled descriptive. **Otherwise** it is documented as failed, and reports keep showing no interval. No second candidate gets tried under this registration.

## Part C: LLM-judge detectors on real data

**Question.** Does the tool keep false alarms near 5% on the score distributions of real LLM judges? Judge scores are bimodal, with heavy mass near 0 and 1.

**Data** (`validation/build_judge_table.py`, built before this registration, with no run on it):
- **Items:** WQA PopQA (7 models) and TriviaQA (8 models), 30,000 answers.
- **Label:** the deterministic rule label `wrong`, which no judge produced. These two datasets are the ones where the rule agrees with humans at κw ≥ 0.80.
- **Detectors:** two judges' risk, 1 − P(Yes): Falcon3-10B and GLM-4-9B.
- **Baselines:** `d1` (the other models' error rate on the question) and log answer words.

**Null controls**, 100 runs each, on one fixed model, Qwen2.5-7B-Instruct (in both datasets, fixed now). The detector is the Falcon judge:
- **N1:** the judge's scores permuted within `d1`-deciles.
- **N2:** rank(d1) + N(0, 0.3).
- **P1:** N1 + 0.6·SD·y, power, reported.
- **Acceptance:** N1 and N2 ≤ 0.10 on both datasets.

**Reported, no threshold:** the per-model increment of each judge over `d1` + length, with its verdict. The label is a rule, not a human, so this is descriptive.

## Part D: speed (exact output)

**Changes:** vectorised group permutation, group bootstrap draws and one-hot encoding (`GroupIndex`). Every draw consumes the generator exactly as before.

**Acceptance:**
- `tests/test_golden.py` passes on the pre-0.12 fixture with `null="residual_gbm"`;
- a test checks the vectorised permutation against the original loop on random inputs, including stratified ones.

**Reported:** wall time of `--fast` at 1,000, 10,000 and 100,000 rows on the development machine (8 cores).

## No second attempt

No candidate, scenario, threshold, seed salt or setting changes after the first acceptance run under this registration. Any failure is reported against the rule it breaks.
