# baseline-increment: technical report

*Method, calibration, validation and limits of an incremental-validity audit for AI detectors and evaluators. This report is regenerated for each release from the recorded evidence. Every number cites the file it comes from.*


## 1. The question

A detector, judge or uncertainty score s is usually reported by its AUROC against a label y. Part of that AUROC can be reproduced from cheap information b that needs no detector: question or source difficulty, response length, task, or the generating model. The audit asks one question:

> Does s improve out-of-sample prediction of y beyond b, and by how much?

From 0.13 this is answered by the gain and a validated 95% interval (§3). The stronger question, whether s carries information about y that b lacks (Y not independent of S given B), has a separate, experimental test. §6 explains why it cannot carry a verdict. The answer is predictive, not causal. It is relative to the declared baselines b and the declared model families. It says nothing about whether y measures the intended construct (§8).

## 2. Estimand and estimator

**Increment.** The increment is

Δ = AUROC(f(b, s)) − AUROC(f(b)),

where both models are fitted and scored out of fold under grouped K-fold cross-validation (K = 5). Items that share a group, such as a question or a source document, never straddle a fold. The detector enters as its rank, optionally within a column such as the model, so monotone rescaling cannot change the result.

**Baseline families.** f is fitted in three families, chosen before any data are seen:
- L2 logistic regression;
- cubic B-splines on the numeric baselines plus one-hot categories;
- shallow gradient boosting.

The *headline* family is the one with the highest baselines-only AUROC. It is chosen without looking at the detector. Every family's increment is reported.

**Seeds.** Δ is averaged over CV seeds: 2 under `--fast`, 5 by default.

## 3. Interval and primary verdict (from 0.13)

- **The interval:** Δ ± 1.96·SE. SE is the clustered DeLong standard error of AUROC(p̂(b, s)) − AUROC(p̂(b)), computed on seed-averaged out-of-fold predictions of the headline family: per-group sums of centred placement-value differences (Obuchowski 1997).
- **Validated coverage:** it covered the mean estimate in 0.905–0.965 of 200 runs in each of six pre-registered scenarios, with and without signal (`PREREG_0.12.md` Part B; `coverage_v0.12.0_delong.json`). It was not validated for scores that are exact functions of b.
- **The verdict:** **PASS** if the lower bound exceeds a declared minimum gain δ (default 0), **FAIL** if the upper bound is below δ, **INCONCLUSIVE** otherwise. The CI gate uses this verdict.
- **What it reads as:** a model-class claim. Under the declared baseline families and at this sample size, adding s improves held-out prediction.

## 3b. Secondary test of conditional independence (experimental)

**The null.** The null hypothesis is that y is independent of s given b. It is tested by an approximate conditional randomisation:
1. Out of fold, ŝ = E[s | b] is estimated by a gradient-boosting regressor. The regressor never sees y.
2. The null detectors are s* = ŝ + π(s − ŝ), where π permutes the residuals across groups (within declared strata).
3. The observed Δ (seed 0) is compared with the Δ of each s*.

**Stopping.** Permutations stop early by the Besag–Clifford rule (stop after h = 10 exceedances; p = h/draws). Otherwise p = (1 + #exceedances)/(1 + B), with B ≤ 200.

**Decision (secondary).** `adds_information` is true exactly when p < 0.05 and Δ > 0. From 0.13 it is reported but drives no verdict and no gate (§6).

**Relatives.** This is the conditional-predictive-impact idea of Watson & Wright (2021), tested with a residual-permutation form of the conditional randomisation test (Candès et al. 2018; Berrett et al. 2020). The test is valid to the extent that ŝ approximates E[s | b] (§6).

## 4. Calibration in simulation

Every rule was checked against acceptance criteria written and committed before the run (`docs/prereg/`). Results are in `simulations/results/` and are summarised in `CALIBRATION.md`.

| Procedure | Highest false-flag rate on no-signal scenarios (nominal 0.05) | Detection of planted signal | Source |
|---|---|---|---|
| audit, `--fast` | 0.10 across 13 scenarios, adversarial ones included | weak 0.81; moderate 1.00; strong 1.00 | `results_v0.7.0_fast.json` |
| stress test: 10 baselines, 10,000 rows, groups of 1–40, clustered noise | 0.10 at most | 1.00 | `stress_v0.11.1.json`, `stress_v0.12.0_sl.json` |
| `compare` | 0.06 (8 registered nulls) | 0.82–1.00 | `compare_v0.8.0.json` |
| ordinal outcome | 0.08 (5 scenarios) | 1.00 | `ordinal_v0.11.0.json` |
| raw-score calibration intervals | coverage 0.955 / 0.945 | — | `calcov_v0.11.0.json` |

## 5. Validation on real data

`docs/validation.md`, pre-registered in `docs/prereg/PREREG_VALIDATION.md`.
- **Agreement.** On seven detectors with independent prior analyses (RAGTruth and TofuEval: LettuceDetect, HHEM; TriviaQA, 8 models: P(True), log-probability, semantic entropy), every verdict was reproduced, and every increment was within 0.006 AUROC.
- **False alarms.** With each detector replaced by a no-information score on the real data structures: 32/600 = 0.053 [0.038, 0.074]. Planted signal was detected in 100% of runs.
- **LLM judges, rule labels (`PREREG_0.12.md` Part C).** Falcon3-10B and GLM-4-9B judges on 30,000 WQA answers added +0.065 (PopQA) and +0.12 (TriviaQA) beyond difficulty and length. The no-information controls gave 0.04–0.06.
- **LLM judges, human labels (`PREREG_JUDGES.md`; `docs/judges.md`).** On SummEval, FaithBench and TREC LLMJudge (33 judges), the cheap baselines alone nearly matched or beat raw judge agreement: 0.69 vs the best judge's 0.72 on TREC, 0.78 vs 0.49–0.60 on FaithBench, and 0.764 vs 0.752 on SummEval coherence.

## 6. Where the conditional-information test fails, and why the gain is the verdict

**The mechanism.** The null replaces s by ŝ(b) + permuted residuals. When s depends on b in a way the score model ŝ cannot reproduce, the residual keeps that part of the dependence. It is correlated with y, permuting it destroys the correlation, and false flags rise above the nominal rate. Shah & Peters (2020) show that no test of Y ⊥ S | B controls its size over all null distributions while keeping power, so this is a limit of the problem, not only of this null.

**Held-out cases (`PREREG_0.12.md` A3, 400 runs each, 600 items).** The old and new nulls fail alike:

| Score s = f(b) + noise | Old null (default) | Super-learner null | Mean Δ |
|---|---|---|---|
| L1·L2 | 0.1475 | 0.1475 | −0.004 |
| threshold rule | 0.1025 | 0.1125 | +0.002 |
| L1·L2·sign(task) | 1.00 | 1.00 | +0.118 |
| large linear copy (10,000 rows) | 0.0575 | 0.0625 | 0.000 |

The super-learner null was **not adopted**.

**Diagnostic map (`PREREG_DIAG.md`; 200 runs per cell, default null, 600 / 3,000 items):**
- **Handled:** linear (0.07 / 0.035), threshold × L (0.09 / 0.04) and noise (0.05 / 0.04).
- **Borderline (0.07–0.12):** sum, product, threshold, exponential, sin + square.
- **Not handled:**
  - **a baseline whose slope differs by task:** 0.575 / 0.325 (super-learner null 0.265 / 0.28);
  - **XOR:** 0.745 / 0.37;
  - **three-way interaction:** 1.00 / 1.00.

**On real data (`PREREG_JUDGES.md`, `PREREG_TREC.md`).** In TREC (25 queries × passage length), a judge permuted within query × length deciles was flagged 33% of the time. Stratifying the null within query did not fix it (0.29). Only the control's own query × length stratification did (0.07), and users cannot know that stratification.

**Model-class gains.** When s encodes a function of b that the baseline families do not learn at the sample size, Δ is a real *predictive* gain that shrinks as data grow: three-way, +0.115 at 600 items and +0.053 at 3,000. The gain and its interval report this honestly. A conditional-information reading of it would be wrong.

**Consequences in 0.13:**
- the verdict and the CI gate use the gain and its validated interval (§3);
- `adds_information` is reported as secondary and experimental;
- users are told to add plausible functions of their baselines, such as task × length, as baseline columns, and to read the ladder (`--ladder`).

## 7. Assurance layer

This layer does not change any statistic.
- **Declared policy** (`--policy`). The analysis is fixed by a hashed file. Rules that are not calibrated (minimum-gain gates, other α, other metrics) are refused by name. `declare` records the policy in the ledger before any run.
- **Evidence package** (`--evidence-dir`). It holds the report, results, the specification, provenance, the environment, the command, a ledger snapshot and `evidence-record.json` (published schema), under a SHA-256 manifest.
- **Run ledger.** It is append-only and hash-chained. It counts runs and distinct analysis specifications on the same input. `verify-evidence --expect-head` detects edits, removals and a rebuilt ledger.
- **Validity envelope.** Each run gets a status: detected, not detected, outside the validated range, or not testable. The calibrated designs span 60 or more groups, 30 or more items in the rarer class, at most 10 baseline columns, groups of at most 40 items, and up to 10,000 rows.
- **Claim ledger** (`docs/CLAIMS.md`). Each public claim lists its evidence and an executable test.

## 8. What the method does not establish

- causal validity;
- the absence of omitted baselines (a sensitivity analysis is under validation, `PREREG_0.13.md`);
- generalisation to other tasks, models or periods;
- that the label measures the intended construct;
- calibration of the raw score (reported separately, descriptive only);
- usefulness after deployment.

## 9. Status of each component

| Feature | Pre-registration | Status |
|---|---|---|
| Effect-size interval (clustered DeLong) | `PREREG_0.12.md` Part B | **validated**: coverage 0.905–0.965; primary verdict from 0.13 |
| Omitted-baseline sensitivity analysis | `PREREG_0.13.md` Part 1 + addenda 1–2 | **validated**: no false comfort in 98.0% (each family ≥ 93.8%); weak spot: interaction-type omitted variables at 600 items |
| Sample-size planner | `PREREG_0.13.md` Part 2 | under validation: S3 passed (P1 and P2 in all 3 cells; power 0.88 at the recommended size); S1 and S2 are being re-run in stages after hitting the 11 h compute limit |
| Conditional-information test structure check | not yet registered | needed before `adds_information` can carry a verdict again |

## References

- Berrett, T. B., Wang, Y., Barber, R. F. & Samworth, R. J. (2020). The conditional permutation test for independence while controlling for confounders. *JRSS B* 82(1).
- Besag, J. & Clifford, P. (1991). Sequential Monte Carlo p-values. *Biometrika* 78(2).
- Candès, E., Fan, Y., Janson, L. & Lv, J. (2018). Panning for gold: model-X knockoffs. *JRSS B* 80(3).
- Chernozhukov, V., Cinelli, C., Newey, W., Sharma, A. & Syrgkanis, V. (2022). Long story short: omitted variable bias in causal machine learning.
- Shah, R. D. & Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *Annals of Statistics* 48(3).
- Cinelli, C. & Hazlett, C. (2020). Making sense of sensitivity: extending omitted variable bias. *JRSS B* 82(1).
- DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics* 44(3).
- Obuchowski, N. A. (1997). Nonparametric analysis of clustered ROC curve data. *Biometrics* 53(2).
- Watson, D. S. & Wright, M. N. (2021). Testing conditional independence in supervised learning algorithms. *Machine Learning* 110.
