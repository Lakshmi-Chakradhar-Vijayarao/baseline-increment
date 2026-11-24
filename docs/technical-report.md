# baseline-increment: technical report

*Method, calibration, validation and limits of an incremental-validity audit for AI detectors and evaluators. This report is regenerated for each release from the recorded evidence. Every number cites the file it comes from.*

<!-- PENDING markers below are filled from recorded results before release; tests/test_claims.py fails if any remain at release time. -->

## 1. The question

A detector, judge or uncertainty score s is usually reported by its AUROC against a label y. Part of that AUROC can be reproduced from cheap information b that needs no detector: question or source difficulty, response length, task, or the generating model. The audit asks one question:

> Does s improve out-of-sample prediction of y beyond b, under a test whose false-alarm rate has itself been measured?

The answer is predictive, not causal. It is relative to the declared baselines b. It says nothing about whether y measures the intended construct (§8).

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

## 3. Test and decision rule

**The null.** The null hypothesis is that y is independent of s given b. It is tested by an approximate conditional randomisation:
1. Out of fold, ŝ = E[s | b] is estimated by a gradient-boosting regressor. The regressor never sees y.
2. The null detectors are s* = ŝ + π(s − ŝ), where π permutes the residuals across groups (within declared strata).
3. The observed Δ (seed 0) is compared with the Δ of each s*.

**Stopping.** Permutations stop early by the Besag–Clifford rule (stop after h = 10 exceedances; p = h/draws). Otherwise p = (1 + #exceedances)/(1 + B), with B ≤ 200.

**Decision.** `adds_information` is true exactly when p < 0.05 and Δ > 0. "Not detected" is never reported as "no information".

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
- **LLM judges.** PENDING: Part C of `PREREG_0.12.md` (Falcon3-10B and GLM-4-9B judges on 30,000 WQA answers with rule labels).

## 6. A documented failure mode

When s is an **exact function of b** that the null's score model cannot reproduce, the residual s − ŝ keeps the part of the function the model missed. That part is correlated with y, permuting it destroys the correlation, and false flags rise above the nominal rate.
- **0.11 default null.** An interaction-only detector (s = L × sign(task)) was flagged in 9.5% of 400 runs, interval [7.0%, 12.8%].
- **0.12 candidate (PREREG_0.12).** The candidate was a label-free super learner: the better of gradient boosting and a ridge regression on spline and interaction terms, chosen by out-of-fold error. It brought the interaction case to 4.75% [3.1%, 7.3%]. On held-out exact-function detectors it then failed:
  - a product of two baselines: 14.75% [11.6%, 18.6%];
  - a threshold rule: 11.25% [8.5%, 14.7%];
  - PENDING: the three-way interaction case and the large linear copy.

  Under its frozen rule it was **not adopted**. PENDING: the old null on the same held-out cases (`stress_v0.12.0_heldout_gbm.json`).
- **Advice to users.** When a detector might be a deterministic function of the baselines, for example a judge whose score is driven by task and length, add that function as a baseline column. Then read a positive result near p = 0.05 with care.

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

## 9. Under validation

| Feature | Pre-registration | Status |
|---|---|---|
| Effect-size interval (clustered DeLong) | `PREREG_0.12.md` Part B | PENDING |
| Omitted-baseline sensitivity analysis | `PREREG_0.13.md` Part 1 + addenda 1–2 | PENDING |
| Sample-size planner | `PREREG_0.13.md` Part 2 | PENDING |

## References

- Berrett, T. B., Wang, Y., Barber, R. F. & Samworth, R. J. (2020). The conditional permutation test for independence while controlling for confounders. *JRSS B* 82(1).
- Besag, J. & Clifford, P. (1991). Sequential Monte Carlo p-values. *Biometrika* 78(2).
- Candès, E., Fan, Y., Janson, L. & Lv, J. (2018). Panning for gold: model-X knockoffs. *JRSS B* 80(3).
- Chernozhukov, V., Cinelli, C., Newey, W., Sharma, A. & Syrgkanis, V. (2022). Long story short: omitted variable bias in causal machine learning.
- Cinelli, C. & Hazlett, C. (2020). Making sense of sensitivity: extending omitted variable bias. *JRSS B* 82(1).
- DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics* 44(3).
- Obuchowski, N. A. (1997). Nonparametric analysis of clustered ROC curve data. *Biometrics* 53(2).
- Watson, D. S. & Wright, M. N. (2021). Testing conditional independence in supervised learning algorithms. *Machine Learning* 110.
