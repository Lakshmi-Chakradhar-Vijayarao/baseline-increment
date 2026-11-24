# Pre-registration for 0.8.0: detector release comparison (`compare`)

Written and committed before any implementation run or calibration run.

## Scope
**What it is for:** comparing two versions of a detector, judge or guardrail score, old (v1) and new (v2), scored on the same items with the same labels.

**What it does not cover:** model-vs-model accuracy comparisons with paired tests, clustered errors and power. `evalconfidence` already does that; the README points there instead of reimplementing it.

## Three questions, reported separately and never merged

**T1. Improvement: does v2 add information beyond v1 and the baselines?**
- **Method:** `audit(score=v2, baselines = baselines + [v1])`, the existing engine with its conditional-randomisation null.

**T2. Regression: does v1 keep information that v2 lacks?**
- **Method:** `audit(score=v1, baselines = baselines + [v2])`.
- **As a gate:** `--fail-on-regression` exits 1 when T2 rejects.

**T3. Standalone: is AUROC(v2) − AUROC(v1) different from 0?**
- **Method:** a paired group bootstrap (2,000 draws, resampling groups) of the AUROC difference on the same resampled items. The decision is that the 95% percentile interval excludes 0.
- **Why it is separate:** T1 and T3 can disagree. A v2 that complements v1 adds information without being better on its own.

**Rules fixed now:**
- **Same transform for both scores.** v1 and v2 get exactly the same transform: rank, ranked within the same `rank_within` column if one is given. Each score enters as the detector or as a baseline column in that transformed form. One score transformed and the other raw is never allowed.
- **Multiplicity:** T1 and T2 decisions use Holm-adjusted p-values over the family {T1, T2} at 0.05. T3 is a separate question with its own interval.
- **Wording:** a non-rejection is reported as "not detected (n items, n groups)", never as "v1 adds nothing" or "v2 supersedes v1". The refit increment interval, demoted in 0.7.0, is not used in any verdict.

## Calibration run

**Data generator:** the suite's design.
- 300 groups of 2 items.
- Tasks A, B and C.
- A metadata column L.
- The label follows a logistic function of task and L.
- The baselines are task and L.

**Scores:** e1, e2 and e3 are independent N(0, 1) noise.

| Scenario | v1 | v2 | T1 (improvement) | T2 (regression) | T3 (standalone) |
|---|---|---|---|---|---|
| `reseed` | 0.8y + e1 | v1 + e2 | null | positive | — |
| `monotone` | 0.8y + e1 | exp(v1 / 2) | null | null | null (identical ranks) |
| `interaction` | 0.8y + e1 | v1 × (2 if task A, else 0.5) | adversarial null (needs gbm) | — | — |
| `coarsened` | 0.8y + e1 | v1 binned at its terciles | null | positive | — |
| `strict_upgrade` | 0.8y + e1 + e2 | 0.8y + e1 | positive | null | positive |
| `complementary` | 0.8y + e1 | 0.8y + e3 | positive | positive | null |

**Settings:** the `--fast` preset (2 seeds, at most 200 permutations, stopping at 10, no refit bootstrap), all three families, 100 replicates per scenario.

## Acceptance

**Ship-blocking. If any of these fails, `compare` does not ship:**
- T1 flag rate ≤ 0.10 in `reseed`, `monotone`, `interaction` and `coarsened`;
- T2 flag rate ≤ 0.10 in `strict_upgrade` and `monotone`;
- T3 says "different" (in either direction) in ≤ 0.10 of `complementary` and `monotone`.

**Power, reported:**
- **Targets:** T1 ≥ 0.80 in `strict_upgrade` and `complementary`; T2 ≥ 0.80 in `reseed`, `coarsened` and `complementary`; T3 ≥ 0.80 in `strict_upgrade`.
- **If one is missed:** `compare` still ships, and the README states the measured power next to that test.

**T3 interval coverage:** the interval must contain the replicate mean of the difference in 0.85–0.99 of runs in `complementary` and `strict_upgrade`. If that fails, T3 is shown as descriptive only, with no decision.

**No second attempt:** no scenario, threshold or setting changes after the first run under this registration.
