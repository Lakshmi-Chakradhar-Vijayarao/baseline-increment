# Pre-registration: real-data validation and stress test (after 0.11.1)

Written and committed before any table was built or any run made. Engine: 0.11.1 (binary engine frozen by `tests/test_golden.py`).

## Part A: real-data validation

**Question:** on real detectors where an independent analysis already gave an answer, does baseline-increment reach the same answer? And on real data structures, does it keep false alarms near 5%?

**Independent prior results.** These come from the `decision-point` analysis code, which has its own estimator and its own pre-registrations (`h1_difficulty_residual.md`, `gate2_tofueval.md`). That code is separate from this package.

| Table | Rows | Groups | Detector | Baseline | Prior increment [95% CI] | Prior verdict |
|---|---|---|---|---|---|---|
| RAGTruth test (2,700 responses, 6 generators) | response | source | LettuceDetect | D_src = leave-one-generator-out hallucination share for the same source | +0.131 [0.111, 0.161] | survives |
| same | | | HHEM-2.1 | D_src | +0.091 [0.071, 0.115] | survives |
| TofuEval originals (1,773 summaries) | summary | source (dataset, doc, topic) | LettuceDetect | D_src (leave-one-out share for the same source) | +0.089 [0.066, 0.122] | survives |
| same | | | HHEM-2.1 | D_src | +0.095 [0.074, 0.126] | survives |
| WQA TriviaQA (8 models × 2,000 questions) | answer | — | P(True) risk | D1 = the other models' error rate on the question, one audit per model | +0.042 [0.038, 0.047] (mean over models) | survives |
| same | | | negative mean log-prob | D1 | +0.038 [0.034, 0.042] | survives |
| same | | | semantic entropy | D1 | +0.036 [0.033, 0.040] | survives |

**Settings:** the `--fast` preset (2 seeds, at most 200 permutations, stopping at 10), all three families. The baseline is exactly the prior's difficulty column. The detector is the prior's score; the tool ranks it. Groups are the source for RAGTruth and TofuEval. For WQA each per-model audit has one row per question, so it needs no group.

**Agreement criteria:**
- **A1, verdict:** `adds_information` is true for every RAGTruth and TofuEval detector. For each WQA detector it is true in at least 6 of the 8 per-model audits.
- **A2, magnitude:**
  - **Where:** the tool's headline increment must fall within the prior 95% CI widened by 0.02 on each side. For WQA this is the mean over the 8 audits.
  - **Why widened:** the estimators differ in the baseline basis, the score transform and the seeds.
- **A3, ordering:** the prior's headline finding is that grounding checks add much more than self-uncertainty signals do. So every RAGTruth and TofuEval increment must exceed every WQA mean increment.
- **Verdict on agreement:** if A1–A3 all hold, the validation AGREES. Any failure is reported against the criterion it breaks, and the tool is not tuned to fix it.

**Real-data null controls,** 100 replicates per table:
- **Labels, groups and baselines** are the real ones; only the detector is replaced.
- **N1, permuted detector:** the real detector's scores are permuted at random within strata. The strata are D-deciles, crossed with task for RAGTruth and with dataset for TofuEval. This removes any information beyond the stratum.
- **N2, metadata function:** the detector is a monotone function of the baselines plus noise, rank(D) + N(0, 0.3).
- **Acceptance:** each flag rate ≤ 0.10.
- **Reported only:** P1, planted signal. The N1 detector plus 0.6·SD·y, giving power on the real structure.
- **Which WQA model:** the WQA null controls use one fixed model, Mistral-7B. It is chosen now, before any run.

## Part B: stress test (simulated)

**Settings:** the `--fast` settings, all three families, 100 replicates unless stated.

| Scenario | Design | Role |
|---|---|---|
| S1a `many_baselines_noise` | 10 baselines: 3 informative (2 numeric, 1 categorical), 7 pure noise (4 numeric, 3 categorical); detector = noise | null |
| S1b `many_baselines_copy` | the same 10 baselines; detector = a nonlinear function of the 3 informative baselines + 0.3 noise | null |
| S1c `many_baselines_signal` | the same; detector = noise + 0.6·y | power |
| S2a `large_noise` | 5,000 groups × 2 items; detector = noise | null |
| S2b `large_copy` | 5,000 groups × 2; detector = η + 0.3 noise | null |
| S2c `large_weak` | 5,000 groups × 2; detector = noise + 0.15·y | power |
| S3a `uneven_noise` | 1,000 groups, sizes geometric with mean 4, capped at 40; detector = noise | null |
| S3b `uneven_copy` | the same sizes; detector = η + 0.3 noise | null |
| S4a `clustered_noise` | 600 groups × 3; the label has a group random effect (SD 1.5); the detector has its own independent group random effect (SD 1.5) + noise | null |
| S4b `clustered_shared_noise` | the same; the detector's group effect is also shared with a baseline column | null |
| S5 borderline re-runs, 400 replicates | `misspecified_baseline`, `interaction_meta` and `few_groups`, from the existing suite generator | null |

**Acceptance:**
- **S1–S4 nulls:** a flag rate ≤ 0.10 at 100 replicates.
- **S5:** pass if the Wilson 95% upper bound is ≤ 0.10. If only the point rate is ≤ 0.10, the result is BORDERLINE, documented. If the point rate is above 0.10, it is FAIL.
- **Power:** reported only, with runtime per replicate recorded for S2.
- **If a null fails,** the failure is documented in CALIBRATION.md and the README, and a fix needs a new registration.

## No second attempt
No table, baseline, threshold or setting changes after the first run under this registration.

## Addendum 1 (24 Nov, before any Part A run): the RAGTruth design matches the prior's estimand

**What was found:** while checking that the tables reproduce the priors, the prior's RAGTruth figure (+0.131 LettuceDetect, +0.091 HHEM) turned out not to be one pooled analysis. It is the equal-weight mean of six per-generator cells of 450 responses each (`decision-point/h1/22_ragtruth_finetuned.py`: "per generator cell (6) and pooled (equal weight)"). Its raw AUROC (0.867) is likewise a mean of the per-cell values.

**Checks that do reproduce exactly:**
- TofuEval: D_src AUROC 0.6826 and the raw AUROCs 0.7302 and 0.7372, all equal to the prior. TofuEval is a single pooled analysis, so its design stands as registered.
- WQA: already registered per model.

**Change, RAGTruth only:**
- **Primary:** one audit per generator. Each has 450 responses, one per source, so no grouping is needed, and the baseline is D_src.
  - The compared increment is the equal-weight mean of the 6 headline increments.
  - A1 for RAGTruth becomes: `adds_information` true in at least 5 of 6 cells.
  - A2 and A3 use the 6-cell mean.
- **Secondary, reported, not part of agreement:** the pooled audit of all 2,700 responses, grouped by source.
- **Null controls N1, N2 and P1:** RAGTruth runs these on the pooled table, as registered, because that has the richer real structure (groups of 6).

Nothing else changes.
