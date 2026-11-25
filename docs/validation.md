# Real-data validation

> Run with 0.11.2. The "adds information" columns below are the pre-0.13 permutation verdict, which from 0.13 is secondary and experimental. The increments, which are what this validation compares, are unchanged; the 0.12 re-run under a different null gave the same agreement.

**The question:** does baseline-increment give the right answer on real detectors, and does it keep false alarms low on real data, not just simulated data?

**How it was run:** everything here was pre-registered before any run, in [`docs/prereg/PREREG_VALIDATION.md`](prereg/PREREG_VALIDATION.md), including one addendum made before any run. The per-item tables (ids, labels, baselines and scores only, no text) are in [`validation/data/`](../validation/data/). The code is [`validation/run_validation.py`](../validation/run_validation.py), and the raw output is [`validation/results.json`](../validation/results.json).

## 1. Agreement with independent analyses: AGREES

**The comparison:** for seven detectors on three public benchmarks, an earlier analysis had already measured how much each detector adds beyond question or source difficulty. That analysis used separate code with its own estimator and its own pre-registrations. baseline-increment was run on the same data with the same difficulty baseline, using its CI preset (`--fast`).

| Benchmark | Detector | baseline-increment | Independent prior [95% CI] | Adds information (cells) | Agrees |
|---|---|---|---|---|---|
| RAGTruth (2,700 RAG responses, 6 generators) | LettuceDetect | +0.129 | +0.131 [0.111, 0.161] | 6/6 | yes |
| RAGTruth | HHEM-2.1 | +0.090 | +0.091 [0.071, 0.115] | 6/6 | yes |
| TofuEval (1,773 dialogue summaries, out of distribution) | LettuceDetect | +0.084 | +0.089 [0.066, 0.122] | 1/1 | yes |
| TofuEval | HHEM-2.1 | +0.093 | +0.095 [0.074, 0.126] | 1/1 | yes |
| TriviaQA (8 open models × 2,000 questions) | P(True) | +0.041 | +0.042 [0.038, 0.047] | 8/8 | yes |
| TriviaQA | negative mean log-prob | +0.037 | +0.038 [0.034, 0.042] | 8/8 | yes |
| TriviaQA | semantic entropy | +0.035 | +0.036 [0.033, 0.040] | 8/8 | yes |

**The three pre-registered criteria all hold:**
- **Verdict:** every detector is found to add information, in every cell (6 of 6 generators, and 8 of 8 models).
- **Size:** every increment is within 0.006 of the prior estimate, and inside its interval.
- **Ordering:** the prior's main finding is that grounding checks add far more than a model's own uncertainty signals do. That is reproduced. The smallest grounding increment, +0.084, exceeds the largest self-uncertainty one, +0.041.

**Reading the numbers:**
- **Grounding detectors** (LettuceDetect, HHEM) add +0.08 to +0.13 AUROC beyond source difficulty, both in and out of distribution.
- **A model's own uncertainty** (P(True), log-probability, semantic entropy) adds only about +0.04 beyond how hard the question is for other models. Most of its pooled AUROC is question difficulty.

**A secondary analysis, not part of the agreement test:** pooling all 2,700 RAGTruth responses, grouped by source, gives larger increments (LettuceDetect +0.201, HHEM +0.142). There, the only baseline is source difficulty, so the detector is also credited for telling generators apart. The per-generator design, which the prior used, rules that out. **Choose baselines that match the question you are asking.**

## 2. False alarms on real data structures

**The setup:** keep the real labels, groups and baselines, and replace the detector with one that carries no information. Then count how often the tool still says "adds information". There are 100 runs per cell, and the acceptance rule was ≤ 0.10.

| Real table | Real detector shuffled within difficulty × task strata | Detector = function of difficulty + noise | Planted signal (0.6 SD) detected |
|---|---|---|---|
| RAGTruth (pooled, groups of 6 responses per source) | 0.08 [0.04, 0.15] | 0.04 [0.02, 0.10] | 1.00 |
| TofuEval | 0.06 [0.03, 0.12] | 0.08 [0.04, 0.15] | 1.00 |
| TriviaQA, Mistral-7B | 0.01 [0.00, 0.05] | 0.05 [0.02, 0.11] | 1.00 |

**Result:** all six no-signal checks pass.
- **Pooled:** across all 600 runs the false-alarm rate is 32/600 = 0.053 [0.038, 0.074], consistent with the nominal 0.05.
- **Planted signal:** it was detected in every run on all three tables.

## 3. Stress test (simulated)

**What was tested** ([`CALIBRATION.md`](../CALIBRATION.md), "Stress test after 0.11.1"):
- 10 baseline columns, 7 of them noise;
- 10,000 rows;
- groups of 1 to 40 items;
- clustered detector noise;
- 400-run re-checks of the three borderline scenarios.

**All 11 no-signal scenarios stay within the 0.10 limit.**

**One is borderline:**
- **The case:** a detector that is an exact *interaction* of two baselines (task × metadata).
- **The rate:** false alarms at 0.095 [0.070, 0.128] over 400 runs, about twice the nominal rate.
- **What to do:** when a detector could be an interaction of your baselines, add that interaction as an explicit baseline column.

## What this does and does not show

**Shown:**
- On seven real detectors, the tool reproduces independent results to within 0.006 AUROC.
- On real data structures, its false-alarm rate is about 5%.

**Not shown:**
- **LLM judges:** none are among the validated detectors; they would need new model runs.
- **Other domains:** code, safety or refusal detectors were not tested.
- **Correctness on these benchmarks** beyond agreeing with one earlier analysis.
