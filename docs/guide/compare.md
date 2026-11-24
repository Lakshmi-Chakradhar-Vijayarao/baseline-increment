# Comparing two versions of a detector



## Compare two versions of a detector (v0.8)

When a detector, judge or guardrail score gets a new version, `compare` answers three separate questions on the same labelled items:

```bash
baseline-increment compare items.csv --old score_v1 --new score_v2 --label wrong --group group \
    --baselines task,log_output_words --fail-on-regression
```

| Question | Test | What "detected" means |
|---|---|---|
| **Improvement:** does v2 add information beyond v1 and the baselines? | the audit of v2 with v1 added as a baseline | v2 knows something v1 did not |
| **Regression:** does v1 keep information that v2 lacks? | the audit of v1 with v2 added as a baseline | the new version lost something; `--fail-on-regression` exits 1 |
| **Standalone:** is AUROC(v2) − AUROC(v1) different from 0? | paired group-bootstrap interval | v2 is better (or worse) on its own |

**How the questions relate:**
- **They can disagree, and the report never merges them.** A v2 can add information without being better alone; it may be complementary.
- **Both detected:** when improvement and regression are both detected, the report says the scores are complementary and worth combining.
- **Same footing:** both scores get the identical transform (rank, ranked within `--rank-within` if given).
- **Multiplicity:** improvement and regression share a Holm correction.
- **Non-rejections:** reported as "not detected (n items, n groups)", never as "adds nothing".

**Calibration** (pre-registered in [`docs/PREREG_0.8.md`](../../docs/prereg/PREREG_0.8.md); 100 runs per scenario; full table in [`CALIBRATION.md`](../../CALIBRATION.md)):
- **False flags:** at most 0.06 on every registered null.
- **Detection:** 0.82–1.00 for planted improvements and regressions.
- **Standalone interval:** coverage 0.95–0.96.

**One caveat, found outside the registration:** when v2 is v1 rescaled differently per task, the regression test flagged 12% of runs. Treat a lone regression on such a change with care.

**Model-vs-model accuracy comparisons** (paired tests, clustered errors, power) are not what this tool does. [`evalconfidence`](https://github.com/stephlinds/evalconfidence) covers them for Inspect and tabular results.
