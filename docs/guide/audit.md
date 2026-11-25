# The audit: does a score improve held-out prediction beyond cheap baselines?

The core test. For running it from harness logs or other exports, see [data.md](data.md); for CI use, see the README quickstart.

## Where it fits

Use it as an **acceptance test before a detector ships**: a hallucination detector, guardrail classifier, LLM-judge score, risk score or router. It answers what a release review needs to know: does this new signal beat a length or task heuristic, or is it re-reading metadata the system already has?

- **In CI:** run `audit()` (or the CLI with `--fail-if-no-increment`) on a held-out evaluation set and fail the build unless `incremental_prediction.verdict` is `PASS`: the lower bound of the validated 95% interval for the gain exceeds your declared minimum gain (default 0).
- **When comparing detectors or vendors:** run each candidate against the same baselines, so they are compared on what they add, not on pooled AUROC.
- **When a difficulty baseline comes from other models:** the tool warns when fewer than five models define it, because increments shrink as that reference panel grows.

## What it reports

- **Detector AUROC**, with a CI that resamples groups (a question or response).
- **Baselines alone and combined**, under three pre-specified model families:
  - `linear`: L2 logistic regression;
  - `spline`: cubic B-splines on numeric columns;
  - `gbm`: shallow gradient boosting, which captures interactions.

  All use grouped cross-validation, so a group never straddles train and test.
- **The increment:** AUROC(baselines + detector) − AUROC(baselines), averaged over CV seeds.
  - The headline uses the family whose *baselines alone* predict best out of fold. It is chosen without looking at the detector.
  - Every family's increment is shown.
- **A 95% interval for the gain** (from 0.13: a clustered DeLong interval on the seed-averaged out-of-fold predictions, grouped by your group column). Its coverage was validated in six pre-registered scenarios (0.905–0.965). **It decides the primary verdict:** PASS when the lower bound exceeds the minimum gain, FAIL when the upper bound is below it, INCONCLUSIVE otherwise. (The older refit-bootstrap interval is kept in the Python API for reproducing earlier results; it is conservative.)
- **A refit permutation p-value:**
  - the null is a conditional randomisation: the detector's dependence on the baselines is fitted out of fold, and only the residual part is shuffled across groups (optionally within your own strata, such as model or difficulty decile). A detector that is just a function of the metadata therefore gets a null that is equally such a function;
  - everything is refit; test labels are never shuffled alone;
  - no DeLong test for nested models.

  `adds_information` is true only if p < 0.05 **and** the increment is above 0. **From 0.13 this is a secondary, experimental result that drives nothing.** Its false-flag rate exceeds the nominal level when a baseline's effect differs by category (26–58% in simulation, 33% on real TREC data), or when the score is an XOR-like or higher-order function of the baselines ([diagnostic](../prereg/PREREG_DIAG.md)). No test can certify "information beyond the baselines" in general (Shah & Peters 2020).
- **AUROC within each stratum** (e.g. task type), and the AUROC of a score that only knows each stratum's base rate.
- **Warnings** when:
  - a stratum's base rate explains most of the pooled AUROC;
  - a cheap baseline matches the detector;
  - the increment is not distinguishable from zero;
  - **the increment over the linear baseline exceeds the increment over the flexible one**, meaning part of the detector's apparent value is just a nonlinear function of the metadata;
  - no group column is given;
  - a stratum has fewer than 10 items in one class;
  - the score is inverted.

The detector enters as its rank, optionally ranked within a column such as `model`, so rescaling the score cannot change the result.

## Error rates, power, and what is (and isn't) new

**α.** The primary verdict uses a two-sided 95% interval, so a PASS against a minimum gain of 0 is a one-sided test at about 2.5%. The secondary permutation test is nominal α = 0.05. On simple structures it runs at 0.05–0.10, and on the structures above it is far higher. The "≤ 0.10" band elsewhere is the development acceptance band for measured false-flag rates, not α.

**Why 0.05 for a release gate.** α is the chance of accepting a detector that adds nothing, at any sample size. Sample size changes power, not α. A wrongly accepted detector ships a signal that only re-reads metadata, while a wrongly rejected one costs a larger evaluation set and a re-run. So the gate keeps the conventional 0.05 and reports power separately (table below). Teams that gate often should account for repeated looks: this tool reports p-values, not a sequential procedure.

**Power** on the five planted-signal scenarios (100 runs each; full table with intervals in [`CALIBRATION.md`](../../CALIBRATION.md)). The signal is the detector's dependence on the label, in units of the score's noise SD:

On the null scenarios, all 13 are consistent with α = 0.05:
- 100 runs each;
- the misspecified-baseline null, re-run with 400 runs, flags 0.065 [0.045, 0.094] (see `CALIBRATION.md`).

| Scenario | Planted effect | Mean AUROC increment | Flag rate |
|---|---|---|---|
| `weak_signal` | 0.3 SD | +0.009 | 0.82 |
| `moderate_signal` | 0.6 SD | +0.033 | 1.00 |
| `strong_signal` | 1.2 SD | +0.101 | 1.00 |
| `signal_with_difficulty` | 0.6 SD, on a detector that also copies the difficulty the baselines already see | +0.029 | 0.99 |
| `hard_only_signal` | 1.2 SD on hard items only | +0.029 | 1.00 |

**One failure the tests catch.** Under a linear-only baseline, a detector that is a nonlinear function of the metadata got a "significant" +0.088 in 100 of 100 runs (v0.1). The `nonlinear_of_metadata` and `interaction_meta` scenarios now fail the build if that comes back.

**Methods it builds on (nothing here is a new statistical method):**
- **The null** is a conditional randomisation test (Candès, Fan, Janson & Lv 2018, model-X knockoffs / CRT): the detector's dependence on the baselines is modelled, and only the residual is permuted.
- **Incremental AUROC and the warning against DeLong's test for nested models** follow Pepe, Kerr, Longton & Wang (2013) and Demler, Pencina & D'Agostino (2012).
- **Grouped CV and the refit group bootstrap** are standard resampling practice.

What this package adds is the **packaging**: pre-specified baseline families, defaults, the `check` battery, a published calibration suite, and a CLI and JSON output built to run in CI.

## Use

> **Version note:** the harness adapters need 0.5.0 or later, `operating_point()` needs 0.5.1 or later, `--lomo` and the `helm_nq` table need 0.6.0 or later, `--fast`, `--fail-if-no-increment`, `--perm-stop`, `--interval` and the GitHub Action need 0.7.0 or later, `compare` needs 0.8.0 or later, native `.eval` reading, `--list`, `--map`, Parquet/JSONL input and item grouping need 0.9.0 or later, `--report` needs 0.10.0 or later, `--outcome ordinal` and `--calibration` need 0.11.0 or later, and the gain-and-interval verdict, `--min-gain`, the validated default preset, `--policy`, `--evidence-dir`, `--sensitivity` and `--ladder` need 0.13.0 or later. Check with `pip show baseline-increment`. If PyPI gives you an older version, install the tagged release: `pip install "git+https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment@v0.13.0"`.

```
pip install baseline-increment          # or: pip install -r requirements-lock.txt baseline-increment
baseline-increment data.csv --score detector_score --label hallucinated --group response_id \
    --baselines task_type,log_words --strata task_type --jobs 8 --json report.json
```

- Columns that aren't numeric are one-hot encoded.
- `--label` is 1 for the event the detector should flag.
- `--perm-strata model,difficulty_decile` keeps every permutation inside those levels.
- `--rank-within model` puts scores from different models on one scale.
- `--boot 0` skips the (slow) refit bootstrap.

In Python:

```python
from baseline_increment import audit
rep = audit(df, score="S", label="wrong", group="qid", baselines=["D"],
            seeds=range(5), n_perm=200, n_boot=200, perm_strata=["D_decile"], n_jobs=8)
rep["increment"], rep["increment_ci"], rep["increment_perm_p"], rep["adds_information"]
```

## Does the test itself work?

`simulations/suite.py` is a permanent calibration suite. Its acceptance rules were fixed before the first run. "Flag rate" means the rate of `adds_information = true`. The current numbers are in `simulations/results_v<version>.json` and in the table below.

Summary for the audit engine (calibrated at v0.2.2, unchanged in 0.4.0; 100 runs per scenario; full table with 95% intervals in [`CALIBRATION.md`](../../CALIBRATION.md); settings in [`MANIFEST_v0.4.1.json`](../../simulations/results/MANIFEST_v0.4.1.json)):

| Scenario type | Flag rate |
|---|---|
| 8 nulls (noise, metadata copy, nonlinear metadata, random labels, few or unequal groups, imbalance) | 0.01–0.07 |
| 5 adversarial nulls (misspecified baseline, interaction, difficulty or scale varying by model, Simpson pooling) | 0.04–0.10 |
| Weak planted signal (0.3 SD) | 0.82 |
| Moderate, strong, signal-with-difficulty, hard-items-only signal | 0.99–1.00 |

What this supports: no material anti-conservative behaviour under the tested nulls. The ≤ 0.10 limit is a development acceptance criterion, not proof of calibration.
- **The misspecified-baseline null** flagged exactly 0.10 at 100 runs. Re-run with 400 runs, it flags 0.065 [0.045, 0.094] (see `CALIBRATION.md`).
- **The refit-bootstrap CI fails its own coverage check under the null.** In the coverage run (40 replicates), it covered the mean estimate 1.000 of the time under pure noise and under a misspecified baseline, above the 0.85–0.99 acceptance band (`CALIBRATION.md` marks both FAIL). The intervals are **too wide** there, so they are conservative: they rarely exclude 0 when nothing is there, at the cost of power. Under a moderate signal, coverage was 0.925.
- **Exact 95% calibration is not claimed.** Use the permutation test for the decision and read the interval as conservative.
- **0.7.0 tried a replacement and rejected it.** A half-sample interval, pre-registered in `docs/PREREG_0.7.md`, failed the same check: coverage was 1.000 under noise and under a misspecified baseline. So the over-width does not come from duplicated groups in the bootstrap. The interval is now printed on its own line, marked "not used for the decision".

**v0.4 estimands** (`simulations/validity_suite.py`; 100 replicates per cell, Q ∈ {300, 1000} × M ∈ {3, 6, 10}; criteria fixed before the first run):
- false "recovers model-specific error" for a pure-noise detector: 0.00–0.01;
- power for a planted detector: 1.00;
- CI coverage of the population share: 0.91–0.97;
- ability-only excess: |≤ 0.003|;
- panel-sensitivity null: ≤ 0.006.

The first run failed two criteria. Details are in [`CALIBRATION.md`](../../CALIBRATION.md).
