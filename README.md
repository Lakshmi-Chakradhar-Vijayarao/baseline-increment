# baseline-increment

Does a detector's score add anything beyond cheap baselines?

Hallucination detectors, uncertainty scores, LLM-judge scores, guardrail classifiers and routers are usually reported as one pooled AUROC. That number often mostly reflects things available without the detector:
- response length;
- which task or dataset an item comes from;
- question difficulty;
- which model produced it.

`baseline-increment` takes per-item scores and labels and answers one question: **does the score add information beyond those baselines, under a test whose false-alarm rate has itself been measured?** It is built to run as an acceptance test before a detector ships, by hand or in CI.

## Install

```bash
pip install baseline-increment            # Python 3.10+
pip install "baseline-increment[parquet]" # adds Parquet input
```

If PyPI serves an older version than the newest [release](https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment/releases), install the tag instead: `pip install "git+https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment@v0.11.1"`. The [CHANGELOG](CHANGELOG.md) says which version each feature needs.

## Quickstart

From Inspect logs of one task, run on a few models:

```bash
baseline-increment from-inspect logs/ --list                       # what scorers and fields the logs hold
baseline-increment from-inspect logs/ --label-scorer match --score-scorer judge -o items.csv --lomo
baseline-increment items.csv --score score --label wrong --group group --rank-within model \
    --baselines lomo_share_wrong,model,log_input_words,log_output_words --fast --report audit.html
```

**How this run is set up:**
- **The label and the detector:** the label comes from one scorer (`match`), and the detector from another (`judge`).
- **The baselines:** `--lomo` adds the error rate of the *other* models on the same item. The length columns come from the adapter.
- **`--fast`:** the calibrated CI preset.
- **`--report`:** writes a document that can be filed and signed off.

**The decision is `adds_information`.** It is true only when the permutation p < 0.05 and the increment is positive. Add `--fail-if-no-increment` to make the command exit 1 when it is false.

## What it answers

| Question | Command | Guide |
|---|---|---|
| Does this score add information beyond the baselines? | `baseline-increment data.csv --score ... --label ... --baselines ...` | [audit](docs/guide/audit.md) |
| Which cheap explanations (difficulty, length, task, model) can it survive? | `baseline-increment check ...` | [check](docs/guide/check.md) |
| Is v2 of the detector better than v1, and did it lose anything? | `baseline-increment compare --old v1 --new v2 ...` | [compare](docs/guide/compare.md) |
| Same question for graded labels (severity 0–3) | `--outcome ordinal` | [outcomes](docs/guide/outcomes.md) |
| Is the raw score a good probability? | `--calibration` | [outcomes](docs/guide/outcomes.md) |
| Can I get it from my logs or platform export? | `from-inspect`, `from-lm-eval`, `--map` for CSV / Parquet / JSON | [data](docs/guide/data.md) |
| Can I file the result? | `--report audit.html` or `.md` | [reports](docs/guide/reports.md) |
| Can CI enforce it? | `--fail-if-no-increment`, `compare --fail-on-regression`, [`action.yml`](action.yml) | [sample workflow](examples/github_action/detector-gate.yml) |

## Why trust the verdict

Every decision rule is checked on simulated data where the truth is known, against acceptance criteria written down *before* the run. The record is [CALIBRATION.md](CALIBRATION.md), and the frozen plans are in [docs/prereg/](docs/prereg/).

| Test | Highest false-flag rate on no-signal scenarios (nominal 0.05) | Detection of planted signal |
|---|---|---|
| Audit, `--fast` | 0.10 (13 scenarios, adversarial ones included; a 400-run re-check of the worst, at the suite settings, gives 0.065) | 0.81 weak, 1.00 moderate and strong |
| `compare` | 0.06 (8 registered null cases) | 0.82–1.00 |
| `--outcome ordinal` | 0.08 (5 scenarios) | 1.00 |
| `--calibration` intervals | coverage 0.955 and 0.945 (target 0.85–0.99) | flags an overconfident score in 91% of runs |

**What is documented as not working:**
- **The refit-bootstrap interval for the increment** is too wide under the null, so it is never used for a decision. A replacement failed its pre-registered check.
- **The rescaling caveat in `compare`:** when v2 is v1 rescaled per task, the regression test flags falsely about 1 time in 8.

The statistics themselves are not new: a conditional randomisation test, incremental AUROC, and grouped resampling. See [related work](docs/related-work.md). What the package adds is pre-specified baselines, calibrated defaults, and output built for release decisions.

## Documentation

- **Guides:** [audit](docs/guide/audit.md) · [check](docs/guide/check.md) · [compare](docs/guide/compare.md) · [data in](docs/guide/data.md) · [graded labels and calibration](docs/guide/outcomes.md) · [reports](docs/guide/reports.md)
- **[Worked examples](docs/examples.md):** LettuceDetect on RAGTruth (+12.0 recall points at a 10% false-positive rate beyond metadata), and a synthetic RAG example.
- **[Calibration record](CALIBRATION.md)** and **[pre-registrations](docs/prereg/)**
- **[Changelog](CHANGELOG.md)** · **[Related work](docs/related-work.md)** · **[Upstream contributions](docs/UPSTREAM.md)**

## Repository layout

```
baseline_increment/     the package: core (binary engine), compare, ordinal, calibration, checks, validity,
                        adapters (Inspect, lm-eval), tables (CSV/Parquet/JSON + --map), report, cli, data/
tests/                  unit tests; test_golden.py freezes the binary engine's output
simulations/            calibration suites (suite, compare_suite, ordinal_suite, calibration_coverage, ...)
simulations/results/    every calibration output (.json) and log, cited by CALIBRATION.md and the reports
scripts/                build the shipped tables and data/calibration_summary.json
examples/               worked examples and the sample GitHub workflow
docs/                   guides, pre-registrations, examples, related work
action.yml              composite GitHub Action (the gate)
```

## Licence

Apache-2.0. The shipped difficulty tables hold only question hashes and summary statistics; their sources' licences apply (see [check](docs/guide/check.md), "Read before using a benchmark table").
