# baseline-increment

Does a detector's score add anything beyond cheap baselines?

Hallucination detectors, uncertainty scores, LLM-judge scores, guardrail classifiers and routers are usually reported as one pooled AUROC. That number often mostly reflects things available without the detector:
- response length;
- which task or dataset an item comes from;
- question difficulty;
- which model produced it.

`baseline-increment` takes per-item scores and labels and answers one question: **does the score improve held-out prediction beyond those baselines, and by how much?** It reports the gain in AUROC with a 95% interval whose coverage has been validated, and gives a PASS / INCONCLUSIVE / FAIL verdict against a minimum gain you declare (default 0). It is built to run as an acceptance test before a detector ships, by hand or in CI.

The gain is measured under the declared baselines and model families, at your sample size: it is *predictive*, not proof that the score holds information the baselines lack. A separate permutation test of that stronger claim is reported as experimental, because it is unreliable on common data structures (see [what is documented as not working](#why-trust-the-verdict)).

## Install

```bash
pip install baseline-increment            # Python 3.10+
pip install "baseline-increment[parquet]" # adds Parquet input
```

If PyPI serves an older version than the newest [release](https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment/releases), install the tag instead: `pip install "git+https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment@v0.13.0"`. The [CHANGELOG](CHANGELOG.md) says which version each feature needs.

## Quickstart

From Inspect logs of one task, run on a few models:

```bash
baseline-increment from-inspect logs/ --list                       # what scorers and fields the logs hold
baseline-increment from-inspect logs/ --label-scorer match --score-scorer judge -o items.csv --lomo
baseline-increment items.csv --score score --label wrong --group group --rank-within model \
    --baselines lomo_share_wrong,model,log_input_words,log_output_words --report audit.html
```

**How this run is set up:**
- **The label and the detector:** the label comes from one scorer (`match`), and the detector from another (`judge`).
- **The baselines:** `--lomo` adds the error rate of the *other* models on the same item. The length columns come from the adapter.
- **The defaults are the validated preset** (2 seeds, the DeLong interval, permutations with early stopping).
- **`--report`:** writes a document that can be filed and signed off.

**The decision is the verdict on the gain.** PASS when the lower 95% bound exceeds `--min-gain` (default 0), FAIL when the upper bound is below it, INCONCLUSIVE otherwise. `--fail-if-no-increment` makes the command exit 1 unless the verdict is PASS, so a CI gate never rests on the experimental test.

### Does my LLM judge add anything beyond cheap signals?

Run it on the set you already labelled to align the judge. (`judge` is newer than 0.13.0 on PyPI. Until the next release, install it from the main branch: `pip install "git+https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment"`.)

```bash
baseline-increment judge alignment.csv --judge judge_score --label human_label \
    --baselines task,difficulty --response response_text --group prompt_id
```

It prints the judge's agreement with the human labels next to the AUROC of the baselines alone, then the judge's held-out gain beyond them with its 95% interval, the verdict and one plain sentence. `--response` adds the response length in words as a baseline. Two public examples. On FaithBench, the summariser and summary length alone predict the human hallucination label with an AUROC of 0.59. HHEM-2.1 adds +0.045 [+0.006, +0.085] beyond them. GPT-4o, whose verdicts agree with the annotators on 0.497 of summaries, adds +0.025 [−0.009, +0.059], which is inconclusive. On TREC 2024 LLMJudge, the query and the passage length alone reach 0.73, and 22 of the 33 submitted judges score no higher than that on their own. Yet all 33 add held-out signal beyond them (median +0.08).

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

**The gain reproduces independent analyses.** LettuceDetect and HHEM on RAGTruth and TofuEval, and P(True), log-probability and semantic entropy on TriviaQA across 8 models, were measured earlier by separate code. baseline-increment reproduces every increment within 0.006 AUROC. Grounding checks add +0.08 to +0.13 beyond difficulty; self-uncertainty signals add about +0.04. See [docs/validation.md](docs/validation.md).

**Its interval has validated coverage, with one stated exception.** The 95% interval for the gain (clustered DeLong on out-of-fold predictions) covered the mean estimate in 0.905–0.965 of runs across six pre-registered scenarios. In a wider pre-registered stress test (R1, 32 structures, 200 runs each; target 0.85–0.99) it covered 0.87–0.98 in 30 structures, including every no-signal and signal control. **It is not validated when the score is a three-way interaction of baselines and a category:** coverage was 0.845 and 0.695 (R1). A refit-aware interval (`--interval refit`) is now available: it covers 0.985 at both sizes, but is conservative under the null (1.000). Use it when the report's nonlinear-gain warning fires. See `docs/prereg/PREREG_R1b.md` §7.

**On human-labelled LLM-judge data, cheap baselines are a hard bar.** On SummEval, FaithBench and TREC LLMJudge (33 judges), the baselines alone nearly matched or beat raw judge agreement: 0.69 vs the best judge's 0.72 on TREC, 0.78 vs 0.49–0.60 on FaithBench. See [docs/judges.md](docs/judges.md).

Every rule is checked against acceptance criteria written down *before* the run. The record is [CALIBRATION.md](CALIBRATION.md), the frozen plans are in [docs/prereg/](docs/prereg/), and every public claim has a test ([docs/CLAIMS.md](docs/CLAIMS.md)).

| What | Evidence | Result |
|---|---|---|
| Interval for the gain (primary verdict) | 6 pre-registered scenarios, 200 runs each | coverage 0.905–0.965 |
| Interval for the gain, stress test (R1) | 32 pre-registered structures, 200 runs each | 0.87–0.98 in 30; three-way interaction 0.845 / 0.695 (not validated; `--interval refit` available, 0.985, conservative) |
| Power | planted signal | 0.81 weak, 1.00 moderate and strong |
| Agreement with independent analyses | 7 detectors, 3 benchmarks | every increment within 0.006 |
| Omitted-baseline sensitivity (`--sensitivity`) | 24 simulated designs with a real omitted variable | no false comfort in 98.0% (each family ≥ 93.8%) |
| `--calibration` intervals | simulated calibrated scores | coverage 0.955 and 0.945 |

**What is documented as not working:**
- **The conditional-information test (`adds_information`, secondary, experimental).** It permutes the score's residual given the baselines. It is reported, but it drives no verdict and no gate.
  - **When it holds:** on simple structures its false-flag rate is 0.05–0.10 (nominal 0.05).
  - **When it fails:**
    - when a baseline's effect differs by category: 26–58% false flags in simulation, and 33% on real TREC data;
    - when the score is an XOR-like function of the baselines: 75%;
    - when the score is a three-way interaction of the baselines: 100%.

    Stratifying the permutation by the category did not fix the TREC case (29%).
  - **Why it cannot be fixed in general:** no test can certify "information beyond the baselines" in general (Shah & Peters 2020). See [the diagnostic](docs/prereg/PREREG_DIAG.md).
- **A gain can be a model-class gain.** If a score encodes a function of the baselines that the baseline models cannot learn at your sample size, the gain is real for prediction but is not new information. In simulation a three-way interaction gave +0.115 at 600 items and +0.053 at 3,000. Add such functions as baseline columns when they are plausible, and read the [baseline ladder](docs/guide/audit.md) (`--ladder`).
- **Graded (ordinal) labels have no validated interval yet.** Their gain is descriptive, and the CI gate refuses them.
- **The rescaling caveat in `compare`:** when v2 is v1 rescaled per task, the regression check can flag falsely.

The statistics themselves are not new: incremental AUROC under grouped cross-validation, a DeLong-type interval, and a conditional randomisation test (Watson & Wright 2021; Candès et al. 2018). See [related work](docs/related-work.md). What the package adds is pre-specified baselines, validated defaults, a documented failure map, and output built for release decisions.

## Documentation

- **Guides:** [audit](docs/guide/audit.md) · [check](docs/guide/check.md) · [compare](docs/guide/compare.md) · [data in](docs/guide/data.md) · [graded labels and calibration](docs/guide/outcomes.md) · [reports](docs/guide/reports.md)
- **[Worked examples](docs/examples.md):** LettuceDetect on RAGTruth (+12.0 recall points at a 10% false-positive rate beyond metadata), and a synthetic RAG example.
- **[Real-data validation](docs/validation.md)**, the **[calibration record](CALIBRATION.md)** and the **[pre-registrations](docs/prereg/)**
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
validation/             real-data validation: per-item tables (no text), runner, results
docs/                   guides, pre-registrations, examples, related work
action.yml              composite GitHub Action (the gate)
```

## Licence

Apache-2.0. The shipped difficulty tables hold only question hashes and summary statistics; their sources' licences apply (see [check](docs/guide/check.md), "Read before using a benchmark table").
