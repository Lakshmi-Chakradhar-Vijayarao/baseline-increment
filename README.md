# baseline-increment

Does a detector's score add anything beyond cheap baselines?

Hallucination detectors, uncertainty scores, probes, routers and early-exit signals are usually reported as one pooled AUROC. That number often mostly reflects things available without the model:
- response length;
- which dataset or task an item comes from;
- question difficulty;
- output confidence.

`baseline-increment` takes per-item scores and metadata and answers one question: **does the detector add information beyond those baselines, under a test that has itself been checked not to give false positives?**

## Where it fits

Use it as an **acceptance test before a detector ships**: a hallucination detector, guardrail classifier, LLM-judge score, risk score or router. It answers what a release review needs to know: does this new signal beat a length or task heuristic, or is it re-reading metadata the system already has?

- **In CI:** run `audit()` on a held-out evaluation set and fail the build when the report's `adds_information` is false. That field is true only when the permutation p-value is below 0.05 and the increment is positive.
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
- **A CI from a refit group bootstrap** of the whole estimator (all seeds, all refits, the headline choice).
- **A refit permutation p-value:**
  - the null is a conditional randomisation: the detector's dependence on the baselines is fitted out of fold, and only the residual part is shuffled across groups (optionally within your own strata, such as model or difficulty decile). A detector that is just a function of the metadata therefore gets a null that is equally such a function;
  - everything is refit; test labels are never shuffled alone;
  - no DeLong test for nested models.

  `adds_information` is true only if p < 0.05 **and** the increment is above 0.
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

**α.** The decision rule is nominal **α = 0.05**: `adds_information` requires a permutation p < 0.05 and a positive increment. The "≤ 0.10" figure elsewhere is not α. It is the development acceptance band for the measured false-flag rate in the simulation suite: 100 replicates cannot separate a true 5% rate from 7%, so a looser band avoids failing a correct test by chance.

**Why 0.05 for a release gate.** α is the chance of accepting a detector that adds nothing, at any sample size. Sample size changes power, not α. A wrongly accepted detector ships a signal that only re-reads metadata, while a wrongly rejected one costs a larger evaluation set and a re-run. So the gate keeps the conventional 0.05 and reports power separately (table below). Teams that gate often should account for repeated looks: this tool reports p-values, not a sequential procedure.

**Power** on the five planted-signal scenarios (100 runs each; full table with intervals in [`CALIBRATION.md`](CALIBRATION.md)). The signal is the detector's dependence on the label, in units of the score's noise SD:

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

## `check`: try to break a detector's claim (v0.3)

One command runs every cheap-explanation test your columns allow, and gives each a status:

```
baseline-increment check answers.csv --score p_true_risk --label hallucinated \
    --question question_id --model model --response answer \
    --benchmark triviaqa --question-text question --html report.html --json report.json
```

| Status | Rule |
|---|---|
| SUPPORTED | increment ≥ 0.02 (`--min-increment`), bootstrap CI lower bound > 0, Holm-adjusted permutation p < 0.05 |
| NOT SUPPORTED | bootstrap CI upper bound < 0.02: whatever the detector adds beyond this explanation is smaller than that |
| INCONCLUSIVE | anything else |
| NOT CHECKABLE | the needed column is missing; the report says which |

**What it tests:**
- **Question difficulty:**
  - `--benchmark` joins a shipped difficulty table, or you give your own column with `--difficulty`.
  - Tables cover TriviaQA (rc.nocontext train), NQ-open (validation), PopQA (test) and SciQ (test + validation), 2,000 questions each. They were measured on 6–8 open models (Qwen2.5 0.5–14B, Mistral-7B, OLMo-2-7B, Phi-3.5-mini).
  - **Join:** by question text (`--question-text`), via a sha1 of the lower-cased, whitespace-collapsed text, so no question text is shipped. Rows that don't match are dropped, with a warning below a 90% match rate.
  - **Columns:** other models' error rate (label-based) and a label-free version (answer dispersion plus mean confidence rank).
- **Response length:** `--response` (word count) or `--length`.
- **Task / source:** `--task`, plus the detector's AUROC within each task.
- **Model identity:** `--model`.
- **All of the above together.**
- **Same-question test** (`--question` + `--model`, several models per question): on questions where some models fail and others don't, does the detector (ranked within each model) put the failing model above the succeeding one?
  - Reports a pair AUROC with a question-bootstrap CI.
  - If statsmodels is installed, also reports a conditional-logit coefficient with model fixed effects.
  - SUPPORTED needs a CI lower bound > 0.5, a pair AUROC ≥ 0.55 and, when available, β > 0 with p < 0.05.
  - It shows model-specific information, not *why*: a length difference between models' answers also passes it.
- **Always listed as not checkable from scores:** leakage or test-set tuning, out-of-distribution generalisation, label quality, downstream usefulness.

**Read before using a benchmark table:**
- **If your model is one of the reference models, the table includes its own labels.** Increments over it are then biased toward zero (`check` warns).
- **Labels:** the reference models were graded by an LLM judge on TriviaQA, NQ and SciQ, and by alias string match on PopQA. Your labels may differ.
- **Data licences:** the shipped tables hold only question hashes and summary statistics.
  - SciQ is CC BY-NC 3.0. Drop `baseline_increment/data/difficulty_sciq.csv` if you need a fully permissive package.
  - TriviaQA, NQ-open, PopQA and RAGTruth tables follow their sources' licences.
- **Runtime:** for 16,000 rows with defaults (200 bootstraps, 200 permutations, 3 seeds), expect tens of minutes. `--jobs -1` uses all cores. With few permutations Holm cannot reach 0.05, and `check` warns.

**Example:** all 8 WQA models on TriviaQA, P(True) risk. The detector's AUROC is 0.781. On the same question it ranks the failing model first 69% of the time (pair AUROC 0.688, CI 0.675–0.698). The warning about the reference models applies here, because these *are* the reference models.

Statements in the report are fixed sentences tied to the rules above; nothing is generated.

## v0.4: how much error is model-specific, and how much of it does the detector read?

The increment over difficulty answers "does the detector add anything beyond how hard the question is?" v0.4 adds the quantities needed to say *how much there is to add*, and how much of it the detector recovers. They need repeated sampled answers per (model, question):

```
baseline-increment check answers.csv --score p_true_risk --label greedy_wrong --question item --model model \
    --samples s1_wrong,s2_wrong,s3_wrong,s4_wrong,s5_wrong,s6_wrong,s7_wrong,s8_wrong,s9_wrong,s10_wrong
```

**What the report adds:**
- **Error-propensity decomposition.** Each (model, question) has an error rate p̂, the share of its sampled answers that are wrong.
  - Its variance is split into:
    - a shared question part (hard for everyone);
    - a model-specific part (hard for this model);
    - binomial sampling noise.
  - The samples are split into two halves, giving two independent estimates. Their split-half correlation shows whether the model-specific part is a repeatable property or noise.
  - The **model-specific share** comes with a question-bootstrap CI.
- **R²_v:** the noise-corrected fraction of the model-specific part that the detector's within-question deviation explains. The slope is fitted on one sample half and scored on the other.
  - It is refused (NOT DEFINED) when there is almost no model-specific variance to explain (σ²_v < 0.005).
- **Ability-artefact check (additive null).** On the probability scale, models that differ only in overall ability, with no model × question interaction at all, still diverge on middling questions. The decomposition would read that divergence as "model-specific".
  - The report simulates exactly that case with your panel's model accuracies and states the share it produces.
  - It also gives the **excess** over that share.
  - The share depends on which models are in the panel. Quote it per panel, never as a property of the models alone.
- **Panel sensitivity.** The increment over D1, other models' error rate on the same question, computed from k = 1 … K−1 reference models.
  - D1 from few models is a noisy difficulty estimate, so the increment over it is inflated. The report shows how much.
  - The c/k extrapolation is an assumption and is labelled so.
- **Incremental d′.** The equal-variance binormal transform of the (baseline, baseline + detector) AUROC pair: the detector's added information on a scale that does not shrink as the baseline gets stronger.
  - It is model-based, and reported beside the AUROCs, not instead of them.
- **Evidence ladder and headline.** One row per claim the data can support. Untested rungs say NOT TESTED, never pass or fail.
  - Some rungs are always listed as untestable from scores: leakage, out-of-distribution generalisation, label quality, usefulness.
  - The headline is the within-question pair AUROC when that test is possible. A difficulty increment is headlined only when `--panel-meta` documents the reference panel.

**Why these exist.** On 7–13 open models (Qwen2.5, Llama 3.x, Gemma-2, Mistral, OLMo-2, Phi-3.5) over PopQA and TriviaQA:
- **Model-specific share:**
  - On the probability scale it is 0.28–0.48, depending on the panel's ability spread.
  - On the logit scale, which is free of the ability artefact, it is 0.09–0.17 (PopQA) and 0.21–0.29 (TriviaQA). It is lowest in narrow single-family panels and highest in the broadest 12–13-model panel.
  - It is repeatable: split-half r 0.88–0.90.
- **Recovery:** common detectors (P(True), log-probability, hidden-state probes) recover R²_v 0.03–0.15 of it.
- **The increment over a single-model D1 shrinks** as the reference panel grows.

## Use

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

## From your evaluation harness

If your evaluations already run in **lm-evaluation-harness** or **Inspect**, convert their per-sample logs instead of building the table by hand:

```bash
# lm-eval: run with --log_samples; the label is "wrong" (1 - acc), the score is the model's own
# risk (1 - top-choice probability from the choices' log-likelihoods)
baseline-increment from-lm-eval out/*/samples_*.jsonl -o items.csv
baseline-increment items.csv --score risk --label wrong --group group --baselines task,log_prompt_words

# Inspect: label from one scorer, detector from another scorer (or a numeric metadata field)
baseline-increment from-inspect logs/*.json --label-scorer match --score-scorer judge -o items.csv
baseline-increment items.csv --score score --label wrong --group group --baselines log_input_words,log_output_words
```

Both adapters are tested against a known result: on real logs (lm-eval on pythia-14m, Inspect on its mock model), the per-sample table reproduces the harness's own reported accuracy exactly. `.eval` Inspect logs are compressed and need `inspect_ai` installed; `.json` logs need nothing extra.

## Does the test itself work? (`simulations/`)

`simulations/suite.py` is a permanent calibration suite. Its acceptance rules were fixed before the first run. "Flag rate" means the rate of `adds_information = true`. The current numbers are in `simulations/results_v<version>.json` and in the table below.

Summary for the audit engine (calibrated at v0.2.2, unchanged in 0.4.0; 100 runs per scenario; full table with 95% intervals in [`CALIBRATION.md`](CALIBRATION.md); settings in [`MANIFEST.json`](MANIFEST.json)):

| Scenario type | Flag rate |
|---|---|
| 8 nulls (noise, metadata copy, nonlinear metadata, random labels, few or unequal groups, imbalance) | 0.01–0.07 |
| 5 adversarial nulls (misspecified baseline, interaction, difficulty or scale varying by model, Simpson pooling) | 0.04–0.10 |
| Weak planted signal (0.3 SD) | 0.82 |
| Moderate, strong, signal-with-difficulty, hard-items-only signal | 0.99–1.00 |

What this supports: no material anti-conservative behaviour under the tested nulls. The ≤ 0.10 limit is a development acceptance criterion, not proof of calibration. The misspecified-baseline null sits exactly at 0.10. Bootstrap CIs were conservative under pure noise in the (v0.2.0) coverage run. Exact 95% calibration is not claimed.

**v0.4 estimands** (`simulations/validity_suite.py`; 100 replicates per cell, Q ∈ {300, 1000} × M ∈ {3, 6, 10}; criteria fixed before the first run):
- false "recovers model-specific error" for a pure-noise detector: 0.00–0.01;
- power for a planted detector: 1.00;
- CI coverage of the population share: 0.91–0.97;
- ability-only excess: |≤ 0.003|;
- panel-sensitivity null: ≤ 0.006.

The first run failed two criteria. Details are in [`CALIBRATION.md`](CALIBRATION.md).

**History:**
- **v0.5.1:** `operating_point()`, which reports the true-positive rate at a fixed false-positive rate for baselines alone and for baselines + detector, applied to the RAGTruth example. Reporting only; the test is unchanged.
- **v0.5.0:**
  - harness adapters: `from-lm-eval` and `from-inspect`, tested to reproduce each harness's own accuracy;
  - README section on α, power and prior art;
  - worked example on a public detector (LettuceDetect on RAGTruth).

  The audit engine is unchanged from v0.2.2, so the calibration results above still apply.
- **v0.4.0:** decomposition, R²_v, additive null, panel sensitivity, incremental d′, evidence ladder.
- **v0.4.1:** packaging and documentation only (project links, install from PyPI, dependency note).
  - Its first calibration run found that an in-sample R²_v slope can never be distinguished from 0 by its CI (100% false recovery). R²_v is now cross-half.
- **v0.1:** used only a linear baseline. A detector that was a nonlinear function of the metadata got a "significant" +0.088 in 100 of 100 runs. Its CI came from a single CV seed.
- **v0.2.0:** added flexible baselines and a refit bootstrap. Its battery then found that an unconfined permutation flagged an interaction-only detector in 17% of runs.
- **v0.2.1:** confined permutations to risk bands and required a positive increment. That made the interaction null worse (27% flagged): an exact metadata function beats any baseline that only approximates it at finite n.
- **v0.2.2:** conditional-randomisation null (fitted detector + shuffled residuals). Interaction null 0.27 → 0.09; all 18 scenarios within their limits.

## Worked example on a public detector (LettuceDetect on RAGTruth)

LettuceDetect (`KRLabsOrg/lettucedect-base-modernbert-en-v1`, revision `bbd7783`) scored all 2,700 RAGTruth test responses (RAGTruth at commit `c103204`, MIT). The run was on a Kaggle T4, through `examples/ragtruth_lettucedetect/kaggle_kernel.py`. Score = max token hallucination probability; label = any annotated hallucination span. The baselines are what a release review would ask about: task type, the generating model, and response length. Groups are source documents (each has six responses). The scores, labels and the full output are in `examples/ragtruth_lettucedetect/kaggle_output/`.

```
baseline-increment 0.4.1
items 2700  groups 450  positive rate 0.349
detector AUROC            0.889 [0.872, 0.904]
baseline task_type         0.717  (linear, alone)
baseline generator         0.711  (linear, alone)
baseline log_words         0.662  (linear, alone)
family linear  baselines 0.851  + detector 0.914  increment +0.062
family spline  baselines 0.851  + detector 0.913  increment +0.062
family gbm     baselines 0.853  + detector 0.913  increment +0.060  <- headline (best baselines-only AUROC)
increment (gbm)      +0.060 [0.047, 0.075]  perm p=0.005
stratum 'task_type': base rate alone 0.717
  Data2txt             n=900   pos=0.64  detector AUROC 0.904 [0.882, 0.923]
  QA                   n=900   pos=0.18  detector AUROC 0.900 [0.867, 0.930]
  Summary              n=900   pos=0.23  detector AUROC 0.756 [0.712, 0.797]
```

**What it says:**
- The detector adds information beyond the metadata: +0.060 AUROC over the best metadata baseline, CI [0.047, 0.075], permutation p = 0.005.
- But metadata alone already reach 0.851, so most of the pooled 0.889 is available without the detector.
- Within task, it is strong on QA and data-to-text (about 0.90) and weaker on summaries (0.756).

**In use:** at a fixed false-positive rate of 10%, the metadata alone catches 63.7% of hallucinated responses, and adding the detector raises that to 75.7% (+12.0 points). At 5% the gain is +16.3 points (52.3% → 68.6%). These are out-of-fold figures for the headline gbm family, from `operating_point()` in `kaggle_output/operating_point.json`.

This is the shape of answer the tool exists for:
- **a real increment,** worth about 12 points of recall at a 10% false-positive rate;
- **much smaller than the pooled AUROC suggests,** because most of that AUROC was available from metadata alone;
- **uneven across tasks.**

**Prior art for this effect:** Dubanowska et al. (EMNLP 2025 Findings) showed that task imbalance drives much of the reported performance of representation-based detectors on RAGTruth. This example applies the same question to a trained token-level detector, with a calibrated test.

## Worked example (synthetic, RAG-style)

`examples/synthetic_ragstyle.csv` (in the GitHub repository, not in the PyPI package): 600 responses from three task types and three generators. Hallucination rates differ by task, and the "detector" mostly tracks task type, with a little real signal.

```
baseline-increment examples/synthetic_ragstyle.csv --score detector_score --label hallucinated \
    --group response_id --baselines task_type,generator,log_words --strata task_type --boot 100 --perms 200 --jobs -1

baseline-increment 0.4.1   # reproduced from a clean PyPI install with requirements-lock.txt (15 Nov 2026)
items 600  groups 600  positive rate 0.348
detector AUROC            0.730 [0.687, 0.773]
baseline task_type         0.703  (linear, alone)
baseline generator         0.598  (linear, alone)
baseline log_words         0.456  (linear, alone)
family linear  baselines 0.759  + detector 0.784  increment +0.025  <- headline (best baselines-only AUROC)
family spline  baselines 0.758  + detector 0.778  increment +0.019
family gbm     baselines 0.741  + detector 0.758  increment +0.017
increment (linear)      +0.025 [-0.008, 0.051]  perm p=0.005
stratum 'task_type': base rate alone 0.703
  data2text            n=200   pos=0.64  detector AUROC 0.621 [0.536, 0.702]
  qa                   n=200   pos=0.18  detector AUROC 0.638 [0.532, 0.739]
  summary              n=200   pos=0.23  detector AUROC 0.641 [0.540, 0.731]
WARNING: 'task_type' base rate alone reaches 0.703: at least 80% of the detector's pooled AUROC above chance (0.730) is stratum identity
WARNING: a cheap baseline (0.759) matches or beats the detector (0.730)
```

This takes about 10 minutes on one core and under a minute on 8 cores (`--jobs -1`). The output is identical either way.
With the pinned versions in `requirements-lock.txt` the output matches exactly. Newer library versions can change the last digit of a bootstrap interval (e.g. pandas 3.1 / scikit-learn 1.9 give −0.009 for the −0.008 lower bound above).


Here the detector does add a little (+0.025, permutation p = 0.005, so `adds_information` is true), but the refit-bootstrap CI [−0.008, 0.051] still includes zero. The CI is deliberately conservative (see CALIBRATION.md); the decision rule uses the permutation test.

The pooled 0.73 looks respectable. The cheap baselines beat it, task identity explains most of it, and within each task the detector is a modest ~0.63.

This pattern is real on RAGTruth.
- Dubanowska et al. (EMNLP 2025 Findings, arXiv 2509.19372) show that a "flag every data-to-text response" rule nearly matches published representation-based detectors.
- On the full RAGTruth test set, task type plus generator identity alone reach AUROC ≈ 0.84.
- We also ran the tool on ReDeEP's released RAGTruth scores (one CV seed's held-out scores, seed 0). That example is not shipped, because its source has no licence. The 20-seed means are 0.41 / 0.58 / 0.63 within task.

## Related work

This package implements guideline (I) of Dubanowska et al. (2025), "check against naïve classifiers based on training set features", as an executable, calibrated test.

The idea of a partial-input or shortcut baseline is old:
- hypothesis-only NLI baselines (Gururangan et al. 2018; Poliak et al. 2018);
- control tasks for probes (Hewitt & Liang 2019);
- leakage checklists (Kapoor & Narayanan 2023; REFORMS);
- incremental-value statistics in clinical prediction (Pepe et al.; Pencina et al.).

For hallucination detectors specifically, length and benchmark artifacts have been reported:
- Janiak et al. 2025, arXiv 2508.08285;
- PARALLAX, arXiv 2605.17028;
- length bias in UQ evaluation, arXiv 2504.13677.

Escalation-signal evaluation has a similar checklist (arXiv 2610.07354).

What this package adds is an executable, detector-agnostic increment test with:
- flexible baselines;
- refit permutation and bootstrap;
- within-stratum breakdowns;
- a published calibration suite.

A different `shortcut-audit` package exists for sensor datasets (github.com/WayneUSC/shortcut-audit).

License: Apache-2.0.
