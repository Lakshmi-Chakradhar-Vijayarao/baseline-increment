# baseline-increment

Does a detector's score add anything beyond cheap baselines?

Hallucination detectors, uncertainty scores, probes, routers and early-exit signals are usually reported as one pooled AUROC. That number often mostly reflects things available without the model:
- response length;
- which dataset or task an item comes from;
- question difficulty;
- output confidence.

`baseline-increment` takes per-item scores and metadata and answers one question: **does the detector add information beyond those baselines, under a test that has itself been checked not to give false positives?**

## Quickstart

From Inspect logs of one task, run on a few models (one `inspect eval ... --log-format json` per model):

```bash
pip install baseline-increment
baseline-increment from-inspect logs/*.json --label-scorer match --score-scorer judge -o items.csv --lomo
baseline-increment items.csv --score score --label wrong --group group --rank-within model \
    --baselines lomo_share_wrong,model,log_input_words,log_output_words --fast --jobs -1
```

- **The label and the detector:** the label comes from one scorer (`match`) and the detector from another (`judge`).
- **The baselines:**
  - `--lomo` adds `lomo_share_wrong`, the error rate of the *other* models on the same item;
  - the length columns come from the adapter.
- **`--fast`:** the CI preset. Its calibration is in [`CALIBRATION.md`](CALIBRATION.md).

The decision is `adds_information`. It is true only when the permutation p < 0.05 and the increment is positive.

**As a CI gate:** add `--fail-if-no-increment`, which makes the command exit 1 when the detector adds nothing. A ready-made GitHub Action is in [`action.yml`](action.yml), with a sample workflow in [`examples/github_action/detector-gate.yml`](examples/github_action/detector-gate.yml).

**Without a harness:** any CSV with a score, a 0/1 label and baseline columns works. See [Use](#use).

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
- **An interval from a refit group bootstrap** of the whole estimator (all seeds, all refits, the headline choice). It is conservative, too wide under the null, and is **not** used for the decision; see the calibration notes below.
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
  - Tables cover TriviaQA (rc.nocontext train), NQ-open (validation), PopQA (test) and SciQ (test + validation), 2,000 questions each. They were measured on 6–8 open models (Qwen2.5 0.5–14B, Mistral-7B, OLMo-2-7B, Phi-3.5-mini). Prefer difficulty from your own panel (`--lomo`, below): these tables come from one protocol and a mostly-Qwen panel. `helm_nq` (v0.6) is a 41-family table under HELM's protocol.
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

> **Version note:** the harness adapters need 0.5.0 or later, `operating_point()` needs 0.5.1 or later, `--lomo` and the `helm_nq` table need 0.6.0 or later, and `--fast`, `--fail-if-no-increment`, `--perm-stop`, `--interval` and the GitHub Action need 0.7.0 or later. Check with `pip show baseline-increment`. If PyPI gives you an older version, install the tagged release: `pip install "git+https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment@v0.7.0"`.

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

If your evaluations already run in **lm-evaluation-harness** or **Inspect**, convert their per-sample logs instead of building the table by hand (needs 0.5.0 or later; see the version note above):

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

## Difficulty from your own panel (v0.6)

**A difficulty table carries over only when your grader resembles the one that built it; a small panel under your own protocol is the safe default.** Before v0.6 we expected a large public panel to fix the single-model case. A first check on one benchmark says it does so only when the grading matches. The build is in `scripts/build_helm_nq.py`; the numbers are in `difficulty_meta.json`.

What we measured, on NaturalQuestions:
- **Within one protocol, a few model families are enough.** In HELM Lite (41 model families, 1,000 questions), difficulty from k families correlates with difficulty from a held-out half-panel:

  | k (families) | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|---|---|
  | median Spearman | 0.79 | 0.89 | 0.92 | 0.94 | 0.96 |

- **Across protocols, it depends on the grader.** On 478 questions shared with the `nq_open` table, we predicted our models' own errors (AUROC):

  | Our labels | HELM 41-family difficulty | 5 other models, our own protocol |
  |---|---|---|
  | LLM judge | 0.59–0.65 | 0.77–0.84 |
  | string match (closer to HELM's grader) | 0.71–0.80 | 0.80–0.90 |

  - Our own panel is mostly Qwen, so the right-hand column is inflated for the Qwen models.
  - On OLMo, the one non-Qwen model: judge labels 0.65 vs 0.78; string-match labels 0.79 vs 0.80.
  - The grader alone moves difficulty a lot: the same responses graded by an LLM judge vs string match give difficulties that correlate at only 0.69.
  - Exploratory: one benchmark, one non-sibling model.
  - **A second benchmark cannot come from public data.** In HELM's public results (Lite and Classic, checked 24 Nov 2026), NaturalQuestions is the only benchmark shared with a shipped table; TriviaQA, PopQA and SciQ are absent. Replicating this needs new model runs.

So the recommended path is a **small panel under your own protocol**. In Inspect, run the same task on a few cheap models, then let the adapter compute each row's difficulty from the *other* models:

```bash
# the same task on several models (one command per model); .json logs need nothing extra to read
for m in openai/gpt-4o-mini together/meta-llama/Llama-3.1-8B-Instruct-Turbo ...; do
  inspect eval my_task.py --model "$m" --log-format json
done
baseline-increment from-inspect logs/*.json --label-scorer match --score-scorer judge -o items.csv --lomo
baseline-increment check items.csv --score score --label wrong --question item --model model \
    --difficulty lomo_share_wrong --panel-meta items.csv.panel.json
```

How `--lomo` (also on `from-lm-eval`) works:
- It adds `lomo_share_wrong`, the error rate of the other models on the same item (each model's epochs averaged first), and `n_ref_models`.
- It groups rows by item, so a question never straddles folds.
- It writes the panel metadata that `check` needs to headline the difficulty increment.
- With fewer than 5 reference models, both tools say the increment is an upper bound.

**Shipped protocol-bound table: `--benchmark helm_nq`.**
- **What it is:** NaturalQuestions closed-book under HELM Lite (5-shot, quasi-exact-match), 92 runs in 41 model families, 1,000 questions, with the reliability curve above in its metadata.
- **When to use it:** when your setup is close to HELM's. `check` warns that the table is protocol-bound.
- **Before relying on it:** check the "benchmark difficulty" baseline's own AUROC against your labels.
- **Rebuild it:** `python scripts/build_helm_nq.py CACHE_DIR` rebuilds it from HELM's public results.

**Question matching** now tolerates a trailing "?" (NQ-open drops it, HELM keeps it), and it is an error, not an empty report, when nothing matches.

## Does the test itself work? (`simulations/`)

`simulations/suite.py` is a permanent calibration suite. Its acceptance rules were fixed before the first run. "Flag rate" means the rate of `adds_information = true`. The current numbers are in `simulations/results_v<version>.json` and in the table below.

Summary for the audit engine (calibrated at v0.2.2, unchanged in 0.4.0; 100 runs per scenario; full table with 95% intervals in [`CALIBRATION.md`](CALIBRATION.md); settings in [`MANIFEST.json`](MANIFEST.json)):

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

The first run failed two criteria. Details are in [`CALIBRATION.md`](CALIBRATION.md).

**History:**
- **v0.7.0:**
  - `--fast` CI preset, using Besag–Clifford permutation stopping, which passed all 18 scenarios. It stops after about 33–51 permutations on no-signal data and keeps weak-signal power at 0.81.
  - `--fail-if-no-increment` exit code, a GitHub Action (`action.yml`) and a README quickstart.
  - The half-sample interval failed its pre-registered coverage check, so the interval is now labelled as not used for the decision.
  - `__version__` now reports the true version; it said 0.5.2 in 0.6.0.
  - Requires Python 3.10 or later; CI covers 3.10–3.13.

  The decision rule is unchanged.
- **v0.6.0:**
  - `--lomo` own-panel difficulty on both adapters;
  - the protocol-bound `helm_nq` table, with its reliability curve and build script;
  - question matching tolerant to a trailing "?", and an error on zero matches;
  - the README finding that difficulty transfers poorly across protocols.

  The audit engine is unchanged.
- **v0.5.2:** documentation only. The README states the bootstrap CI's coverage FAIL under the null, the 400-run re-check, and which version each feature needs.
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

**The result in use:** at a 10% false-positive rate, metadata alone (task type, generating model, response length) catches 63.7% of hallucinated responses. Adding LettuceDetect raises that to 75.7%, **+12.0 points of recall**. In AUROC terms, +0.060 [0.047, 0.075] over metadata that alone reaches 0.853.


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

Per-item difficulty estimated from many models' public results is not new either:
- item response theory fitted to Open LLM Leaderboard results (metabench, Kipnis et al.; tinyBenchmarks, Polo et al.);
- leaderboard-derived difficulty labels (Easy2Hard-Bench, Ding et al.);
- adaptive evaluation (Fluid Benchmarking, Hofmann et al.).

Leave-one-model-out difficulty as a baseline for factual-error detectors is due to Kawiak et al. (2026). Seo et al. (arXiv 2509.15339) show that question-side effects alone reach about 0.70 AUROC.

What v0.6 adds:
- difficulty built from the user's own protocol, as the baseline for a detector increment;
- a measurement of how little such tables carry over across protocols.

License: Apache-2.0.
