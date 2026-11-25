# `check`: try to break a detector's claim

One command that runs every cheap-explanation test your columns allow, plus the v0.4 error-propensity estimands.

## `check`: try to break a detector's claim (v0.3)

One command runs every cheap-explanation test your columns allow, and gives each a status:

```
baseline-increment check answers.csv --score p_true_risk --label hallucinated \
    --question question_id --model model --response answer \
    --benchmark triviaqa --question-text question --html report.html --json report.json
```

| Status | Rule |
|---|---|
| PASS | the lower bound of the validated 95% interval for the gain exceeds `--min-increment` (default 0) |
| FAIL | the upper bound is below `--min-increment`: the gain beyond this explanation is smaller than that |
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
  - PASS needs a CI lower bound > 0.5, a pair AUROC ≥ 0.55 and, when available, β > 0 with p < 0.05.
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
