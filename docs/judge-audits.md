# What do LLM judges add beyond cheap signals? Two public audits

**The question.** A judge's agreement with human labels says how often it matches them. It does not say whether the judge knows anything that cheap, judge-free signals do not already give: the query or task, the response length, the model that produced the response. These audits ask exactly that.

**What was measured.** For each judge, the audit reports:
- the held-out improvement in predicting the human label when the judge is added to those signals (the *gain*, in AUROC);
- the gain's 95% interval and a verdict: PASS if the whole interval is above zero, FAIL if the whole interval is below zero, and INCONCLUSIVE otherwise.

**How.** The command is `baseline-increment judge` (validated preset: grouped 5-fold cross-validation, 2 seeds, three model families, clustered DeLong interval). The baselines named here use **no human labels**.

## TREC 2024 LLMJudge: 33 relevance judges

- **Data:** the TREC assessors' relevance labels, binarised as relevant = 2 or 3, the TREC convention. Grouped by passage.
- **Baselines:** the query (25 levels) and the passage length.

| | |
|---|---|
| Query + length alone, held-out AUROC | **0.734** |
| Judges whose own AUROC is at or below that | **22 of 33** (range 0.591–0.770) |
| Judges with a positive gain beyond query + length (PASS) | **33 of 33** |
| Gain: median (range) | **+0.080** (+0.011 to +0.115) |
| Largest | Olz-gpt4o +0.115 [+0.101, +0.128]; willia-umbrela1 +0.112; h2oloo-zeroshot1 +0.107 |
| Smallest | TREMA-nuggets +0.011 [+0.006, +0.016]; TREMA-rubric0 +0.031; TREMA-questions +0.035 |

**Reading.** On their own ranking quality, two thirds of the judges look no better than knowing the query and the passage length. All of them still carry held-out signal that those two do not, so raw agreement understates them. The gain also separates them more sharply than agreement does.

## FaithBench: 6 hallucination detectors

- **Data:** expert hallucination labels on 750 summaries, grouped by source article. FaithBench is a stress set, built from summaries where detectors disagreed.
- **Baselines:** which model wrote the summary, and the summary's length.

| Detector | Agreement with annotators | Alone (AUROC) | Gain beyond baselines [95% interval] | Verdict |
|---|---|---|---|---|
| HHEM-2.1 | (continuous score) | 0.597 | **+0.045** [+0.006, +0.085] | PASS |
| GPT-4o | 0.497 | 0.562 | +0.025 [−0.009, +0.059] | INCONCLUSIVE |
| GPT-4-turbo | 0.505 | 0.561 | +0.014 [−0.020, +0.047] | INCONCLUSIVE |
| TrueTeacher | 0.459 | 0.522 | +0.006 [−0.007, +0.020] | INCONCLUSIVE |
| TRUE-NLI (2 unscored rows left out) | 0.421 | 0.503 | −0.004 [−0.007, −0.000] | FAIL |
| GPT-3.5-turbo | 0.444 | 0.486 | −0.009 [−0.016, −0.002] | FAIL |

The two baselines alone reach an AUROC of 0.593.

**Reading.** Only the dedicated detector clearly adds to the two cheap signals. The general-purpose LLM judges are not distinguishable from them here, and two make held-out prediction slightly worse.

**Sensitivity check.** Adding a label-derived source-difficulty baseline (the share of the *other* summaries of the same article marked hallucinated) is reported in `validation/judges/judge_audits_results.json`.

## Reproduce

```bash
pip install "git+https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment"
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment && cd baseline-increment
# FaithBench table: rebuilt locally from vectara/FaithBench (CC BY-NC-SA), see validation/judges/build_judge_tables.py
python validation/judges/audit_public_judges.py            # writes validation/judges/judge_audits_results.json
```

One detector by hand:

```bash
baseline-increment judge validation/judges/data/j2_faithbench.csv --judge hhem_2_1_risk \
    --label hallucinated --baselines summarizer,log_words --group source_id
```

## What this does and does not show

- **The gain is predictive, not causal.** It is relative to the declared baselines and the model families used. Other baselines can give other answers; that is why they are named.
- **"Alone" is the judge's raw AUROC.** The baselines' AUROC is held out. The gain, which is what the verdict rests on, is held out on both sides.
- **The interval's validation:**
  - Its coverage was validated in six pre-registered scenarios (0.905–0.965).
  - A wider stress test (R1, `docs/prereg/PREREG_R1.md`) is in progress. So far one structure under-covers: a three-way interaction, where the gain comes from a flexible model partly learning the interaction (coverage 0.695).
  - A refit-aware interval for that case is pre-registered (`PREREG_R1b.md`).
- **Two datasets are evidence about these two datasets.** They do not show that agreement misleads in general.
