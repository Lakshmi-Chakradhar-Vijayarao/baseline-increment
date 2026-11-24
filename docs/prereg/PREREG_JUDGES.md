# Pre-registration: do LLM judges add information beyond cheap baselines on human-labelled data?

Written and committed before any audit is run on these data. The engine is the released default (null `residual_gbm`; the 0.12 super-learner null was not adopted). Settings are `--fast`: 2 seeds, at most 200 permutations, stopping at 10, all three families.

## Questions

- **Q1.** For each LLM judge on three public, human-labelled datasets, does the judge's score add information about the *human* label beyond cheap baselines (generating system, response length, item or source difficulty, query)?
- **Q2.** Does ranking judges by their increment over the baselines order them differently from ranking them by raw agreement with humans?
- **Q3 (descriptive).** How much of each judge's raw agreement can the baselines alone reproduce?

## What has been looked at

The datasets were downloaded and their file structures read in order to map columns. One thing beyond structure was computed:
- **FaithBench:** the mean of each detector's score among summaries with and without an "Unwanted" annotation, to check orientation.
- **What it shows:** every detector's mean is lower on hallucinated summaries, except GPT-3.5-turbo's. That reveals raw direction, and indirectly raw separation, but nothing about any increment over baselines.
- **Orientation still follows the documentation:** every FaithBench detector output is a consistency score, so its risk is 1 − score.

No other statistic involving a label and a judge has been computed.

## Data

Tables are built by `validation/judges/build_judge_tables.py` from the public sources. Only per-item ids, labels, baselines and scores are kept, never text. The FaithBench table is not committed (CC BY-NC-SA); the builder recreates it.

| Set | Source and licence | Items | Human label (outcome) | Judges (detectors) | Baselines | Group |
|---|---|---|---|---|---|---|
| **J1** SummEval + G-Eval | nlpyang/geval, MIT | 1,600 summaries (100 CNN/DM documents × 16 systems) | **Primary:** consistency, the mean of 3 expert 1–5 ratings rounded to the nearest integer (ordinal). **Secondary:** coherence, fluency, relevance | G-Eval GPT-4, as the mean of its 20 sampled ratings, each parsed as the first number 1–5. An item with fewer than 10 parsable ratings is dropped, and the count is reported. | system id; log summary words; D_doc, the mean human score of the other 15 systems' summaries of the same document | document |
| **J2** FaithBench (stress set) | vectara/FaithBench, CC BY-NC-SA 4.0 | 750 summaries | **Primary:** hallucinated = 1 if any annotation is Unwanted (worst-label aggregation, as in `binarize.py`). **Secondary:** Unwanted or Questionable | HHEM-2.1 (continuous; a non-LLM reference); GPT-3.5-turbo, GPT-4-turbo and GPT-4o verdicts (binary); TrueTeacher and TRUE-NLI (binary; reported) | summarizer; log summary words; D_src, the hallucination rate of the other summarizers' summaries of the same source | source passage |
| **J3** LLMJudge, TREC DL 2023 | llm4eval/LLMJudge(-benchmark), MIT | 4,423 query–passage pairs, 25 queries | NIST relevance 0–3 (ordinal) | all 33 released judge runs (0–3 verdicts) | query (categorical); log passage words | passage id |

**J2 caveat, fixed now:** FaithBench samples were chosen where detectors disagreed or found items hard. Its results are a stress test and are never pooled with J1 or J3.

## Analysis

- **Audits.** One audit per (set, judge, outcome), at the settings above:
  - ordinal outcome for J1 and J3;
  - binary outcome for J2.

  Each audit records the increment, p, `adds_information`, the validity-envelope status, and the judge's raw concordance (AUROC on binary labels).
- **Multiplicity.** Within each set, the count of judges that add information is reported raw and after Holm adjustment over that set's judges (primary outcome only).
- **Q2.** In J3 (33 judges), compute Kendall's τ between the judges' raw concordance and their increments over the baselines. Also report the number of judges whose rank moves by 5 or more places, and whether the judge ranked top by raw concordance is also ranked top by increment.
- **Q3.** Report each judge's raw concordance next to the baselines-only concordance.

## Criteria (fixed now)

- **H1 (Q1):** reported per set as "k of m judges add information (Holm)". There is no pass/fail; this is a measurement.
- **H2 (Q2):** "ranking by increment differs materially from ranking by raw agreement" is **supported** if τ < 0.80, and **not supported** if τ ≥ 0.80. Either outcome is reported.
- **Calibration control (must pass for the J3 results to be reported as calibrated):** in J3, one fixed judge (the first file alphabetically, `NISTRetrieval-instruct0`) has its scores permuted within query × passage-length-decile strata, 100 times. The false-flag rate must be ≤ 0.10. If it fails, the J3 increments are reported as uncalibrated.

## No second attempt

No dataset, label definition, baseline, judge set, threshold or setting changes after the first audit under this registration.
