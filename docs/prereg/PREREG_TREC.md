# Pre-registration: does stratifying the null within query fix the TREC false alarms?

Written before any run of this study. It follows `PREREG_JUDGES.md`: the J3 calibration control failed there (33 of 100 runs flagged, Wilson interval [0.246, 0.427]).

**Question.** In the TREC LLMJudge structure, the baselines are a 25-level categorical (query) and a continuous one (log passage words). Does permuting the null's residuals only *within query* bring the false-flag rate of a no-information judge back to ≤ 0.10, while keeping power?

**Data and engine.**
- **Data:** `validation/judges/data/j3_llmjudge.csv`, as built for PREREG_JUDGES.
- **Outcome:** ordinal NIST relevance 0–3.
- **Baselines:** query and log passage words; the group is the passage.
- **Engine:** the default null `residual_gbm`, at the `--fast` settings.
- **The no-information judge:** exactly as in the failed control. `NISTRetrieval-instruct0` is permuted within query × passage-length-decile strata, with seeds 880000 + rep, rep = 0–99.

**Arms** (100 replicates each):

| Arm | Null permutation strata | Role |
|---|---|---|
| A0 | none (the default) | sanity check: must reproduce 0.33 |
| **A1** | query | **the candidate**: users can declare this whenever a high-cardinality categorical baseline exists |
| A2 | query × passage-length decile | descriptive only: it matches the control's own construction, which a user cannot know in general |
| P1 | query (as A1) | power: the A0 judge plus 0.6·SD·(y − mean y), reported |

All 33 J3 judges are also re-audited under A1, for reporting only.

**Decision for A1, fixed now.**
- **Pass:** the Wilson 95% upper bound ≤ 0.10 and P1 power ≥ 0.80. Within-query stratification then becomes the documented recommendation for data with a high-cardinality categorical baseline, and the J3 results under A1 are reported as calibrated *for this structure*.
- **Borderline:** only the point estimate ≤ 0.10. This is documented as partial.
- **Fail:** otherwise. The documentation then says stratification does not fix this structure, and the conditional-information verdict is unavailable for such data.

**Scope.** One real dataset. Nothing here is claimed for other structures without its own check.

**No second attempt.** No arm, seed, judge or setting changes after the first run.
