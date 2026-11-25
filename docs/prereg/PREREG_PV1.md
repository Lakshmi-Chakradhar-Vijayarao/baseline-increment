# PV1 pre-registration: does a baseline check change how people validate LLM judges?

This is a product-validation study, not a statistics experiment. It runs alongside the research track; neither can stand in for the other.
- A method that is valid but changes nobody's decisions is not a product.
- A tool people like but whose statistics are invalid is not defensible.

This file is written before any interview or audit. Participants' data, names and organisations are never stored in this repository. Results are reported here only in aggregate, anonymised form.

## 1. Hypotheses

- **H0.** Researchers and evaluators who validate LLM judges against human labels get too little extra decision value from a baseline-relative analysis to change how they validate.
- **H1.** A baseline-relative analysis shows something that materially changes how some of them assess or use their LLM judge.

**Observable outcomes.** Only behaviour counts as evidence, never "this looks useful":
1. **a changed decision:** whether to rely on the judge, its rubric or prompt, how much human review is kept, or what is reported in a paper or release;
2. **an unprompted rerun:** the team runs the check again on its next alignment round, paper or model without being asked;
3. **integration:** the check is added to a pipeline, CI job, eval config or paper-reproduction script.

## 2. Participants

**Problem interviews (12).**
- 5 researchers who publish results resting on an LLM judge;
- 4 evaluation engineers at product teams that run a judge-alignment workflow;
- 3 third-party or government-network evaluators.

**Concierge audits (8).** Drawn from the interviewees who hold a real judge and a real human-labelled set.
- **Market A:** researchers publishing benchmark or judge-based results.
- **Market B:** production engineering teams.

Each audit records which market the participant belongs to, so that "A cares, B does not" can be told apart from "nobody cares".

## 3. Problem interviews

The interviews reconstruct past events. The tool is **not** mentioned until the interview's last section.

1. Tell me about the last time you validated an LLM judge.
2. What made you decide the validation was sufficient?
3. What labelled data did you use, and how many items?
4. What statistic did you report?
5. Did anyone challenge the judge? What did they ask?
6. What alternatives did you consider?
7. Did the judge ever disagree with another evaluator? What happened?
8. Did the result affect a model release, a paper, a benchmark, or a production decision?
9. **Have you ever found that your judge was measuring something other than the property you meant it to measure?** How did you find out?
10. Did you build, buy, adopt or abandon anything for this? Who maintains it?

**Recorded per interview:**
- a concrete past instance of the doubt (yes/no, with a one-line description);
- whether the participant built or wanted a tool for it;
- the market (A or B).

**Problem signal.**
- **Exists:** 6 or more of the 12 describe a concrete past instance, and 3 or more built or wanted something for it.
- **Weak:** 2 or fewer of the 12 can name an instance.

## 4. Concierge audit protocol

**Steps, in order:**
1. **Their current validation.** The participant reports, or we compute with them, their existing statistic (agreement, kappa, correlation).
2. **The baseline check.**
   - **Command:** `baseline-increment judge` on their labelled set, with baselines they choose from columns they already have, such as task, length, difficulty or source model.
   - **Who runs it:** they run it themselves, or we run it on their data with their consent.
   - **Version:** the version used is recorded.
3. **Interpretation.** "Does this change how you read your judge?" They record their answer in their own words.
4. **Decision.** "Did you or will you do anything differently?" Followed up once, after their next decision point (their next alignment round, paper revision or release), not after a fixed interval.
5. **Repeat and integration.** Observed, not asked: did they rerun it unprompted, and did it enter a pipeline?

**For every "no", the reason is recorded, classified as one of:**
- (a) agreement is enough because the decision does not need more;
- (b) a real need exists but this output does not meet it;
- (c) a workflow or integration barrier;
- (d) distrust of the method;
- (e) other.

## 5. Decision rule (fixed now)

**BUILD / CONTINUE**, all three met:
- 3 or more of 8 changed a real decision;
- 2 or more of 8 reran it unprompted;
- 1 or more of 8 integrated it.

This justifies the next product iteration (judge-vs-judge comparison, graded labels, log readers for the participants' tools).

**VALIDATE FURTHER.** Meaningful behaviour, but the full threshold is not met. Example: 3/8 changed decisions, 1/8 reran, 0/8 integrated.
- **Reading:** the problem may be real, but the workflow or output is not yet compelling.
- **Action:** one more targeted iteration aimed at the recorded "no" reasons, with no infrastructure.

**STOP THE PRODUCT TRACK**, any of:
- 0 of 8 changed a real decision;
- 5 or more of 8 give reason (a), with a real judge-validation workflow and considered alternatives;
- most participants decline to use the analysis after seeing their own result.

The research track continues regardless; a stop here is not a research failure.

**Reading by market.** If Market A meets BUILD / CONTINUE and Market B meets STOP, the outcome is recorded as an ICP finding: the product continues for Market A only.

## 6. What the study does not do

- No pricing, no sales and no paid pilots. Adoption is the current goal, not revenue.
- No new engineering beyond the `judge` command and readers for the participants' own log formats, if asked.
- Evidence packages, ledgers, dashboards and enterprise features stay frozen for the duration.

## 7. Reporting

**Shown in aggregate in this repository:**
- counts for each outcome and reason class, by market;
- anonymised one-line descriptions of the problem instances.

**Customer evidence:** none exists before this study. Nothing from it is presented as customer evidence until it has been observed.
