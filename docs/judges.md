# LLM judges against cheap baselines, on human-labelled data

Pre-registered in [`prereg/PREREG_JUDGES.md`](prereg/PREREG_JUDGES.md) (with addendum 1), and run once, on Kaggle CPU at commit 63c5593 with pinned dependencies. The results are in [`validation/judges/results.json`](../validation/judges/results.json). The table builder and runner are in [`validation/judges/`](../validation/judges/).

**Reading the numbers.**
- **Gains are model-class predictive gains:** the improvement in held-out prediction of the *human* label when the judge's score is added to the declared baselines, under the declared model families.
- **They are not proof of information beyond the baselines.** See the README's list of what does not work.

## The headline: cheap baselines are a hard bar for LLM judges

| Dataset | Human label | Cheap baselines | Baselines alone | Judge raw agreement | Judge's gain beyond baselines |
|---|---|---|---|---|---|
| SummEval + G-Eval (GPT-4), 1,600 summaries | expert 1–5 ratings | system, length, document difficulty | 0.76–0.85 | 0.75–0.89 | +0.04 to +0.06 on all four dimensions (all significant after Holm) |
| FaithBench (stress set; items chosen where detectors disagreed), 750 summaries | expert hallucination spans | summarizer, length, source difficulty | 0.78 | 0.49–0.60 | GPT-4o +0.025, HHEM-2.1 +0.020, GPT-4-turbo +0.011; GPT-3.5-turbo and TRUE-NLI nothing detectable |
| TREC DL 2023 LLMJudge, 4,423 query–passage pairs, 33 judges | NIST relevance 0–3 | query, passage length | 0.69 | 0.57–0.72 | +0.01 to +0.11 (uncalibrated: see below) |

On SummEval coherence the baselines alone (0.764) beat the judge's raw agreement (0.752). On FaithBench every detector's raw agreement is far below what source difficulty and the summarizer's identity give.

**Do rankings change?** In TREC, ranking the 33 judges by gain beyond the baselines instead of by raw agreement gives Kendall's τ = 0.795. That is just under the pre-registered 0.80, so "rankings differ materially" is supported, narrowly. 7 judges move 5 or more places; the top judge (Olz-gpt4o) is the same either way.

## What went wrong, reported as found

1. **The TREC calibration control failed.**
   - **The control:** a judge whose scores were permuted within query × length-decile strata, so it carries no information beyond the baselines.
   - **The result:** the conditional-information test flagged it in 33 of 100 runs (Wilson [0.25, 0.43]).
   - **The consequence:** under the pre-registered rule, the TREC gains are reported as uncalibrated for that claim.
   - **The fix tried:** a follow-up study ([`prereg/PREREG_TREC.md`](prereg/PREREG_TREC.md)) tested stratifying the permutation by query. It **failed** at 0.29. Only stratifying by the control's own query × length construction brought it to 0.07, and users cannot know that construction. The cause is that the effect of length differs by query, the same structure as diagnostic case D10.
2. **A design error: too few permutations for 33 tests.** With at most 200 permutations, the smallest p is 0.005, so a Holm correction across 33 judges could never reach 0.05. The "0 after Holm" count for TREC is therefore an artefact of the design. Future studies size permutations to the number of tests; this one is not re-run, because changing it after seeing results would be tuning.

## Secondary check: WQA judges against rule labels

On WQA PopQA and TriviaQA ([`prereg/PREREG_0.12.md`](prereg/PREREG_0.12.md) Part C), Falcon3-10B and GLM-4-9B judges added +0.065 (PopQA) and +0.12 (TriviaQA) beyond question difficulty and length. Their raw AUROC is 0.98–0.99, because the judges grade correctness and nearly reproduce the rule labels. The no-information controls there passed (0.04–0.06).
