# Pre-registration: where the null holds (deterministic-function diagnostic)

Written before any run of this suite. It follows `PREREG_0.12.md` addendum 3.

**Question.** For detectors that carry no information about the label beyond the baselines, because they are (functions of the baselines) + independent noise, which functional forms does each null handle? How does that change with sample size?

**Purpose.** This is a characterisation, not an acceptance test. Its output defines the documented scope of the `adds_information` claim (the validity envelope and the README), with no tuning of either null.

**Design.**
- **Baselines:** task (3 levels, assigned by group), plus L1, L2 ~ N(0, 1) at group level. There are 2 items per group.
- **Label:** y ~ Bernoulli(σ(1.2·f(B))).
- **Detector:** S = f(B) + 0.3·ε, with ε ~ N(0, 1) independent. So Y ⊥ S | B holds exactly.
- **Baselines passed to the audit:** task, L1, L2.

| ID | f(B) |
|---|---|
| D1 | L1 |
| D2 | L1 + L2 |
| D3 | L1·L2 |
| D4 | L1·L2·s(task), with s = +1 for task A and −1 otherwise |
| D5 | 2·1[L1 > 0.5] − 0.7 |
| D6 | 1[L1 > 0.5]·L2 |
| D7 | exp(L1) / 2 |
| D8 | sign(L1·L2) (XOR) |
| D9 | sin(2·L1) + L2² − 1 |
| D10 | L1 + 1[task = A]·L2 |

**Controls.**
- N0: S = ε, with f = L1 for the label.
- P1: S = ε + 0.6·y, with f = L1·L2 for the label (genuine signal).
- P2: S = L1·L2 + ε + 0.6·y, with f = L1·L2 (genuine signal alongside a baseline function).

**Runs.**
- **Each cell:** both nulls (`residual_gbm`, the default, and `residual_sl`) at both sizes (600 items in 300 groups; 3,000 items in 1,500 groups).
- **Replication:** 200 replicates per cell, run at the `--fast` settings with seed salt 21.
- **Recorded per cell:** the flag rate with its Wilson 95% interval, the mean increment, and the mean permutation p.

**How the scope will be written (fixed now).** Under the default null at each size:
- A form is **handled** if the Wilson upper bound of its flag rate is ≤ 0.10.
- It is **borderline** if only the point estimate is ≤ 0.10.
- It is **not handled** otherwise.

The README and the validity envelope will list the forms in each class, by name. A form that is not handled and has a mean increment ≥ 0.01 is described as a *model-class increment*: the detector supplies a function of the baselines that the baseline models do not learn at that sample size, so `adds_information` is not evidence of information beyond the baselines.

**No second attempt.** No function, size, null or setting changes after the first run.
