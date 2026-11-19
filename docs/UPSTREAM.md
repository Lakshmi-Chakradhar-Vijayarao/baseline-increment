# Upstream contribution plan (Part C). Drafted 18 Nov 2026; nothing has been opened or sent.

## What the harnesses' maintainers have already said

**lm-evaluation-harness**: paired comparison is taken.
- Issue #3831, "Paired significance test for comparing two runs", is open (updated 10 Nov 2026).
- PR #3835 (McNemar + paired bootstrap) and PR #3868 (paired McNemar) are both open and competing.
- #4017 proposes Wilson CIs, MDE and BH-FDR on results output.

A third implementation would add noise to work already under review. **Do not open another one.** Useful moves, if any:
- review #3835 or #3868 with a test case;
- or fix a confirmed bug. #4288 (bootstrap_stderr drops the remainder chunk) already has three open PRs (#4293, #4298, #4625), so it is taken too.

**Inspect AI**: paired statistics belong outside core, by the maintainers' decision.
- Proposal #4206 and PR #4260 (paired deltas, cluster/McNemar, multiplicity, power) were closed on 11 Sep 2026. The maintainer wrote: *"we think this would be useful as a standalone extension rather than part of Inspect core."*
- That engine now exists as a standalone library: `stephlinds/evalconfidence`.
- Analysis tools reach Inspect users through the **extensions listing**, `docs/extensions/extensions.yml`, category "Analysis". These are small PRs that merge: "ARCI verdict" was merged on 17 Nov 2026.

## Proposal

**1. Inspect extensions listing (recommended first step).** Open one PR adding baseline-increment under "Analysis". Draft entry:

```yaml
- name: "[baseline-increment](https://github.com/Lakshmi-Chakradhar-Vijayarao/baseline-increment)"
  description: |
    Does a detector, judge or confidence score add information beyond cheap baselines (length, task,
    model, difficulty)? Reads Inspect logs (`baseline-increment from-inspect`), takes the label from one
    scorer and the detector from another, and runs a refit conditional-randomisation test whose
    false-flag rate is published from a simulation suite.
  author: "[Lakshmi Chakradhar Vijayarao](https://github.com/Lakshmi-Chakradhar-Vijayarao)"
  categories: ["Analysis"]
```

**Preconditions, all before opening:**
- `from-inspect` is released on PyPI (0.5.0);
- the README's Inspect section has a worked example;
- the entry's description is checked against the README.

**2. lm-evaluation-harness.** No new PR. If useful, leave one review on #3835 or #3868: a reproducible paired case from the pythia fixtures, and the clustered-items point (several questions from one document are not independent).

## Prior work to cite in any proposal

- Miller (2024), *Adding Error Bars to Evals*, Anthropic.
- Candès et al. (2018), conditional randomisation test.
- Pepe et al. (2013) and Demler et al. (2012), on incremental AUROC and why DeLong's test does not apply to nested models.

## Who does what

The user opens every PR and issue and answers maintainers. This file is a draft for them.

## Status (19 Nov 2026)

- **Inspect extensions listing:** PR [UKGovernmentBEIS/inspect_ai#6664](https://github.com/UKGovernmentBEIS/inspect_ai/pull/6664), opened 19 Nov, awaiting review.
- **PyPI:** 0.5.2 is live, including the adapters.
