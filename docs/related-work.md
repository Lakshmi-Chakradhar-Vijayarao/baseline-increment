# Related work



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

The statistical core has close relatives, which the package does not claim to have invented:
- **Conditional predictive impact** (Watson & Wright, *Machine Learning* 2021) tests whether a feature improves out-of-sample loss beyond the other covariates, using knockoffs, and gives p-values.
- **The conditional randomisation test** (Candès et al. 2018) and leave-one-covariate-out inference.
- **The residual null used here (0.2.2 on)** approximates a conditional randomisation: the detector is replaced by its predicted value given the baselines plus permuted residuals. From 0.12, the score model is a small super learner (van der Laan et al. 2007).
- **Its validity depends on how well that score model fits.** This is the known sensitivity of conditional randomisation to the conditional model (Berrett et al. 2020), and the reason for the 0.12 change.

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
