# Changelog

Newest first. Each release's calibration evidence is in [CALIBRATION.md](CALIBRATION.md).

- **v0.11.1:** layout and documentation only.
  - The README is now a short front page. The guides are in `docs/guide/`, and the history is in this file.
  - Pre-registrations moved to `docs/prereg/`.
  - Simulation outputs moved to `simulations/results/`; the shipped calibration summary cites the new paths.
  - The legacy v0.4 report generator is guarded so it cannot overwrite `CALIBRATION.md`.
  - `CALIBRATION.md` has an index.
  - Package metadata now has keywords and classifiers.
  - No code behaviour changed.
- **v0.11.0:**
  - `--outcome ordinal`: graded labels, concordance, same null and rule.
  - `--calibration`: raw-score slope, calibration-in-the-large, Brier and a reliability table, with group-bootstrap intervals.
  - Both passed their pre-registered checks.
  - Binary commands reject labels other than 0/1. This fixes silent mishandling of graded labels in all earlier versions.
  - The binary engine is unchanged, guarded by a golden-output test.
- **v0.10.0:**
  - `--report` (HTML or Markdown) on audit and `compare`: verdict, operating points, calibration evidence matched to your settings, what is not tested, provenance and sign-off.
  - Ships `data/calibration_summary.json`, built by `scripts/build_calibration_summary.py`.
  - The GitHub Action takes `report`.
  - The audit engine is unchanged.
- **v0.9.0** (changes results for multi-model Inspect logs that do not use `--lomo`):
  - **Inspect adapter:** a native `.eval` reader, verified against `inspect_ai`, and directory input with `--list`.
  - **Grouping:** item-level grouping across models and epochs, with a guard that ids name the same question in every log.
  - **Length:** input and output length counts message text only. This fixes an inflated count for chat-style inputs in all earlier versions.
  - **Scorers:** `name:key` for dict-valued scorers, plus `--partial-as` / `--noanswer-as`.
  - **Dropped rows:** counted by reason.
  - **Other data:** a generic reader for CSV, TSV, Parquet and JSON with a `--map` mapping file.
  - **Unchanged:** the audit engine.
- **v0.8.0:**
  - `compare` for detector releases: improvement, regression (`--fail-on-regression`) and standalone AUROC.
  - All pre-registered criteria are met.
  - The audit engine is unchanged.
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
