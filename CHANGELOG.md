# Changelog

Newest first. Each release's calibration evidence is in [CALIBRATION.md](CALIBRATION.md).

- **Unreleased:**
  - **New: `baseline-increment judge`.** A thin wrapper for judge validation. It takes the labelled set used to align an LLM judge and reports judge-human agreement and Cohen's kappa, the AUROC of the baselines alone, and the judge's held-out gain with the validated interval and verdict. `--response` adds the response length as a baseline. The statistics are unchanged; the conditional-information test is not run in this mode.
  - **Registered:** `PREREG_R1b.md` (a refit-aware interval, after R1's D4 failure) and `PREREG_PV1.md` (a product-validation study: problem interviews and concierge audits).
- **v0.13.0** (research release; changes the primary output and the CI gate):
  - **Primary result:** the held-out gain beyond the declared baselines with a 95% clustered DeLong interval. Its coverage was validated at 0.905–0.965 in six pre-registered scenarios (`PREREG_0.12.md` Part B).
  - **Verdict:** PASS / INCONCLUSIVE / FAIL against a declared minimum gain (`--min-gain`, policy `minimum_gain`; default 0).
  - **The gate uses it:** `--fail-if-no-increment`, `compare --fail-on-regression` and the GitHub Action now use this verdict.
  - **`adds_information` (the permutation test of conditional independence) is now secondary and experimental, and drives nothing.** The pre-registered diagnostic and the TREC study found false flags of 26–58% when a baseline's effect differs by category, 33% on real TREC data, and 75–100% for XOR-like and three-way functions of the baselines (`PREREG_DIAG.md`, `PREREG_TREC.md`, `docs/judges.md`).
  - **Defaults are now the validated preset**, for both the CLI and the Python API: 2 seeds, up to 200 permutations stopping at 10, the DeLong interval, all cores in the CLI. `--fast` is kept as a no-op alias. Legacy options are hidden from CLI help and kept in the API.
  - **`check` statuses** are PASS / INCONCLUSIVE / FAIL from the interval. The old rule (≥ 0.02 plus a refit-bootstrap CI plus Holm p) is retired.
  - **Graded (ordinal) labels:** the gain is descriptive and the CI gate refuses them, because they have no validated interval yet.
  - **New, validated:** `--sensitivity`, the omitted-baseline sensitivity analysis (`PREREG_0.13.md`; no false comfort in 98.0%).
  - **New, under validation:** `plan`, a simulation-based sample-size planner.
  - **New assurance features:**
    - `--policy` and `declare`;
    - `--evidence-dir` with a hash-chained run ledger and `evidence-record.json`;
    - `verify-evidence --expect-head`;
    - the validity envelope (not testable / outside / within the validated range);
    - the claim ledger with executable claim tests.
  - **New report additions:** a three-part report (summary, engineers, auditors), `--ladder`, AUPRC, operating points per 1,000 positives, `--text` duplicate check, and the statistical preset recorded in results and evidence.
  - **Exact-output speedups:** reshuffling at 100,000 rows went from 15 s to 0.02 s.
  - **Not adopted:** the 0.12 super-learner null, which failed held-out cases (`PREREG_0.12.md` addenda 2–3).
- **v0.11.2:** validation and documentation only; no code change.
  - **Real-data validation** (`docs/validation.md`, `validation/`):
    - seven detectors on RAGTruth, TofuEval and TriviaQA agree with independent analyses, every increment within 0.006 AUROC;
    - false alarms on real data structures were 0.053 pooled over 600 runs.
  - **Stress test** (`simulations/stress_suite.py`): 10 of 11 nulls pass. The interaction-only case is borderline at 0.095, and the README now flags it.
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
