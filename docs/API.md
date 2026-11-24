# Stability policy (from 1.0)

From 1.0 the package follows semantic versioning. **A minor or patch release never changes a verdict that an earlier release in the same major version would give on the same data and settings.** A change to the decision engine (the folds, the families, the null, the decision rule) is a major release, and its calibration must be re-run and published before release (see [prereg/](prereg/)).

## Stable surface

| Kind | Stable names |
|---|---|
| Python functions | `audit`, `compare`, `check`, `auc` (top level); `baseline_increment.ordinal.audit_ordinal`; `baseline_increment.calibration.score_calibration`; `baseline_increment.adapters.from_inspect`, `from_lm_eval`, `list_inspect`, `add_lomo_difficulty`; `baseline_increment.tables.read_table` |
| Their arguments | names and defaults as documented in the guides; new arguments are added with defaults that keep old behaviour |
| CLI | the commands `baseline-increment`, `check`, `compare`, `from-inspect`, `from-lm-eval`; every documented flag; exit codes: 0 success; 1 a gate failed (`--fail-if-no-increment`, `--fail-on-regression`); 2 a usage or input error (from 1.0; before 1.0 input errors exited 1) |
| Audit JSON (`--json`) | the keys checked by `tests/test_api_contract.py`: `version`, `n_items`, `n_groups`, `settings`, `detector_auc`, `baseline_auc`, `families`, `headline_family`, `baselines_combined_auc`, `baselines_plus_detector_auc`, `increment`, `increment_ci`, `increment_perm_p`, `adds_information`, `warnings` |
| Reports | the sections listed in [guide/reports.md](guide/reports.md); wording may improve, sections are not removed |
| Shipped tables | file names and columns in `baseline_increment/data/`; contents may be extended |

New keys may appear in the JSON at any release. Code that reads it should ignore keys it does not know.

## Not stable

- Names starting with `_`, and every module-level helper not listed above.
- `simulations/`, `validation/`, `scripts/` (research code; kept reproducible, not an API).
- Timing and the exact text of warnings.

## Reproducibility

Every random step has a fixed seed, so the same version, data, settings and pinned dependency versions (`requirements-lock.txt`) give the same numbers, whatever `--jobs` is. Different numpy or scikit-learn releases can change floating-point results. `tests/test_golden.py` fixes the binary engine's output to 1e-12, and `tests/test_vectorised_equivalence.py` checks the fast paths against the reference loops.

## Offline use

The package never opens a network connection: no telemetry, no downloads, no update checks (`tests/test_offline.py` runs the CLI with sockets disabled). Benchmark tables ship inside the wheel. It can run on an air-gapped machine.

## Verifying a release

Each GitHub release carries the wheel, the sdist, `SHA256SUMS.txt`, a CycloneDX SBOM (`sbom.cdx.json`) and a Sigstore build-provenance attestation:

```bash
gh attestation verify baseline_increment-1.0.0-py3-none-any.whl --repo Lakshmi-Chakradhar-Vijayarao/baseline-increment
```
