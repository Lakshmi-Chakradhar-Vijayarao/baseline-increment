# Assurance: declared policies, evidence packages and the run ledger

This layer makes a result defensible later. It does not change any statistic or verdict. The same data and settings give the same numbers with or without it.

The results it records answer three questions:
1. What exactly was tested?
2. Was the test declared before the result was known, or chosen after trying several?
3. Can someone else check, months later, that nothing was edited?

## 1. Declare the evaluation: `--policy`

```json
{
  "version": 1,
  "description": "Hallucination detector acceptance, RAG support bot",
  "owner": "evaluation team",
  "detector": {"name": "LettuceDetect v0.1", "score": "lettucedetect"},
  "label": "hallucinated",
  "group": "source_id",
  "baselines": ["d_src", "task_type", "log_answer_words"],
  "operating_points": [0.05, 0.10]
}
```

```bash
baseline-increment items.csv --policy policy.json --evidence-dir evidence/
```

**What the policy fixes:**
- the score, label, group and baselines;
- the outcome type, `binary` or `ordinal`;
- the preset: `fast` is the default and is calibrated for CI, and `full` adds refit replicates;
- the operating points.

The policy is the full statement of the analysis, so command-line flags that would change the analysis are refused rather than merged. These are `--score`, `--baselines`, `--families`, `--null` and the seeds and permutation settings.

**Only calibrated rules are accepted.** A policy that asks for something this version has not calibrated is refused, naming the rule and the reason:

| Key | Status |
|---|---|
| `minimum_gain` | **accepted from 0.13** (binary outcome): PASS needs the lower 95% bound of the gain to exceed it; the interval is validated |
| `minimum_operational_gain` | refused: not calibrated |
| `auprc` | refused: not calibrated |
| `alpha` other than 0.05 | refused: not calibrated |
| `omitted_baseline_bound`, `heterogeneity_test`, `sample_size` | refused: not implemented yet |

Unknown keys are refused too, so a misspelt `baseline:` cannot silently drop every baseline.

**Two hashes.**
- **The policy file hash** is the SHA-256 of the policy in canonical JSON, where key order and whitespace do not matter. It identifies the file you declared.
- **The analysis specification hash** covers only what changes the analysis:
  - the columns;
  - the baselines as a set;
  - the outcome;
  - the resolved settings (families, null, seeds, permutations).

  Editing a policy's `description`, `owner` or detector name, or running the same analysis once from a policy and once from flags, leaves it unchanged. The ledger counts specifications by this hash.

**Declare before running.** To make "declared before the first run" provable rather than asserted, record the policy in the ledger first:

```bash
baseline-increment declare policy.json --evidence-dir evidence/ --data items.csv
baseline-increment items.csv --policy policy.json --evidence-dir evidence/
```

The report's audit record then shows "Declared in the ledger before the first run: yes, at <time> (ledger line 1); declared for this exact input". If the policy was used before it was declared, or never declared, the report says so.

## 2. The evidence package: `--evidence-dir DIR`

Each run writes a self-contained folder `DIR/<UTC time>-<specification hash>/`:

| File | Contents |
|---|---|
| `report.html` | the report: summary, engineer's numbers, auditor's record |
| `results.json` | the full machine-readable result |
| `specification.json` | the policy (or command-line specification) and its hash |
| `provenance.json` | the input file's SHA-256, row count, command, versions and time |
| `environment.json` | Python, platform and the installed version of every runtime dependency |
| `command.txt` | the command, with paths reduced to file names |
| `evidence-record.json` | the one machine-readable record of the run: the claim, its numbers, hashes, the command and what the result does not establish ([schema](../../baseline_increment/data/evidence_record.schema.json), version 1) |
| `ledger.jsonl` | the run ledger up to and including this run |
| `MANIFEST.sha256` | SHA-256 of every file above, in `sha256sum` format |

To check a package with standard tools, run `cd <package> && sha256sum -c MANIFEST.sha256`.

## 3. The run ledger

`DIR/run-ledger.jsonl` gets one line per run, and earlier lines are never rewritten. Each line carries:
- the run number and time;
- the input hash, the analysis specification hash and the policy file hash;
- the verdict;
- the SHA-256 of the previous line.

The chain makes deleting, editing or reordering an earlier run detectable:

```bash
baseline-increment verify-evidence evidence/
# ledger: 7 runs checked
# 20261124T101500Z-a913889e: intact
# ...
# OK: ledger chain intact and every package matches its manifest
```

The command exits 1 if anything fails.

**What a consistent chain proves.** It shows that no entry was edited, removed or reordered relative to the others. Someone with write access could still rebuild the whole directory from scratch. To rule that out, record the ledger's head hash somewhere outside the directory, such as a ticket, a signed email or a commit message: `verify-evidence` prints it. Later, check against it:

```bash
baseline-increment verify-evidence evidence/ --expect-head 3f2a...c9
```

**What the report says from the ledger:**
- **Its place in the history:** "run 3 of 3 recorded on this input", and how many distinct specifications were tested on the same input.
- **If earlier runs used a different specification,** it says plainly that this is not a first, pre-declared test.
- **If several specifications were tested,** it also shows the p-value counted over all of them: Bonferroni, min(1, k × p). This figure is for the reader. The verdict itself follows the pre-declared rule for this run.

The ledger does not forbid exploration. It stops an exploratory run from being presented as the pre-declared one.

**What it cannot do.** The ledger records runs made with `--evidence-dir` into that directory. It cannot know about runs made elsewhere. A team that wants the guarantee keeps one evidence directory per evaluation, under version control or write-once storage, and makes every run into it.

## 4. Three readers, one result

Reports are now in three parts, all built from the same numbers:
- **Summary:** one fixed sentence carrying the verdict on the gain: "PASS: … improves held-out prediction beyond the declared baselines (…) by +x AUROC (95% CI …)", or INCONCLUSIVE ("… not shown … This is not evidence that the gain is zero"), or FAIL. The experimental conditional-information test is stated on a separate line and never in the summary.
- **For engineers:**
  - key numbers;
  - recall at fixed false-positive rates;
  - every baseline family;
  - strata and warnings.

  For `compare`, this part has a v1/v2 table and the three questions answered separately: improvement, regression and standalone.
- **For auditors:**
  - the audit record: specification hash, policy file, run position, specification count, every setting, the statistical preset and the decision rule;
  - the calibration evidence for these settings;
  - what was not tested, and what the result does not establish (causal validity, absence of omitted baselines, generalisation, label validity, calibration, usefulness after deployment);
  - provenance;
  - a sign-off line.
