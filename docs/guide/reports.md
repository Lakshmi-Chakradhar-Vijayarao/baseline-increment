# Audit reports



## Audit reports (v0.10)

Add `--report audit.html` (or `audit.md`) to an audit or a `compare` run to get a document that can be filed and signed off:

```bash
baseline-increment items.csv --score score --label wrong --group group --baselines task,log_output_words \
    --report audit.html
```

**What it contains:**
- **Verdict:** fixed sentences tied to the decision rule, with the numbers that decided it. Nothing is generated.
- **Key numbers:** detector, baselines, increment and p-value. The interval is labelled as not used for the decision.
- **What it means in use:** recall at 5% and 10% false-positive rates, for baselines alone and for baselines plus detector, measured out of fold.
- **Every baseline family,** and every stratum you asked for.
- **Calibration evidence for the settings you actually ran:**
  - **If your settings match a calibrated run** (the default settings, with all three families), the report quotes that run's measured false-alarm and detection rates and the commit they came from.
  - **If they don't match,** it says so plainly instead of borrowing numbers that were not measured at your settings.
- **What it does not test:** leakage, out-of-distribution behaviour, label quality, downstream usefulness, and baselines you did not supply.
- **Provenance:**
  - the input file's SHA-256 and row count, plus the mapping file's hash;
  - the exact command, with paths reduced to file names so shared reports don't expose home directories;
  - versions, settings and a UTC timestamp.
- **A sign-off line.**

**Format:** the HTML is self-contained, with no external assets, and all text from the data is escaped. It has print styles, so "Print → Save as PDF" gives a clean document. The GitHub Action takes `report: audit.html`.
