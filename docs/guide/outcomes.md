# Graded labels and probability quality



## Graded labels and probability quality (v0.11)

**Graded labels** (`--outcome ordinal`):
- **What it takes:** severity levels such as 0–3 (3 to 10 integer levels).
- **The question:** does the detector add information about *how bad* an item is, beyond the baselines?
- **The metric:** concordance. Among pairs of items with different severity, it is the share where the more severe item has the higher score. On 0/1 labels it is exactly AUROC.
- **Same machinery:** the same baseline families, grouped cross-validation, null and decision rule as the binary audit.
- **Calibration:** pre-registered in [`docs/PREREG_0.11.md`](../../docs/prereg/PREREG_0.11.md). False flags were at most 0.08 across five no-signal scenarios, two of them adversarial. Planted-signal detection was 1.00.
- **Where it works:** the main audit command and the report.
- **Where it doesn't:** `check`, `compare` and recall-at-FPR refuse graded labels instead of computing yes/no numbers on them.

```bash
baseline-increment items.csv --score risk --label severity --group group --baselines task,log_output_words \
    --outcome ordinal --fast --report audit.html
```

**Probability quality** (`--calibration`, binary labels, score in [0, 1]):
- **What it answers:** when a score is used *as a probability*, is it one?
- **What it reports:**
  - the calibration slope: 1 is ideal, below 1 is overconfident;
  - calibration-in-the-large: 0 is ideal, above 0 means the score under-predicts;
  - the Brier score;
  - a 10-bin reliability table;
  - ECE, labelled as biased upward.
- **Intervals:** each comes from a group bootstrap. They are descriptive, with no verdict.
- **Interval coverage:** on simulated scores that are calibrated by construction (clustered like real data), the intervals covered the truth 0.955 (slope) and 0.945 (calibration-in-the-large) of the time. On a deliberately overconfident score, the slope interval flagged it in 91% of runs.

**Binary labels are now checked.** Every binary command rejects labels other than 0/1, with an error that points to `--outcome ordinal`. Before 0.11, a 0–3 severity column given as a binary label was silently mishandled.
