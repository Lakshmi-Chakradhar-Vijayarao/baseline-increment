# Getting your data in

Harness logs (lm-evaluation-harness, Inspect), any tabular export, and difficulty from your own panel of models.

## From your evaluation harness

If your evaluations already run in **lm-evaluation-harness** or **Inspect**, convert their per-sample logs instead of building the table by hand (needs 0.5.0 or later; see the version note above):

```bash
# lm-eval: run with --log_samples; the label is "wrong" (1 - acc), the score is the model's own
# risk (1 - top-choice probability from the choices' log-likelihoods)
baseline-increment from-lm-eval out/*/samples_*.jsonl -o items.csv
baseline-increment items.csv --score risk --label wrong --group group --baselines task,log_prompt_words

# Inspect: label from one scorer, detector from another scorer (or a numeric metadata field)
baseline-increment from-inspect logs/*.json --label-scorer match --score-scorer judge -o items.csv
baseline-increment items.csv --score score --label wrong --group group --baselines log_input_words,log_output_words
```

Both adapters are tested against a known result: on real logs (lm-eval on pythia-14m, Inspect on its mock model), the per-sample table reproduces the harness's own reported accuracy exactly. `.eval` Inspect logs are compressed and need `inspect_ai` installed; `.json` logs need nothing extra.

## Reading your data (v0.9)

**Inspect logs, as they are:**
- **Formats:** `.eval` (Inspect's default) and `.json` logs, or whole directories of them. `.eval` is read natively, so `inspect_ai` is not needed. A CI test checks that the native reader gives exactly what `inspect_ai`'s own reader gives.
- **Discover what's in the logs:** `--list` shows every scorer, its value types and its dict keys, plus the metadata fields.

  ```bash
  baseline-increment from-inspect logs/ --list
  baseline-increment from-inspect logs/ --label-scorer match --score-scorer judge:confidence -o items.csv
  ```

- **Grouping:** rows are grouped by question (`task:sample id`) across models and epochs, so one question never sits in both training and test folds. Before 0.9 they were grouped per log file; `--group-by file` restores that. The adapter refuses to run when one sample id carries different input text in different logs.
- **Length baselines** count message text only:
  - chat message lists, content parts and `attachment://` references are resolved;
  - before 0.9, chat-style inputs were counted from the list's printed form, which inflated them (about 70% too many words on short chat prompts).
- **Scorers:** a dict-valued scorer is chosen with `name:key`.
- **Partial and missing answers:** `P` (partial) and `N` (no answer) follow `--partial-as` / `--noanswer-as`. In safety evaluations a no-answer is often a refusal, so choose deliberately.
- **No silent drops:** every row that cannot be used is counted by reason, printed, and written to the panel JSON.
- **Accuracy check:** on the shipped fixture (30 samples × 3 epochs), the adapter reproduces Inspect's reported accuracy exactly.

**Any other export (CSV, TSV, Parquet, JSON Lines or a JSON array):**
- **Formats:** every command reads all of them. Nested JSON is flattened to dot paths. Parquet needs `pip install 'baseline-increment[parquet]'`.
- **Mapping file:** `--map` renames those dot paths to the column names you use, and derives a 0/1 label from text verdicts. A value it doesn't know is an error, not a silent gap.

  ```json
  {"columns": {"score": "scores.judge.value", "group": "metadata.doc_id", "task": "metadata.task"},
   "derive":  {"wrong": {"from": "labels.verdict", "map": {"pass": 0, "fail": 1}}}}
  ```

  ```bash
  baseline-increment export.jsonl --map map.json --score score --label wrong --group group --baselines task
  ```

**Platform exports: check the row count first.** We tried a local MLflow 3.18 GenAI evaluation (code-based scorers, 40 items, 24 Nov 2026). The run's metric covered all 40 items, but its results table and stored traces held only 13–14 rows. So there is no dedicated MLflow adapter. If you map a platform's per-row export, first check that its row count matches the N the platform reports.

## Difficulty from your own panel (v0.6)

**A difficulty table carries over best when your grader resembles the one that built it; a small panel under your own protocol is the safe default.** Before v0.6 we expected a large public panel to fix the single-model case. A first check on one benchmark says it does so only when the grading matches. The build is in `scripts/build_helm_nq.py`; the numbers are in `difficulty_meta.json`.

What we measured, on NaturalQuestions:
- **Within one protocol, a few model families are enough.** In HELM Lite (41 model families, 1,000 questions), difficulty from k families correlates with difficulty from a held-out half-panel:

  | k (families) | 1 | 3 | 5 | 10 | 20 |
  |---|---|---|---|---|---|
  | median Spearman | 0.79 | 0.89 | 0.92 | 0.94 | 0.96 |

- **Across protocols, it depends on the grader.** On 478 questions shared with the `nq_open` table, we predicted our models' own errors (AUROC):

  | Our labels | HELM 41-family difficulty | 5 other models, our own protocol |
  |---|---|---|
  | LLM judge | 0.59–0.65 | 0.77–0.84 |
  | string match (closer to HELM's grader) | 0.71–0.80 | 0.80–0.90 |

  - Our own panel is mostly Qwen, so the right-hand column is inflated for the Qwen models.
  - On OLMo, the one non-Qwen model: judge labels 0.65 vs 0.78; string-match labels 0.79 vs 0.80.
  - The grader alone moves difficulty a lot: the same responses graded by an LLM judge vs string match give difficulties that correlate at only 0.69.
  - Exploratory: one benchmark, one non-sibling model.
  - **A second benchmark cannot come from public data.** In HELM's public results (Lite and Classic, checked 24 Nov 2026), NaturalQuestions is the only benchmark shared with a shipped table; TriviaQA, PopQA and SciQ are absent. Replicating this needs new model runs.
  - **The grader half, re-checked on three more benchmarks (24 Nov 2026; pre-registered, same responses graded both ways):**
    - **What was measured:** difficulty from the other models under the *same* grader, against the *other* grader, as a predictor of the held-out model's errors.
    - **Matched grading always predicted better.** The size of the gap varied.
    - **TriviaQA,** the pre-registered primary: +0.02 to +0.04 AUROC on the three non-Qwen models, CIs excluding 0. That is smaller than the 0.05 the pre-registration needed to call it a replication, so the verdict is inconclusive.
    - **SciQ:** +0.07 to +0.08. **PopQA:** +0.02 to +0.03. **NQ:** +0.09.
    - **Pattern, noticed afterwards (not confirmatory):** the gap appears to track how much the two graders disagree. On NQ and SciQ they agree at Spearman 0.65–0.68; on TriviaQA and PopQA, at 0.91.
    - **In practice:** a mismatched grader always cost something here, and cost most where the graders disagree most.
  - **The protocol half, re-checked on TriviaQA (24 Nov 2026; pre-registered):**
    - **Panel:** five base-model families that share nothing with ours (Falcon3, Granite-3.1, Yi-1.5, SmolLM2, Pythia).
    - **Setup:** run under a different protocol (5-shot `Q:/A:`, greedy) and graded with our string-match grader.
    - **Result:** this panel predicted each non-Qwen model's errors almost as well as its own-protocol peers did. The AUROC gap was −0.000, +0.009 and +0.028.
    - **With judge labels instead,** the gap grew to +0.036 to +0.055.
    - **What it says:** the prompting setup barely mattered here, and a mismatched grader cost about 0.03 more. That is real, but smaller than the 0.05 the pre-registration needed to call the grader effect replicated. A sixth model, DeepSeek-7B, was dropped because of a text-decoding fault in our run script.

So the recommended path is a **small panel under your own protocol**. In Inspect, run the same task on a few cheap models, then let the adapter compute each row's difficulty from the *other* models:

```bash
# the same task on several models (one command per model); .json logs need nothing extra to read
for m in openai/gpt-4o-mini together/meta-llama/Llama-3.1-8B-Instruct-Turbo ...; do
  inspect eval my_task.py --model "$m" --log-format json
done
baseline-increment from-inspect logs/ --label-scorer match --score-scorer judge -o items.csv --lomo
baseline-increment check items.csv --score score --label wrong --question item --model model \
    --difficulty lomo_share_wrong --panel-meta items.csv.panel.json
```

How `--lomo` (also on `from-lm-eval`) works:
- It adds `lomo_share_wrong`, the error rate of the other models on the same item (each model's epochs averaged first), and `n_ref_models`.
- It groups rows by item, so a question never straddles folds.
- It writes the panel metadata that `check` needs to headline the difficulty increment.
- With fewer than 5 reference models, both tools say the increment is an upper bound.

**Shipped protocol-bound table: `--benchmark helm_nq`.**
- **What it is:** NaturalQuestions closed-book under HELM Lite (5-shot, quasi-exact-match), 92 runs in 41 model families, 1,000 questions, with the reliability curve above in its metadata.
- **When to use it:** when your setup is close to HELM's. `check` warns that the table is protocol-bound.
- **Before relying on it:** check the "benchmark difficulty" baseline's own AUROC against your labels.
- **Rebuild it:** `python scripts/build_helm_nq.py CACHE_DIR` rebuilds it from HELM's public results.

**Question matching** now tolerates a trailing "?" (NQ-open drops it, HELM keeps it), and it is an error, not an empty report, when nothing matches.
