# Worked examples



## Worked example on a public detector (LettuceDetect on RAGTruth)

**The result in use:** at a 10% false-positive rate, metadata alone (task type, generating model, response length) catches 63.7% of hallucinated responses. Adding LettuceDetect raises that to 75.7%, **+12.0 points of recall**. In AUROC terms, +0.060 [0.047, 0.075] over metadata that alone reaches 0.853.


LettuceDetect (`KRLabsOrg/lettucedect-base-modernbert-en-v1`, revision `bbd7783`) scored all 2,700 RAGTruth test responses (RAGTruth at commit `c103204`, MIT). The run was on a Kaggle T4, through `examples/ragtruth_lettucedetect/kaggle_kernel.py`. Score = max token hallucination probability; label = any annotated hallucination span. The baselines are what a release review would ask about: task type, the generating model, and response length. Groups are source documents (each has six responses). The scores, labels and the full output are in `examples/ragtruth_lettucedetect/kaggle_output/`.

```
baseline-increment 0.4.1
items 2700  groups 450  positive rate 0.349
detector AUROC            0.889 [0.872, 0.904]
baseline task_type         0.717  (linear, alone)
baseline generator         0.711  (linear, alone)
baseline log_words         0.662  (linear, alone)
family linear  baselines 0.851  + detector 0.914  increment +0.062
family spline  baselines 0.851  + detector 0.913  increment +0.062
family gbm     baselines 0.853  + detector 0.913  increment +0.060  <- headline (best baselines-only AUROC)
increment (gbm)      +0.060 [0.047, 0.075]  perm p=0.005
stratum 'task_type': base rate alone 0.717
  Data2txt             n=900   pos=0.64  detector AUROC 0.904 [0.882, 0.923]
  QA                   n=900   pos=0.18  detector AUROC 0.900 [0.867, 0.930]
  Summary              n=900   pos=0.23  detector AUROC 0.756 [0.712, 0.797]
```

**What it says:**
- The detector adds information beyond the metadata: +0.060 AUROC over the best metadata baseline, CI [0.047, 0.075], permutation p = 0.005.
- But metadata alone already reach 0.851, so most of the pooled 0.889 is available without the detector.
- Within task, it is strong on QA and data-to-text (about 0.90) and weaker on summaries (0.756).

**In use:** at a fixed false-positive rate of 10%, the metadata alone catches 63.7% of hallucinated responses, and adding the detector raises that to 75.7% (+12.0 points). At 5% the gain is +16.3 points (52.3% → 68.6%). These are out-of-fold figures for the headline gbm family, from `operating_point()` in `kaggle_output/operating_point.json`.

This is the shape of answer the tool exists for:
- **a real increment,** worth about 12 points of recall at a 10% false-positive rate;
- **much smaller than the pooled AUROC suggests,** because most of that AUROC was available from metadata alone;
- **uneven across tasks.**

**Prior art for this effect:** Dubanowska et al. (EMNLP 2025 Findings) showed that task imbalance drives much of the reported performance of representation-based detectors on RAGTruth. This example applies the same question to a trained token-level detector, with a calibrated test.

## Worked example (synthetic, RAG-style)

`examples/synthetic_ragstyle.csv` (in the GitHub repository, not in the PyPI package): 600 responses from three task types and three generators. Hallucination rates differ by task, and the "detector" mostly tracks task type, with a little real signal.

```
baseline-increment examples/synthetic_ragstyle.csv --score detector_score --label hallucinated \
    --group response_id --baselines task_type,generator,log_words --strata task_type --boot 100 --perms 200 --jobs -1

baseline-increment 0.4.1   # reproduced from a clean PyPI install with requirements-lock.txt (15 Nov 2026)
items 600  groups 600  positive rate 0.348
detector AUROC            0.730 [0.687, 0.773]
baseline task_type         0.703  (linear, alone)
baseline generator         0.598  (linear, alone)
baseline log_words         0.456  (linear, alone)
family linear  baselines 0.759  + detector 0.784  increment +0.025  <- headline (best baselines-only AUROC)
family spline  baselines 0.758  + detector 0.778  increment +0.019
family gbm     baselines 0.741  + detector 0.758  increment +0.017
increment (linear)      +0.025 [-0.008, 0.051]  perm p=0.005
stratum 'task_type': base rate alone 0.703
  data2text            n=200   pos=0.64  detector AUROC 0.621 [0.536, 0.702]
  qa                   n=200   pos=0.18  detector AUROC 0.638 [0.532, 0.739]
  summary              n=200   pos=0.23  detector AUROC 0.641 [0.540, 0.731]
WARNING: 'task_type' base rate alone reaches 0.703: at least 80% of the detector's pooled AUROC above chance (0.730) is stratum identity
WARNING: a cheap baseline (0.759) matches or beats the detector (0.730)
```

This takes about 10 minutes on one core and under a minute on 8 cores (`--jobs -1`). The output is identical either way.
With the pinned versions in `requirements-lock.txt` the output matches exactly. Newer library versions can change the last digit of a bootstrap interval (e.g. pandas 3.1 / scikit-learn 1.9 give −0.009 for the −0.008 lower bound above).


This output is from 0.4 and predates 0.13. Since 0.13 the verdict comes from the validated DeLong interval for the gain, not from the permutation test, which is now secondary and experimental. Re-run the example to see the current output: on this synthetic data the gain is +0.023 with 95% CI [+0.006, +0.040], so PASS.

The pooled 0.73 looks respectable. The cheap baselines beat it, task identity explains most of it, and within each task the detector is a modest ~0.63.

This pattern is real on RAGTruth.
- Dubanowska et al. (EMNLP 2025 Findings, arXiv 2509.19372) show that a "flag every data-to-text response" rule nearly matches published representation-based detectors.
- On the full RAGTruth test set, task type plus generator identity alone reach AUROC ≈ 0.84.
- We also ran the tool on ReDeEP's released RAGTruth scores (one CV seed's held-out scores, seed 0). That example is not shipped, because its source has no licence. The 20-seed means are 0.41 / 0.58 / 0.63 within task.
