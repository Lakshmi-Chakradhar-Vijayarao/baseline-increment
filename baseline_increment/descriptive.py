"""Descriptive additions to an audit report (1.0). None of these enters `adds_information`; none is calibrated as a test.

- ladder: the headline-family increment as the declared baselines are added one at a time, in the declared order
- precision-recall: average precision (AUPRC) of the detector alone and, out of fold, of the baselines with and
  without the detector; informative when positives are rare, where AUROC can look good while precision is poor
- duplicates: exact duplicate texts (after lower-casing and collapsing whitespace) that sit in different groups,
  which grouped CV cannot keep apart
"""
import hashlib
import re

import numpy as np
from sklearn.metrics import average_precision_score

from .core import cv_predict, estimate, split_design, transform_score


def ladder(df, score, label, group, baselines, families, seeds, score_transform="rank", rank_within=None):
    y = np.asarray(df[label], int)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    rows = []
    for k in range(len(baselines) + 1):
        num, lin = split_design(df, list(baselines[:k]))
        est, head = estimate(num, lin, s, y, groups, tuple(families), list(seeds))
        rows.append({"baselines": list(baselines[:k]), "family": head, "baselines_auc": est[head]["baseline_auc"],
                     "increment": est[head]["increment"]})
    return rows


def precision_recall(df, score, label, group, baselines, family, seeds, score_transform="rank", rank_within=None):
    y = np.asarray(df[label], int)
    s = transform_score(df, score, score_transform, rank_within)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    num, lin = split_design(df, list(baselines))
    b = np.mean([average_precision_score(y, cv_predict(family, num, lin, None, y, groups, sd)) for sd in seeds])
    f = np.mean([average_precision_score(y, cv_predict(family, num, lin, s, y, groups, sd)) for sd in seeds])
    return {"prevalence": float(y.mean()), "detector_ap": float(average_precision_score(y, np.asarray(df[score], float))),
            "baselines_ap": float(b), "baselines_plus_detector_ap": float(f), "ap_gain": float(f - b), "family": family}


def _norm(t):
    return hashlib.sha1(re.sub(r"\s+", " ", str(t).strip().lower()).encode()).hexdigest()


def duplicates(df, text, group):
    """Exact duplicate texts across different groups."""
    h = df[text].map(_norm)
    if not group:
        n = int(h.duplicated(keep=False).sum())
        return {"text_column": text, "rows_with_duplicates": n, "across_groups": None,
                "note": "no group column: every duplicate can straddle folds"}
    g = df[group].astype(str)
    per = h.groupby(h).apply(lambda idx: g.loc[idx.index].nunique())
    cross = per[per > 1]
    rows = int(h.isin(cross.index).sum())
    return {"text_column": text, "duplicate_texts_across_groups": int(len(cross)), "rows_affected": rows,
            "share_of_rows": rows / len(df)}
