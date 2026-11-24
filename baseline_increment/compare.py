"""Detector release comparison: old score (v1) vs new score (v2) on the same items (pre-registered in docs/PREREG_0.8.md).

Three questions, never merged:
  T1 improvement: does v2 add information beyond v1 + baselines?   audit(v2 | baselines + v1)
  T2 regression:  does v1 keep information v2 lacks?               audit(v1 | baselines + v2)
  T3 standalone:  is AUROC(v2) - AUROC(v1) different from 0?       paired group bootstrap interval

Both scores get exactly the same transform (rank, optionally within a column) before entering either role.
T1/T2 decisions use Holm-adjusted p over {T1, T2}. A non-rejection is reported as "not detected", never as
"no information".
"""
import numpy as np

from .core import FAMILIES, VERSION, audit, auc, boot_ci, transform_score

_V1, _V2 = "__v1_ranked", "__v2_ranked"


def holm2(p1, p2):
    """Holm-adjusted p-values for two tests."""
    lo, hi = sorted([p1, p2])
    a_lo = min(1.0, 2 * lo)
    a_hi = max(a_lo, min(1.0, hi))
    return (a_lo, a_hi) if p1 <= p2 else (a_hi, a_lo)


def compare(df, old, new, label, group, baselines=(), rank_within=None, families=FAMILIES, seeds=range(2),
            n_perm=200, perm_stop=10, n_boot=0, n_boot_diff=2000, n_jobs=1, alpha=0.05):
    baselines = list(baselines)
    d = df.copy()
    d[_V1] = transform_score(df, old, "rank", rank_within)
    d[_V2] = transform_score(df, new, "rank", rank_within)
    kw = dict(families=families, seeds=seeds, n_perm=n_perm, n_boot=n_boot, perm_stop=perm_stop, n_jobs=n_jobs,
              score_transform="raw", n_boot_desc=200)
    t1 = audit(d, _V2, label, group, baselines + [_V1], **kw)
    t2 = audit(d, _V1, label, group, baselines + [_V2], **kw)
    p1, p2 = holm2(t1["increment_perm_p"], t2["increment_perm_p"])
    imp = bool(p1 < alpha and t1["increment"] > 0)
    reg = bool(p2 < alpha and t2["increment"] > 0)

    y = np.asarray(df[label], int)
    groups = np.asarray(df[group]).astype(str) if group else np.arange(len(y)).astype(str)
    s1, s2 = d[_V1].to_numpy(float), d[_V2].to_numpy(float)
    a1, a2 = auc(y, s1), auc(y, s2)
    ci = boot_ci(y, s2, groups, n_boot_diff, seed=0, s2=s1)
    if ci[0] > 0:
        standalone = "v2 better alone"
    elif ci[1] < 0:
        standalone = "v2 worse alone"
    else:
        standalone = "difference not detected"

    n_items, n_groups = int(len(y)), int(len(np.unique(groups)))
    size = f"{n_items} items, {n_groups} groups"
    rep = {
        "version": VERSION, "n_items": n_items, "n_groups": n_groups, "old": old, "new": new,
        "settings": {"baselines": baselines, "rank_within": rank_within, "families": list(families), "seeds": len(list(seeds)),
                     "n_perm": n_perm, "perm_stop": perm_stop, "n_boot_diff": n_boot_diff, "alpha": alpha},
        "improvement": {"question": "does v2 add information beyond v1 + baselines?", "increment": t1["increment"],
                        "p": t1["increment_perm_p"], "p_holm": p1, "detected": imp, "headline_family": t1["headline_family"]},
        "regression": {"question": "does v1 keep information that v2 lacks?", "increment": t2["increment"],
                       "p": t2["increment_perm_p"], "p_holm": p2, "detected": reg, "headline_family": t2["headline_family"]},
        "standalone": {"question": "is AUROC(v2) - AUROC(v1) different from 0?", "auc_old": a1, "auc_new": a2,
                       "difference": a2 - a1, "ci": ci, "verdict": standalone},
        "audit_improvement": t1, "audit_regression": t2,
    }
    lines = []
    lines.append("v2 adds information beyond v1 and the baselines" if imp else
                 f"an improvement of v2 over v1 + baselines was not detected ({size})")
    lines.append("v1 keeps information that v2 lacks (regression)" if reg else
                 f"a regression (information in v1 missing from v2) was not detected ({size})")
    if imp and reg:
        lines.append("the two scores are complementary: each carries information the other lacks; consider combining them")
    rep["summary"] = lines
    return rep


def fmt_compare(rep):
    i, r, s = rep["improvement"], rep["regression"], rep["standalone"]
    out = [f"baseline-increment {rep['version']} compare",
           f"items {rep['n_items']}  groups {rep['n_groups']}  old = {rep['old']}  new = {rep['new']}",
           f"standalone AUROC  old {s['auc_old']:.3f}  new {s['auc_new']:.3f}  difference {s['difference']:+.3f} "
           f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  -> {s['verdict']}",
           f"improvement (v2 beyond v1 + baselines)  increment {i['increment']:+.3f}  p={i['p']:.3f}  Holm p={i['p_holm']:.3f}"
           f"  detected={str(i['detected']).lower()}",
           f"regression  (v1 beyond v2 + baselines)  increment {r['increment']:+.3f}  p={r['p']:.3f}  Holm p={r['p_holm']:.3f}"
           f"  detected={str(r['detected']).lower()}"]
    out += [f"- {x}" for x in rep["summary"]]
    return "\n".join(out)
