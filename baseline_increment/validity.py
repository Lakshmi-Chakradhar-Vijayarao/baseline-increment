"""v0.4 estimands beyond the difficulty increment (decision-point H-2V / panel-size / re-analysis, Nov 2026).

* ``decomposition``: probability-scale split of model x item error propensity into a shared item part, a model-specific
  part and binomial sampling noise, from repeated sampled-answer labels, plus each detector's efficiency R2_v (the
  noise-corrected fraction of model-specific propensity variance its within-item deviation explains). The two halves
  of the samples give independent estimates; the split-half correlation of the model-specific part is reported.
* ``panel_sensitivity``: the detector increment over leave-one-out panel difficulty D1 built from k reference models,
  for k = 1..K-1, and a c/k extrapolation (an assumption, labelled as such).
* ``incremental_dprime``: equal-variance binormal transform of a (baseline, baseline + detector) AUROC pair. Model-based;
  reported beside, never instead of, the AUROC numbers.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def normal_score(s):
    from scipy.stats import rankdata
    s = np.asarray(s, float)
    return norm.ppf(rankdata(s) / (len(s) + 1))


def incremental_dprime(base_auc, full_auc):
    """d'_add = sqrt(d'_full^2 - d'_base^2), d' = sqrt(2) * Phi^-1(AUROC). Returns (d_base, d_full, d_add, standalone_auc)."""
    d0, d1 = (np.sqrt(2) * norm.ppf(np.clip(a, 1e-6, 1 - 1e-6)) for a in (base_auc, full_auc))
    add = float(np.sqrt(max(d1 ** 2 - d0 ** 2, 0.0)))
    return {"d_base": float(d0), "d_full": float(d1), "d_add": add, "standalone_equivalent_auc": float(norm.cdf(add / np.sqrt(2))),
            "assumption": "equal-variance binormal, independent additive information"}


def _grid(df, item, model, cols):
    """Wide arrays items x models for each column; requires a complete grid."""
    out = {}
    for c in cols:
        w = df.pivot(index=item, columns=model, values=c)
        if w.isna().any().any():
            raise ValueError(f"incomplete item x model grid for {c!r}: every model needs every item")
        out[c] = w.to_numpy(float)
    return out


def _half_stats(P, n):
    M = P.shape[1]; r = P - P.mean(0, keepdims=True); w = r - r.mean(1, keepdims=True)
    s2e = float(np.mean(P * (1 - P) / (n - 1))); sv = float(np.var(w, ddof=1) * M / (M - 1) - s2e)
    sq = float(np.var(r.mean(1), ddof=1) - (sv + s2e) / M)
    return w, s2e, sv, sq


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _solve_offsets(pbar, theta):
    """a_m with mean_q expit(a_m + theta_q) = pbar_m (bisection; monotone in a_m)."""
    lo = np.full(len(pbar), -30.0); hi = np.full(len(pbar), 30.0)
    for _ in range(60):
        mid = (lo + hi) / 2; f = _expit(mid[:, None] + theta[None, :]).mean(1) - pbar
        lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
    return (lo + hi) / 2


def additive_null(P_halves, n, grid=(0.5, 1, 1.5, 2, 3, 4, 6, 8), reps=4, seed=0):
    """Probability-scale share produced with NO model x item interaction: logit p = a_m + theta_q, theta ~ N(0, s^2),
    a_m solved so each model keeps its observed mean error. Logistic nonlinearity alone makes models of different
    ability diverge on middling items, which the probability-scale estimator reads as 'model-specific'.
    s is chosen so the simulated between-item variance matches the observed one."""
    P = np.mean(P_halves, 0); Q, M = P.shape; pbar = np.clip(P.mean(0), 1e-4, 1 - 1e-4)
    obs_q = float(np.var(P.mean(1), ddof=1)); rng = np.random.default_rng(seed); rows = []
    for s in grid:
        sh, vq = [], []
        for _ in range(reps):
            th = rng.normal(0, s, Q); a = _solve_offsets(pbar, th); p = _expit(a[None, :] + th[:, None])
            Ph = rng.binomial(n, np.broadcast_to(p, (Q, M))) / n
            _, _, sv, sq = _half_stats(Ph, n); sh.append(sv / (sq + sv)); vq.append(np.var(Ph.mean(1), ddof=1))
        rows.append((s, float(np.mean(sh)), float(np.mean(vq))))
    best = min(rows, key=lambda r: abs(r[2] - obs_q))
    return {"null_share": best[1], "matched_sd_theta": best[0], "null_share_range": [min(r[1] for r in rows), max(r[1] for r in rows)],
            "ability_spread_logit_sd": float(np.std(np.log(pbar / (1 - pbar)), ddof=1)),
            "note": "share expected from ability differences alone (no interaction); subtract before interpreting"}


def decomposition(df, item, model, sample_cols, scores=(), n_boot=1000, seed=0, min_sig2_v=0.005):
    """sample_cols: >= 4 columns of 0/1 'wrong' labels of independently sampled answers (split into two halves).
    scores: detector columns (higher = more likely wrong). Returns shares, split-half r, R2_v per score, with question-bootstrap CIs."""
    sample_cols = list(sample_cols)
    if len(sample_cols) < 4:
        raise ValueError("need at least 4 sampled-answer label columns (two halves of >= 2)")
    h = len(sample_cols) // 2; A, B = sample_cols[:h], sample_cols[h:2 * h]
    d = df.copy(); d["_pA"] = d[A].mean(1); d["_pB"] = d[B].mean(1)
    G = _grid(d, item, model, ["_pA", "_pB"] + list(scores))
    Z = {s: np.column_stack([normal_score(G[s][:, j]) for j in range(G[s].shape[1])]) for s in scores}

    def stats(ix):
        res = {}; per = []
        for P in (G["_pA"][ix], G["_pB"][ix]):
            w, s2e, sv, sq = _half_stats(P, h); per.append((w, s2e, sv, sq))
        res["model_specific_share"] = float(np.mean([sv / (sq + sv) for _, _, sv, sq in per]))
        res["sig2_v"] = float(np.mean([sv for _, _, sv, _ in per]))
        res["split_half_r"] = float(np.corrcoef(per[0][0].ravel(), per[1][0].ravel())[0, 1])
        M = per[0][0].shape[1]
        for s in scores:
            # cross-half: the slope is fitted on one sample half and scored on the other (as in DP), so a pure-noise
            # detector is centred on 0 and can go negative; an in-sample slope makes R2_v >= 0 by construction
            Sw = Z[s][ix] - Z[s][ix].mean(1, keepdims=True); vals = []
            for (wf, _, _, _), (w, s2e, sv, _) in ((per[0], per[1]), (per[1], per[0])):
                b = (wf * Sw).sum() / (Sw * Sw).sum(); svs = np.var(w - b * Sw, ddof=1) * M / (M - 1) - s2e
                vals.append(1 - svs / sv)
            res[f"R2_v[{s}]"] = float(np.mean(vals))
        return res

    Q = G["_pA"].shape[0]; pt = stats(np.arange(Q)); rng = np.random.default_rng(seed)
    boots = [stats(rng.integers(0, Q, Q)) for _ in range(n_boot)]
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in pt}
    defined = pt["sig2_v"] >= min_sig2_v
    nl = additive_null([G["_pA"], G["_pB"]], h, seed=seed)
    nl["excess_share"] = pt["model_specific_share"] - nl["null_share"]
    out = {"n_items": int(Q), "n_models": int(G["_pA"].shape[1]), "samples_per_half": h, "estimate": pt, "ci": ci,
           "r2_v_defined": bool(defined), "scale": "probability (moment estimator, binomial noise subtracted)",
           "additive_null": nl,
           "caveat": "the probability-scale share depends on the panel's ability spread (logistic nonlinearity); "
                     "read it with additive_null.excess_share and quote it per panel, never as a property of the models alone"}
    if not defined:
        out["note"] = f"model-specific variance {pt['sig2_v']:.4f} < {min_sig2_v}: R2_v is not defined (nothing model-specific to explain)"
    return out


def panel_sensitivity(df, item, model, label, score, n_subsets=10, seed=0, seeds=range(3)):
    """Increment of `score` over D1 = mean label of k other models (same item), pooled over models, for k = 1..K-1.
    Returns delta(k), the all-others value, and a c/k extrapolation (assumed form)."""
    from .core import auc
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    Y = _grid(df, item, model, [label])[label].astype(int); Sg = _grid(df, item, model, [score])[score]
    Qn, M = Y.shape
    if M < 3:
        return {"status": "NOT CHECKABLE", "reason": f"needs >= 3 models sharing items (got {M})"}
    rng = np.random.default_rng(seed)

    def inc(y, D, s):
        vals = []
        for sd in seeds:
            f = np.random.default_rng(sd).permutation(len(y)) % 5; p0 = np.zeros(len(y)); p1 = np.zeros(len(y))
            for k in range(5):
                te = f == k; tr = ~te
                if y[tr].min() == y[tr].max():
                    continue
                X0 = D[:, None]; X1 = np.column_stack([D, normal_score(s)])
                p0[te] = LogisticRegression(max_iter=1000).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1]
                p1[te] = LogisticRegression(max_iter=1000).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
            vals.append(auc(y, p1) - auc(y, p0))
        return float(np.mean(vals))

    by_k = {}
    for k in range(1, M):
        cell = []
        for j in range(M):
            if not 0 < Y[:, j].sum() < Qn:
                continue
            others = [o for o in range(M) if o != j]
            subs = [others] if k == M - 1 else [list(rng.choice(others, k, replace=False)) for _ in range(n_subsets)]
            cell.append(np.mean([inc(Y[:, j], Y[:, s_].mean(1), Sg[:, j]) for s_ in subs]))
        by_k[k] = float(np.mean(cell))
    ks = np.array(sorted(by_k)); dv = np.array([by_k[k] for k in ks])
    if len(ks) >= 2:
        X = np.column_stack([np.ones(len(ks)), 1 / ks]); w = np.sqrt(ks)
        c = np.linalg.lstsq(X * w[:, None], dv * w, rcond=None)[0]; d_inf = float(c[0])
    else:
        d_inf = None
    return {"delta_by_k": {int(k): v for k, v in by_k.items()}, "delta_all_others": by_k[M - 1], "n_models": int(M),
            "delta_extrapolated": d_inf, "extrapolation_form": "delta(k) = delta_inf + c/k (assumed)",
            "sensitive": bool(d_inf is not None and abs(by_k[M - 1] - d_inf) >= 0.005)}
