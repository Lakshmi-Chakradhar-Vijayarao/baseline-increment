"""The 0.12 vectorised helpers must reproduce the original per-group loops exactly (PREREG_0.12.md, Part D)."""
import numpy as np

from baseline_increment.core import GroupIndex, permute_scores, split_design


def _permute_loop(s, groups, strata_key, rng):
    g = np.unique(groups)
    items = {k: np.where(groups == k)[0] for k in g}
    if strata_key is None:
        target = dict(zip(g, rng.permutation(g)))
    else:
        target = {}
        gs = np.array([strata_key[items[k][0]] for k in g])
        for v in np.unique(gs):
            members = g[gs == v]
            target.update(zip(members, rng.permutation(members)))
    sp = np.empty(len(s))
    for k in g:
        src = items[target[k]]
        for n, i in enumerate(items[k]):
            sp[i] = s[src[min(n, len(src) - 1)]]
    return sp


def _boot_loop(groups, rng):
    g = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in g}
    return np.concatenate([idx[k] for k in rng.choice(g, len(g))])


def test_permutation_and_bootstrap_match_loops():
    r = np.random.default_rng(0)
    for trial in range(120):
        n = int(r.integers(5, 800))
        groups = r.integers(0, int(r.integers(2, n + 1)), n).astype(str)
        s = r.normal(size=n)
        sk = None if trial % 3 == 0 else r.choice(list("abc"), n)
        assert np.array_equal(_permute_loop(s, groups, sk, np.random.default_rng(trial)),
                              permute_scores(s, groups, sk, np.random.default_rng(trial)))
        assert np.array_equal(_boot_loop(groups, np.random.default_rng(trial)),
                              GroupIndex(groups).boot(np.random.default_rng(trial)))


def test_one_hot_matches_loop():
    import pandas as pd
    df = pd.DataFrame({"c": ["b", "a", "c", "a", 3, "b"]})
    _, lin = split_design(df, ["c"])
    v = np.asarray(df["c"])
    lv = sorted(set(map(str, v)))
    assert np.array_equal(lin, np.array([[str(x) == l for l in lv] for x in v], float))
