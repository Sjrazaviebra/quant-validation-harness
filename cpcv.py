# -*- coding: utf-8 -*-
"""
ValidationHarness / cpcv.py - Combinatorial Purged Cross-Validation (Lopez de Prado). v0.1

Splits the trade/observation stream into N contiguous time groups; tests on every combination
of k groups; trains/selects on the rest with PURGE (drop train obs whose label span overlaps a
test obs span) + EMBARGO (drop train obs within an embargo buffer after each test block). Yields
C(N,k) out-of-sample evaluations -> a DISTRIBUTION of OOS performance instead of one fragile
number. Works for a fixed strategy (selector=None) or a strategy that re-selects per fold.

Observations carry a label SPAN [t0, t1] (entry..exit) so purge removes look-ahead from
overlapping holds — the swing's multi-day holds made this essential.
"""
import itertools
import numpy as np


def _purge_embargo(train_idx, test_idx, t0, t1, embargo):
    """Remove train obs whose [t0,t1] overlaps any test obs [t0,t1] (purge), and train obs that
    start within `embargo` (in obs-index units) AFTER a test block's end (embargo)."""
    test_spans = [(t0[i], t1[i]) for i in test_idx]
    keep = []
    test_starts = set(test_idx)
    for i in train_idx:
        a, b = t0[i], t1[i]
        overlap = any(not (b < ts or a > te) for ts, te in test_spans)   # span intersection
        if overlap:
            continue
        # embargo: drop train obs whose index falls in (test_end, test_end+embargo]
        emb = any(0 < (i - j) <= embargo for j in test_starts)
        if emb:
            continue
        keep.append(i)
    return keep


def cpcv_paths(returns, t0, t1, n_groups=6, k_test=2, embargo_frac=0.02,
               selector=None, evaluator=None):
    """Run CPCV. returns: per-obs returns (used by default evaluator). t0/t1: label span per obs
    (numeric, comparable; e.g. integer bar index or epoch seconds). selector(train_idx)->params
    (None = fixed strategy). evaluator(test_idx, params)->array of returns (None = returns[test]).
    Returns dict with the list of per-combination OOS metric tuples and the pooled OOS returns."""
    returns = np.asarray(returns, dtype=float)
    t0 = np.asarray(t0); t1 = np.asarray(t1)
    n = len(returns)
    if n < 3 * n_groups:                    # guard: too few obs -> groups would be empty/degenerate
        return dict(per_combo=[], pooled=np.array([]), n_combos=0, sharpe_median=float("nan"),
                    sharpe_p05=float("nan"), mean_median=float("nan"), frac_sharpe_pos=float("nan"),
                    var_oos_sharpe=float("nan"), insufficient=True)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [list(range(bounds[g], bounds[g + 1])) for g in range(n_groups)]
    embargo = int(round(embargo_frac * n))
    combos = list(itertools.combinations(range(n_groups), k_test))
    per_combo = []
    pooled = []
    for c in combos:
        test_idx = [i for g in c for i in groups[g]]
        train_idx = [i for g in range(n_groups) if g not in c for i in groups[g]]
        train_idx = _purge_embargo(train_idx, test_idx, t0, t1, embargo)
        params = selector(train_idx) if selector else None
        oos = evaluator(test_idx, params) if evaluator else returns[test_idx]
        oos = np.asarray(oos, dtype=float)
        if len(oos) < 3:
            continue
        per_combo.append(dict(combo=c, n_test=len(oos), mean=float(oos.mean()),
                              sharpe=float(oos.mean() / oos.std(ddof=1)) if oos.std(ddof=1) > 0 else 0.0,
                              total=float(oos.sum())))
        pooled.append(oos)
    pooled = np.concatenate(pooled) if pooled else np.array([])
    sharpes = [d["sharpe"] for d in per_combo]
    means = [d["mean"] for d in per_combo]
    return dict(per_combo=per_combo, pooled=pooled, n_combos=len(per_combo),
                sharpe_median=float(np.median(sharpes)) if sharpes else float("nan"),
                sharpe_p05=float(np.percentile(sharpes, 5)) if sharpes else float("nan"),
                mean_median=float(np.median(means)) if means else float("nan"),
                frac_sharpe_pos=float(np.mean([s > 0 for s in sharpes])) if sharpes else float("nan"),
                var_oos_sharpe=float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else float("nan"))
