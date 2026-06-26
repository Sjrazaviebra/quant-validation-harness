# -*- coding: utf-8 -*-
"""
ValidationHarness / directional.py - directional-skill PRE-GATE. v0.1

Why this exists: a low RMSE/MSE does NOT imply tradeable directional skill, and that single
confusion manufactures most "AI trading" false edges. The strongest controlled study to date -
Saidd (2026), "A Controlled Comparison of Deep Learning Architectures for Multi-Horizon
Financial Forecasting: Evidence from 918 Experiments", arXiv:2603.16886 - found directional
accuracy statistically INDISTINGUISHABLE FROM 50% across ALL 54 model-category-horizon
combinations (nine architectures incl. PatchTST / ModernTCN / TimesNet / iTransformer on
crypto, forex and equity at 4h & 24h), even though RMSE gaps between architectures were large
and significant. MSE-trained forecasters optimise a point forecast, not the SIGN -> no
directional skill unless the loss is explicitly directional (e.g. Mean Absolute Directional
Loss). So: a forecasting strategy must clear a directional-skill gate BEFORE the Sharpe/DSR
machinery is even worth running.

Run with py -3.13 (needs numpy, scipy). Self-contained, project-agnostic.
"""
import numpy as np
from scipy.stats import binomtest
import validate as V


def directional_accuracy(pred, realized):
    """Hit rate = fraction of correct SIGN calls, counted only where the model made a directional
    call (pred != 0) AND the move was non-flat (realized != 0). Returns
    (hit_rate, hits, n_eff, hit_array)."""
    p = np.sign(np.asarray(pred, dtype=float))
    y = np.sign(np.asarray(realized, dtype=float))
    n = min(len(p), len(y))
    p, y = p[:n], y[:n]
    mask = (p != 0) & (y != 0)
    n_eff = int(mask.sum())
    if n_eff == 0:
        return float("nan"), 0, 0, np.array([], dtype=float)
    hit = (p[mask] == y[mask]).astype(float)
    return float(hit.mean()), int(hit.sum()), n_eff, hit


def breakeven_hit_rate(avg_win, avg_loss, cost=0.0):
    """Cost-adjusted breakeven hit rate. avg_win, avg_loss are POSITIVE magnitudes (same units as
    cost). Break-even p* solves p*avg_win - (1-p)*avg_loss - cost = 0
    -> p* = (avg_loss + cost) / (avg_win + avg_loss). Symmetric case (avg_win==avg_loss==m):
    p* = 0.5 + cost/(2m). Clamped to [0, 0.999)."""
    w = float(avg_win); l = float(avg_loss); c = float(cost)
    denom = w + l
    if denom <= 0:
        return 0.5
    return float(min(0.999, max(0.0, (l + c) / denom)))


def objective_mismatch_warning(trained_on, evaluated_on):
    """Warn when a model is OPTIMISED on a point-forecast loss (mse/rmse/mae...) but JUDGED on a
    directional/PnL objective -> the Saidd 2026 trap. Returns a message, or '' if no mismatch."""
    if not trained_on or not evaluated_on:
        return ""
    point = {"mse", "rmse", "mae", "l2", "l1", "huber", "smape", "mape"}
    direc = {"directional", "direction", "sign", "hit_rate", "hitrate", "pnl",
             "sharpe", "accuracy", "return"}
    if str(trained_on).lower() in point and str(evaluated_on).lower() in direc:
        return (f"objective mismatch: entraine sur '{trained_on}' (perte point-forecast) mais "
                f"juge sur '{evaluated_on}' -> un bon RMSE n'implique PAS de competence "
                f"directionnelle (Saidd 2026, arXiv:2603.16886). Re-entrainer avec une perte "
                f"directionnelle (ex. Mean Absolute Directional Loss).")
    return ""


def directional_gate(pred, realized, breakeven_hitrate=0.50, margin=0.0, b=1000, conf=0.95):
    """PASS iff the OUT-OF-SAMPLE directional hit rate clears the bar = max(0.50, breakeven)+margin
    at BOTH the lower bound of a Wilson binomial 95% CI AND a stationary-bootstrap 95% CI. A
    near-50% forecaster (the Saidd 2026 result) FAILS. Returns a dict with the verdict + diagnostics.

    breakeven_hitrate : cost-adjusted breakeven (use breakeven_hit_rate()); pass 0.50 if costs are
                        already baked into the SIGN of `realized`.
    margin            : extra cushion required above the bar (a "meaningful margin"; default 0).
    """
    hit_rate, hits, n_eff, hit = directional_accuracy(pred, realized)
    bar = max(0.50, float(breakeven_hitrate)) + float(margin)
    out = dict(hit_rate=hit_rate, hits=hits, n_eff=n_eff, bar=bar,
               breakeven_hitrate=float(breakeven_hitrate), margin=float(margin))
    if n_eff < 1 or hit_rate != hit_rate:
        out.update(ci_binom_lo=float("nan"), ci_boot_lo=float("nan"), low_power=True,
                   passed=False, reason="aucun appel directionnel exploitable (n_eff=0)")
        return out
    # Wilson binomial CI (two-sided), lower bound = does the hit rate provably clear the bar?
    ci = binomtest(int(hits), int(n_eff), p=0.5, alternative="two-sided").proportion_ci(
        confidence_level=conf, method="wilson")
    ci_binom_lo = float(ci.low)
    # stationary-bootstrap CI of the hit rate (same convention as the rest of the harness)
    ci_boot_lo, _, _ = V.block_bootstrap_ci(hit, lambda x: x.mean(), b=b, lo=(1 - conf) / 2 * 100)
    low_power = n_eff < 30
    passed = (ci_binom_lo > bar) and (ci_boot_lo != ci_boot_lo or ci_boot_lo > bar)
    out.update(ci_binom_lo=ci_binom_lo, ci_boot_lo=float(ci_boot_lo), low_power=low_power,
               passed=bool(passed),
               reason=("competence directionnelle nette (les bornes basses des IC passent la barre)"
                       if passed else
                       f"hit {hit_rate:.3f} : CI95_lo binom={ci_binom_lo:.3f} / boot={ci_boot_lo:.3f} "
                       f"n'efface pas la barre {bar:.3f} -> indistinguable du 50% (cf Saidd 2026)"))
    return out
