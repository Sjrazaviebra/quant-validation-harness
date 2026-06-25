# -*- coding: utf-8 -*-
"""
ValidationHarness / validate.py - core statistics (Lopez de Prado lineage). v0.1

The math that judges every future strategy idea HONESTLY, so a beta illusion (like the NDX
swing) gets flagged BEFORE any euro. Per-OBSERVATION Sharpe convention throughout (a "return"
= one trade P&L %, or one period return) — PSR/DSR and the deflation benchmark must share the
same frequency, so we never annualise inside the significance tests.

Run with py -3.13 (needs numpy, scipy). Self-contained, project-agnostic.

References:
  - Bailey & Lopez de Prado (2012/2014), "The Sharpe Ratio Efficient Frontier" (PSR),
    "The Deflated Sharpe Ratio" (DSR / deflation for multiple testing).
"""
import math
import numpy as np
from scipy.stats import norm, skew, kurtosis

GAMMA = 0.5772156649015329   # Euler-Mascheroni


def sharpe(returns):
    """Per-observation Sharpe (mean/std, ddof=1). NOT annualised."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def ann_sharpe(returns, obs_per_year):
    """Annualised Sharpe for reporting only (not used in significance tests)."""
    sr = sharpe(returns)
    return sr * math.sqrt(obs_per_year) if sr == sr else float("nan")


def psr(returns, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark) given skew/kurtosis & n.
    Bailey-LdP: PSR = Phi( (SRhat - SR*) * sqrt(n-1) / sqrt(1 - g3*SRhat + (g4-1)/4 * SRhat^2) ),
    SRhat per-observation, g3 skew, g4 NON-excess kurtosis (normal=3). Returns probability in [0,1]."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 8:
        return float("nan")
    sr = sharpe(r)
    if sr != sr:
        return float("nan")
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))   # non-excess (normal -> 3)
    denom_sq = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sharpe(var_sr, n_trials):
    """Expected MAX of n_trials i.i.d. Sharpes (per-obs units) = the deflation benchmark SR0.
    SR0 = sqrt(Var_SR) * [ (1-gamma)*Z^-1(1 - 1/N) + gamma*Z^-1(1 - 1/(N*e)) ]."""
    if n_trials is None or n_trials < 2 or var_sr is None or var_sr <= 0:
        return 0.0
    e = math.e
    return math.sqrt(var_sr) * ((1.0 - GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
                                + GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * e)))


def deflated_sharpe(returns, n_trials, sr_trials=None, var_sr=None):
    """Deflated Sharpe Ratio = PSR with the benchmark set to the expected-max-Sharpe under
    `n_trials` (multiple-testing correction). var_sr = variance of the trials' per-obs Sharpes
    (pass sr_trials to estimate it). Returns (dsr_prob, sr0_benchmark, sr_hat).

    HARD GUARD (review blocker): the deflation must NEVER be silently disabled. A deflation is
    requested whenever sr_trials or var_sr is given; if the trial count is then missing/invalid,
    we RAISE rather than degrade to a plain PSR (which would accept a data-mined false edge)."""
    if sr_trials is not None and len(sr_trials) >= 2:
        if var_sr is None:
            var_sr = float(np.var(np.asarray(sr_trials, dtype=float), ddof=1))
        n_trials = n_trials or len(sr_trials)
    deflation_requested = (sr_trials is not None) or (var_sr is not None) or (n_trials is not None and n_trials > 1)
    if deflation_requested and (n_trials is None or n_trials < 2 or var_sr is None or var_sr <= 0):
        raise ValueError(f"deflation demandee mais parametres invalides (n_trials={n_trials}, "
                         f"var_sr={var_sr}) -> refus de degrader en PSR non-corrige. "
                         f"Fournir n_trials>=2 ET sr_trials (ou var_sr>0).")
    sr0 = expected_max_sharpe(var_sr, n_trials)
    return psr(returns, sr_benchmark=sr0), sr0, sharpe(returns)


# ---------------------------------------------------------------- alpha / beta
def capm_alpha_beta(strat_ret, mkt_ret):
    """OLS of strat on market. Returns (alpha_per_obs, beta, alpha_stream, ok) where alpha_stream =
    alpha + residuals (mean == alpha) -> run PSR/DSR/bootstrap on alpha_stream to test if ALPHA
    (skill beyond beta) is significant. `ok`=False signals a degenerate fit (no market variance or
    too few points) -> the caller must NOT treat the returned stream as a real alpha test."""
    x = np.asarray(mkt_ret, dtype=float)
    y = np.asarray(strat_ret, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 8 or x.std() == 0:
        return float("nan"), float("nan"), y.copy(), False     # fallback flagged
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    return alpha, beta, alpha + resid, True


# ---------------------------------------------------------------- block bootstrap
def block_bootstrap_ci(returns, stat_fn, b=1000, block=None, lo=2.5, hi=97.5, seed=7):
    """TRUE stationary bootstrap (Politis & Romano 1994): GEOMETRIC block lengths (mean=`block`),
    circular wrap -> preserves autocorrelation and does NOT understate the CI (a fixed-block
    bootstrap gives too-narrow CIs -> false-accept risk). `block` defaults to ~n^(1/3) when None.
    Returns (lo_pct, hi_pct, point)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    point = stat_fn(r)
    if n < 30:
        return float("nan"), float("nan"), point
    if block is None:
        block = max(2, int(round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    vals = np.empty(b)
    for k in range(b):
        idx = np.empty(n, dtype=int)
        filled = 0
        while filled < n:
            L = int(rng.geometric(1.0 / block))            # >=1, mean = block
            s = int(rng.integers(0, n))
            take = min(L, n - filled)
            idx[filled:filled + take] = (s + np.arange(take)) % n
            filled += take
        vals[k] = stat_fn(r[idx])
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi)), float(point)


def alpha_is_significant(alpha_stream, n_trials, sr_trials, b=1000):
    """Alpha significant iff BOTH: deflated-Sharpe(alpha) >= 0.95 (multiple-testing) AND the
    stationary-bootstrap 95% CI of the mean alpha excludes 0 (guards against autocorrelated
    residuals inflating the analytic PSR). Returns (is_sig, dsr_alpha, ci_lo)."""
    dsr_a, _, _ = deflated_sharpe(alpha_stream, n_trials=n_trials, sr_trials=sr_trials)
    lo, hi, _ = block_bootstrap_ci(alpha_stream, lambda x: x.mean(), b=b)
    is_sig = (dsr_a == dsr_a and dsr_a >= 0.95) and (lo == lo and lo > 0.0)
    return bool(is_sig), float(dsr_a), float(lo)


# ---------------------------------------------------------------- equity / DD (ported from 1c/1d)
def equity_curve_metrics(pnl_pct, mae_pct=None):
    """Equity path from per-trade % P&L (compounding). maxDD bounded INTRA-trade via MAE (% of
    equity adverse excursion) when provided -> the FN-trailing-relevant drawdown, not understated.
    Returns dict(net, maxdd, maxdd_intra, calmar_periodless)."""
    pnl = np.asarray(pnl_pct, dtype=float)
    mae = np.asarray(mae_pct, dtype=float) if mae_pct is not None else np.zeros(len(pnl))
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    maxdd_i = 0.0
    for i in range(len(pnl)):
        intra_low = eq * (1.0 - mae[i] / 100.0)
        maxdd_i = max(maxdd_i, (peak - intra_low) / peak * 100.0 if peak > 0 else 0.0)
        eq *= (1.0 + pnl[i] / 100.0)
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak * 100.0 if peak > 0 else 0.0)
    net = (eq - 1.0) * 100.0
    return dict(net=net, maxdd=maxdd, maxdd_intra=max(maxdd, maxdd_i))


def annualise(net_pct, years):
    if net_pct <= -100 or years <= 0:
        return -100.0
    return ((1.0 + net_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


def excess_over_buy_hold(strat_ann, bh_ann):
    """IMPOSED on every eval (Coordinator rule 4). Positive = beats just holding. For a
    long-biased strategy this is the honest bar; a rising underlying flatters absolute return."""
    return strat_ann - bh_ann
