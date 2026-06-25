# -*- coding: utf-8 -*-
"""
ValidationHarness / data.py - multi-regime data loader + synthetic generator. v0.1

Two jobs:
 (1) Real ingestion : load deep OHLC history (Dukascopy / Norgate / broker CSV) and TAG known
     stress regimes (2008 GFC, 2018-Q4, 2020 COVID, 2022 bear) so any walk-forward spans
     multiple regimes — the single biggest gap in a single-regime backtest (e.g. a 24-month bull-only window).
 (2) Synthetic generator : market path + strategies with KNOWN ground truth (pure-beta fake
     edge / real market-neutral alpha) for the harness self-test.

Ingestion is interface + parser (no network here). Regime tagging works on any datetime index.
"""
import os
import numpy as np
import pandas as pd

# Known stress regimes (UTC date ranges) - extend as needed
REGIMES = {
    "GFC_2008":   ("2008-09-01", "2009-06-30"),
    "Q4_2018":    ("2018-10-01", "2018-12-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "BEAR_2022":  ("2022-01-01", "2022-12-31"),
}


def tag_regime(ts):
    """Return the regime name for a timestamp, or 'normal'."""
    d = pd.Timestamp(ts)
    for name, (a, b) in REGIMES.items():
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name
    return "normal"


def regime_coverage(index):
    """How many distinct stress regimes does this datetime index cover? (gate input: a swing WF
    on <2 regimes is not multi-regime -> harness will warn)."""
    tags = pd.Series([tag_regime(t) for t in index])
    covered = sorted(set(tags) - {"normal"})
    return covered, len(covered)


def load_csv_ohlc(path, tz_to_utc_offset_sec=0):
    """Load a deep OHLC CSV (Dukascopy/Norgate/broker export). Expected columns (case-insensitive):
    time/date, open, high, low, close, [volume]. Returns a UTC-indexed DataFrame. Server-time
    offset (sec) subtracted if the source is broker-server-time (a common DST/server-time pitfall)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"data file not found: {path}")
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time") or cols.get("date") or cols.get("datetime") or df.columns[0]
    df["utc"] = pd.to_datetime(df[tcol], utc=False) - pd.Timedelta(seconds=tz_to_utc_offset_sec)
    out = df.rename(columns={cols.get("open", "open"): "open", cols.get("high", "high"): "high",
                             cols.get("low", "low"): "low", cols.get("close", "close"): "close"})
    keep = ["utc", "open", "high", "low", "close"]
    if "volume" in cols:
        out = out.rename(columns={cols["volume"]: "volume"}); keep.append("volume")
    out = out[keep].dropna().sort_values("utc").reset_index(drop=True)
    return out


def ingest_dukascopy_hint():
    return ("Dukascopy: telecharger via dukascopy-node ou l'historique tick/min, exporter CSV "
            "OHLC ; Norgate: NDX continuous futures depuis 2000+. Puis load_csv_ohlc(path). "
            "Viser >=2008 pour couvrir GFC/2018/2020/2022 (multi-regime).")


# ---------------------------------------------------------------- synthetic (self-test ground truth)
def synth_market(n=4000, mu_ann=0.20, vol_ann=0.20, periods_per_year=252 * 7, seed=1):
    """Geometric random walk 'market' (bull drift by default -> buy-hold is positive, the trap)."""
    rng = np.random.default_rng(seed)
    mu = mu_ann / periods_per_year
    sd = vol_ann / np.sqrt(periods_per_year)
    r = rng.normal(mu, sd, n)
    return r   # per-period market returns


def synth_strategy(mkt_ret, kind, alpha_ann=0.0, beta=0.0, vol_ann=0.10,
                   periods_per_year=252 * 7, participation=1.0, seed=2):
    """Build a strategy's per-trade-equivalent return stream with KNOWN ground truth.
      kind='fake_beta' : return = beta*market + noise, alpha=0 (no skill, pure exposure).
      kind='real_alpha': return = alpha + beta*market + noise (genuine skill on top of beta).
    participation = fraction of periods actually in a position (rest = 0 return)."""
    rng = np.random.default_rng(seed)
    n = len(mkt_ret)
    a = alpha_ann / periods_per_year
    noise = rng.normal(0.0, vol_ann / np.sqrt(periods_per_year), n)
    base = beta * mkt_ret + noise
    if kind == "real_alpha":
        base = base + a
    inpos = rng.random(n) < participation
    r = np.where(inpos, base, 0.0)
    return r
