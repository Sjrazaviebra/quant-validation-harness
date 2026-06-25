# -*- coding: utf-8 -*-
"""
ValidationHarness / costs.py - pessimistic cost model (spread + swap + slippage). v0.1

Every backtest P&L MUST pass through this before any verdict. Pessimistic by design:
slippage adds to spread; swap is debited per night with TRIPLE on Wednesday (T+2 settlement);
all in instrument points, converted to account % by the caller's sizing. Generalised
from prior internal trading-research code.
"""
import numpy as np
import pandas as pd


def wednesdays_between(t0, t1):
    """Count Wednesdays in [date(t0), date(t1)) -> +2 swap-nights each (triple-Wed)."""
    d0 = pd.Timestamp(t0).normalize(); d1 = pd.Timestamp(t1).normalize()
    if d1 <= d0:
        return 0
    rng = pd.date_range(d0, d1 - pd.Timedelta(days=1), freq="D")
    return int((rng.dayofweek == 2).sum())


def trade_cost_points(spread_pts, slippage_pts, slip_mult=2.0):
    """Per-trade round-trip transaction cost in instrument points.
    ASSUMPTION (review): the strategy's gross P&L is measured at MID -> one full `spread_pts`
    covers the entry+exit half-spreads (round trip). If your gross is already executed on one
    side, pass spread_pts accordingly. Slippage is counted on BOTH crossings (slip_mult=2.0,
    pessimistic) since breakouts/stops through-fill on entry AND exit. Raise to stress."""
    return float(spread_pts) + slip_mult * float(slippage_pts)


def swap_cost_money(notional, swap_pct_night, t_entry, t_exit, nights):
    """Overnight financing in account currency over the hold, triple-Wed. `nights` = calendar
    nights from entry to exit (caller computes, conservative-rounded). Returns positive cost."""
    wed = wednesdays_between(t_entry, t_exit)
    swap_nights = nights + 2 * wed
    return swap_pct_night * notional * swap_nights


def net_pnl_money(gross_pts, pt_value, lot, spread_pts, slippage_pts,
                  notional, swap_pct_night, t_entry, t_exit, nights):
    """Full net P&L (account ccy) for one trade after spread+slippage+swap. The single funnel
    every strategy P&L passes through in this harness."""
    cost_pts = trade_cost_points(spread_pts, slippage_pts)
    swap = swap_cost_money(notional, swap_pct_night, t_entry, t_exit, nights)
    return (gross_pts - cost_pts) * pt_value * lot - swap


# sane pessimistic defaults (index CFD) - override per venue
DEFAULTS = dict(spread_pts=1.3, slippage_pts=0.5, swap_pct_night=0.013 / 100.0)
