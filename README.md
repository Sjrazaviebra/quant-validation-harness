# Quant Validation Harness

> An honest, López de Prado–style validation harness that **kills false trading edges before they cost you a euro.**
> Built by **Javad RAZAVI** — *The Solution Maker* · [javadrazavi.fr](https://www.javadrazavi.fr)

Most retail backtests lie. A purely **beta** strategy (just being long a rising market) shows a "significant" Sharpe, a positive excess over buy-&-hold, and mostly-positive out-of-sample folds — **and still loses money live.** This harness corrects the two biases that manufacture false edges: the **number of trials** (multiple testing) and **beta** (exposure ≠ skill).

It exists because I learned it the hard way: a momentum strategy with an in-sample Calmar of ~2 turned out to be **indistinguishable from zero after deflation, and below buy-&-hold** — pure beta on a bull market. This harness would have flagged it *before* any demo. Now every future idea goes through it first.

## What it checks
| Module | Role |
|---|---|
| `directional.py` | **directional-skill PRE-GATE** — out-of-sample hit-rate vs **50% and the cost-adjusted breakeven**, at the lower bound of a **Wilson binomial** *and* a **stationary-bootstrap** 95% CI. Runs **first**: a forecaster with no directional edge (however good its RMSE) is discarded before Sharpe/DSR. Also warns on *objective mismatch*. |
| `validate.py` | Sharpe, **PSR** (Bailey–López de Prado), **Deflated Sharpe** (deflates by # trials via expected-max-Sharpe), **CAPM alpha/beta**, **stationary block bootstrap** (CI), intra-trade-bounded equity/drawdown (MAE) |
| `cpcv.py` | **Combinatorial Purged Cross-Validation** — purge (overlapping spans) + embargo → a distribution of OOS results, not one fragile number |
| `costs.py` | **pessimistic costs** — spread + slippage + triple-Wednesday swap |
| `data.py` | **multi-regime loader** (2008 / 2018 / 2020 / 2022 stress windows) + a synthetic generator (ground truth for the self-test) |
| `gonogo.py` | the **verdict**: `DISCARD` / `INSUFFICIENT DATA` / `DEMO CANDIDATE` (never an auto "go-live") |
| `selftest.py` | the **proof** — flags a fake beta edge, validates a real alpha edge |

## Decision rule (pre-registered, v0.2)
**Pre-gate (forecasting strategies only).** If you pass `pred`/`realized`, the harness first checks **directional skill**: the out-of-sample hit-rate must clear **50% AND the cost-adjusted breakeven hit-rate**, at the *lower* bound of **both** a Wilson binomial and a stationary-bootstrap 95% CI. A model that is near-50% directional — *however good its RMSE* — is `DISCARD`ed **before** the Sharpe/DSR machinery ever runs (see *Why RMSE is not edge* below). It also emits an **objective-mismatch** warning when a model is trained on MSE/RMSE but judged on direction/PnL.

Primary significance = the **ALPHA** (skill beyond beta), tested by **Deflated Sharpe ≥ 0.95 AND a stationary-bootstrap CI of the alpha > 0** (the double test covers autocorrelated residuals).

`DISCARD` if **any** is true:
- alpha not significant (DSR-alpha < 0.95 or bootstrap CI includes 0) → *discard, don't tune*;
- long-biased (`|β| > 0.10`) **and** excess over buy-&-hold ≤ 0 (the beta illusion);
- CPCV unstable (OOS folds not ≥ 60 % positive, p05 ≤ 0, or < 5 folds at n ≥ 30).

Otherwise → `DEMO CANDIDATE` (significant skill → **forward-test on demo first, never straight to live**). `< 50 observations` → `INSUFFICIENT DATA`.

> **Hard guardrail:** if deflation is requested without valid `n_trials` / `sr_trials`, `deflated_sharpe` **raises** — it never silently degrades into an uncorrected PSR.

## Why RMSE is not edge
A low RMSE/MSE does **not** imply tradeable directional skill — and that single confusion manufactures most "AI trading" false edges. The strongest controlled study to date, **Saidd (2026), *A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments*** ([arXiv:2603.16886](https://arxiv.org/abs/2603.16886)), benchmarked nine architectures (Autoformer, DLinear, iTransformer, LSTM, ModernTCN, N-HiTS, PatchTST, TimesNet, TimeXer) across crypto, forex and equity at 4h/24h horizons under a strict five-stage protocol. The result: **directional accuracy was statistically indistinguishable from 50% across *all* 54 model-category-horizon combinations**, even though RMSE differences between architectures were large and statistically significant.

The reason is structural: an MSE/RMSE loss optimises a *point forecast*, not the *sign* of the next move — so a model can win the RMSE leaderboard and still have zero directional skill, unless the loss is explicitly directional (e.g. a Mean Absolute Directional Loss). A leaderboard RMSE gain is therefore **not** an edge.

`directional.py` enforces this lesson: it gates on out-of-sample hit-rate (vs 50% **and** the cost-adjusted breakeven, both via confidence-interval lower bounds) and warns on *objective mismatch* — so neither the harness's users nor my future self get fooled by an RMSE win that doesn't trade.

## Usage
```python
import gonogo as G

res = G.evaluate(
    returns,             # net per-trade P&L in % (AFTER costs via costs.py)
    label="my_strat",
    mkt_ret=mkt,         # per-obs market returns (alpha/beta + buy-&-hold) — ALWAYS provide
    n_trials=40,         # number of configs tested (deflation) — CRITICAL, else toothless
    sr_trials=srs,       # per-obs Sharpes of EVERY config tried (variance → deflation)
    spans=(t0, t1),      # [entry, exit] per trade (CPCV purge)
    strat_ann=..., bh_ann=...,   # annualized returns (excess-over-buy&hold enforced)
    # --- forecasting strategies (ML/transformer): the directional PRE-GATE runs FIRST ---
    pred=preds, realized=rlzd,   # predicted vs realized returns -> directional-skill gate
    breakeven_hitrate=0.50,      # or directional.breakeven_hit_rate(avg_win, avg_loss, cost)
    trained_on="rmse", evaluated_on="directional",  # -> objective-mismatch warning
)
print(res["verdict"])   # near-50% directional -> "JETER" before Sharpe/DSR even run
```
> **`n_trials` + `sr_trials` must account for *every* config you tried** on the same data — otherwise the deflation doesn't bite. That's pitfall #1.

## Self-test (regression) — PASSES ✓
```
python selftest.py     # exit 0 iff all cases pass (multi-seed) + the guardrail raises
```
v0.2, 20 seeds/case: **fake-beta → DISCARD 20/20 · real-alpha → DEMO 20/20 · modest-but-significant alpha → DEMO 20/20 · too-thin alpha → DISCARD 20/20 · deflation guardrail: raises.**
The fake-beta trace shows the trap caught: raw PSR 0.995, excess-over-buy&hold +12 pts, CPCV 93 % positive (everything screams GO) — **but DSR-alpha 0.74 → DISCARD.** The harness sees what the naive eye misses.

It also regression-tests the **directional pre-gate**: a no-skill forecaster (~50 %) is `DISCARD`ed via the pre-gate, a genuine 58 %-directional forecaster passes it, and the objective-mismatch warning fires on (trained=`rmse`, judged=`directional`).

## Requirements
Python 3.13, `numpy`, `scipy`. No paid dependencies.

## License
[MIT](LICENSE) — use it, learn from it, validate honestly.
