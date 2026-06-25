# -*- coding: utf-8 -*-
"""
ValidationHarness / gonogo.py - the verdict orchestrator. v0.2 (post-review)

Honest verdict on a strategy's per-observation net returns. Pre-registered rule (Coordinator):
if the Deflated Sharpe is not significant AFTER the multiple-testing correction -> VERDICT JETER
(do NOT tune). Plus the beta/alpha guard that catches the NDX illusion.

v0.2 review fixes: (a) PRIMARY significance = ALPHA (skill beyond beta), tested by Deflated
Sharpe AND stationary-bootstrap CI (not the over-strict raw+alpha double gate) ; (b) excess-over
-buy-hold required as soon as beta is meaningfully positive (>0.10) ; (c) CAPM-fallback detected
-> cannot separate alpha -> conservative ; (d) CPCV needs enough OOS per fold.

Verdicts: "JETER", "DONNEES INSUFFISANTES", "CANDIDAT DEMO" (never auto GO-live).
"""
import numpy as np
import validate as V
import cpcv as C

PSR_SIG = 0.95
BETA_LONGBIAS = 0.10       # |beta| above this -> excess-over-buy-hold is a hard gate (review)
MIN_OBS = 50
MIN_CPCV_FRAC_POS = 0.60
MIN_CPCV_OBS_PER_FOLD = 30


def evaluate(returns, label="strategy", mkt_ret=None, n_trials=1, sr_trials=None,
             spans=None, strat_ann=None, bh_ann=None, obs_per_year=None, verbose=True, boot_b=1000):
    r = np.asarray(returns, dtype=float)
    out = {"label": label, "n": len(r)}
    log = []

    def p(s):
        log.append(s)
        if verbose:
            print(s)

    p(f"=== VALIDATION : {label} (n={len(r)}) ===")
    if len(r) < MIN_OBS:
        out["verdict"] = "DONNEES INSUFFISANTES"
        p(f"  n={len(r)} < {MIN_OBS} -> DONNEES INSUFFISANTES.")
        out["log"] = log
        return out

    sr = V.sharpe(r)
    psr0 = V.psr(r, 0.0)
    dsr_raw, sr0, _ = V.deflated_sharpe(r, n_trials=n_trials, sr_trials=sr_trials)
    lo_m, hi_m, mean_m = V.block_bootstrap_ci(r, lambda x: x.mean(), b=boot_b)
    out.update(sharpe_perobs=sr, psr_gt0=psr0, dsr_raw=dsr_raw, sr0_deflation=sr0,
               n_trials=n_trials, mean=mean_m, mean_lo=lo_m, mean_hi=hi_m)
    p(f"  Sharpe/obs={sr:.3f} | PSR(>0)={psr0:.3f} | n_trials={n_trials} sr0={sr0:.3f} DSR(raw)={dsr_raw:.3f}")
    p(f"  mean/obs IC95 block-boot [{lo_m:.5f} .. {hi_m:.5f}] -> {'>0' if lo_m>0 else 'INCLUT 0'}")

    # ---- core significance ----
    have_mkt = mkt_ret is not None
    if have_mkt:
        alpha, beta, astream, ok = V.capm_alpha_beta(r, mkt_ret)
        if not ok:
            p("  !! fit CAPM degenere (variance marche nulle / n trop faible) -> alpha non separable")
            core_sig = (dsr_raw >= PSR_SIG and lo_m > 0)      # fall back to raw, conservatively
            core_lbl = "raw (CAPM fallback)"
            out.update(beta=float("nan"), alpha=float("nan"), capm_fallback=True)
        else:
            a_sig, dsr_a, a_lo = V.alpha_is_significant(astream, n_trials, sr_trials, b=boot_b)
            core_sig = a_sig
            core_lbl = "alpha"
            out.update(beta=beta, alpha=alpha, dsr_alpha=dsr_a, alpha_ci_lo=a_lo, capm_fallback=False)
            p(f"  vs marche : beta={beta:.2f} alpha/obs={alpha:.5f} | DSR(alpha)={dsr_a:.3f} "
              f"alpha-CI_lo={a_lo:.5f} -> alpha {'SIGNIFICATIF' if a_sig else 'non significatif'}")
    else:
        core_sig = (dsr_raw >= PSR_SIG and lo_m > 0)
        core_lbl = "raw (pas de marche fournie)"
        beta = float("nan")
    out["core_significant"] = bool(core_sig)

    # ---- excess over buy-hold (IMPOSED) ----
    excess = None
    if strat_ann is not None and bh_ann is not None:
        excess = V.excess_over_buy_hold(strat_ann, bh_ann)
        out.update(strat_ann=strat_ann, bh_ann=bh_ann, excess_over_bh=excess)
        p(f"  rendement {strat_ann:+.1f}%/an vs buy-hold {bh_ann:+.1f}%/an -> EXCES {excess:+.1f} pts")

    # ---- CPCV OOS distribution ----
    cp = None
    if spans is not None:
        t0, t1 = spans
        cp = C.cpcv_paths(r, t0, t1, n_groups=6, k_test=2)
        valid = [d for d in cp["per_combo"] if d["n_test"] >= MIN_CPCV_OBS_PER_FOLD]
        cp_ok = len(valid) >= 5
        out.update(cpcv_sharpe_median=cp["sharpe_median"], cpcv_sharpe_p05=cp["sharpe_p05"],
                   cpcv_frac_pos=cp["frac_sharpe_pos"], cpcv_valid_folds=len(valid))
        p(f"  CPCV : {cp['n_combos']} folds ({len(valid)} avec n>={MIN_CPCV_OBS_PER_FOLD}) | "
          f"Sharpe median={cp['sharpe_median']:.3f} p05={cp['sharpe_p05']:.3f} "
          f"frac>0={cp['frac_sharpe_pos']:.0%}")
    else:
        cp_ok = True

    # ---- pre-registered verdict ----
    reasons = []
    if not core_sig:
        reasons.append(f"{core_lbl} non significatif apres deflation #essais (DSR/bootstrap)")
    if (beta == beta and beta > BETA_LONGBIAS) and (excess is not None) and excess <= 0:
        reasons.append(f"long-biais (beta {beta:.2f}) ET excess buy-hold {excess:+.1f}<=0 (illusion beta)")
    if cp is not None:
        if not cp_ok:
            reasons.append(f"CPCV non concluant ({out.get('cpcv_valid_folds',0)} folds valides) = donnees insuffisantes")
        elif not (cp["frac_sharpe_pos"] >= MIN_CPCV_FRAC_POS and cp["sharpe_p05"] > 0):
            reasons.append(f"CPCV instable (frac>0 {cp['frac_sharpe_pos']:.0%}, p05 {cp['sharpe_p05']:.2f})")

    verdict = "JETER" if reasons else "CANDIDAT DEMO"
    out["verdict"] = verdict
    out["reasons"] = reasons
    p(f"  >>> VERDICT : {verdict}")
    for rs in reasons:
        p(f"      - {rs}")
    if verdict == "CANDIDAT DEMO":
        p("      (skill significatif apres deflation ; DEMO FORWARD d'abord, jamais live direct.)")
    out["log"] = log
    return out
