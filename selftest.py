# -*- coding: utf-8 -*-
"""
ValidationHarness / selftest.py - proof the tool works, multi-seed (post-review). v0.2

The harness must, ROBUSTLY ACROSS SEEDS (not by luck of one seed): FLAG fake edges and VALIDATE
real ones, on a bull market (buy-hold positive = the trap). Cases (ground truth known):
  A fake_beta        : 0.8*market + noise, alpha=0          -> must JETER
  B real_alpha_strong: +alpha + 0.2*market + noise          -> must CANDIDAT DEMO
  C real_alpha_modest: small but genuine, enough data       -> must CANDIDAT DEMO (calibration:
                       the gate is strict but NOT so strict it kills a real modest edge)
  D alpha_too_tiny   : real but tiny on limited data        -> must JETER (unprovable = not greenlit)
Plus G) deflation-bypass guard: deflated_sharpe with bad params MUST RAISE (blocker fix).

    py -3.13 selftest.py
Exit 0 iff all pass-rates meet thresholds AND the guard raises.
"""
import sys
import numpy as np
import data as D
import gonogo as G
import validate as V
import directional as DIR

PPY = 252 * 7
N_TRIALS = 40
SEEDS = 20
SWEEP_B = 200            # light bootstrap for the multi-seed sweep (speed)
DETAIL_B = 1000


def ann(returns):
    n = len(returns)
    return V.annualise((np.prod(1 + np.asarray(returns) / 100.0) - 1) * 100.0, n / PPY)


def spans(n):
    t0 = np.arange(n, dtype=float)
    return t0, t0 + 1.0


def one_run(kind, seed, n, kw, boot_b):
    mkt = D.synth_market(n, seed=1)                          # shared data across trials
    srs = [V.sharpe(D.synth_strategy(mkt, kind, seed=100 + s, **kw) * 100.0) for s in range(N_TRIALS)]
    best = 100 + int(np.argmax(srs))                        # winner's curse (realistic selection)
    r = D.synth_strategy(mkt, kind, seed=best, **kw) * 100.0
    mkt_pct = mkt * 100.0
    t0, t1 = spans(len(r))
    return G.evaluate(r, label=f"{kind}_s{seed}", mkt_ret=mkt_pct, n_trials=N_TRIALS, sr_trials=srs,
                      spans=(t0, t1), strat_ann=ann(r), bh_ann=ann(mkt_pct),
                      obs_per_year=PPY, verbose=False, boot_b=boot_b)


CASES = [
    ("A_fake_beta",         "fake_beta",  "JETER",         dict(beta=0.8, vol_ann=0.12, participation=0.5), 3200, 18),
    ("B_real_alpha_strong", "real_alpha", "CANDIDAT DEMO", dict(alpha_ann=0.12, beta=0.2, vol_ann=0.06, participation=0.7), 3200, 18),
    ("C_real_alpha_modest", "real_alpha", "CANDIDAT DEMO", dict(alpha_ann=0.09, beta=0.1, vol_ann=0.05, participation=0.85), 5000, 14),
    ("D_alpha_too_tiny",    "real_alpha", "JETER",         dict(alpha_ann=0.02, beta=0.1, vol_ann=0.09, participation=0.5), 3200, 16),
]


def main():
    print("### SELF-TEST harnais v0.2 (multi-seed, ground truth, marche haussier) ###")
    # --- guard: deflation must never be silently disabled ---
    print("\n-- garde-fou deflation (doit LEVER ValueError) --")
    guard_ok = False
    try:
        V.deflated_sharpe(np.random.default_rng(0).normal(0, 1, 200), n_trials=None, var_sr=0.01)
        print("   ECHEC : deflated_sharpe n'a PAS leve (deflation neutralisable !)")
    except ValueError as e:
        guard_ok = True
        print(f"   OK : a leve -> {str(e)[:70]}...")

    all_ok = guard_ok

    # --- directional pre-gate (Saidd 2026) : no-skill -> ECHEC, skilled -> PASS, mismatch warns ---
    print("\n-- gate directionnel (Saidd 2026 : ~50% directionnel => ECHEC) --")
    ns_fail = sk_pass = integ = 0
    for s in range(SEEDS):
        rng = np.random.default_rng(500 + s)
        realized = rng.normal(0.0, 1.0, 1500)
        pred_ns = rng.normal(0.0, 1.0, 1500)                       # independent of realized -> ~50%
        if not DIR.directional_gate(pred_ns, realized, b=SWEEP_B)["passed"]:
            ns_fail += 1
        agree = rng.random(1500) < 0.58                            # genuine 58% directional skill
        pred_sk = np.where(agree, np.sign(realized), -np.sign(realized))
        if DIR.directional_gate(pred_sk, realized, b=SWEEP_B)["passed"]:
            sk_pass += 1
        # integration: a no-skill forecaster must be JETER via the pre-gate (before Sharpe/DSR)
        v = G.evaluate(realized, label="ns", n_trials=N_TRIALS,
                       sr_trials=[1.0 + 0.01 * i for i in range(N_TRIALS)],
                       pred=pred_ns, realized=realized, verbose=False, boot_b=SWEEP_B)["verdict"]
        if v == "JETER":
            integ += 1
    mism_ok = bool(DIR.objective_mismatch_warning("rmse", "directional")) and \
        not DIR.objective_mismatch_warning("rmse", "mse")
    dir_ok = ns_fail >= 19 and sk_pass >= 19 and integ >= 19 and mism_ok
    all_ok = all_ok and dir_ok
    print(f"  no-skill -> ECHEC : {ns_fail}/{SEEDS} | skilled(58%) -> PASS : {sk_pass}/{SEEDS} | "
          f"evaluate() pre-gate JETER : {integ}/{SEEDS} | objective-mismatch warn : "
          f"{'OK' if mism_ok else 'ECHEC'} -> {'OK' if dir_ok else 'ECHEC'}")

    print("\n-- robustesse multi-seed (verdict stable sur les seeds) --")
    for name, kind, expect, kw, n, thr in CASES:
        hits = 0
        verdicts = {}
        for s in range(SEEDS):
            v = one_run(kind, s, n, kw, SWEEP_B)["verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1
            if v == expect:
                hits += 1
        ok = hits >= thr
        all_ok = all_ok and ok
        print(f"  {name:22s} attendu={expect:14s} : {hits}/{SEEDS} (seuil {thr}) {verdicts} "
              f"-> {'OK' if ok else 'ECHEC'}")

    # --- one detailed run for the record (case A + B) ---
    print("\n-- run detaille (trace) : A faux-beta puis B vrai-alpha --")
    for name, kind, expect, kw, n, thr in (CASES[0], CASES[1]):
        mkt = D.synth_market(n, seed=1)
        srs = [V.sharpe(D.synth_strategy(mkt, kind, seed=100 + s, **kw) * 100.0) for s in range(N_TRIALS)]
        best = 100 + int(np.argmax(srs))
        r = D.synth_strategy(mkt, kind, seed=best, **kw) * 100.0
        t0, t1 = spans(len(r))
        print(f"\n   ### {name} (attendu {expect}) ###")
        G.evaluate(r, label=name, mkt_ret=mkt * 100.0, n_trials=N_TRIALS, sr_trials=srs,
                   spans=(t0, t1), strat_ann=ann(r), bh_ann=ann(mkt * 100.0), boot_b=DETAIL_B)

    print("\n=================== RESULTAT SELF-TEST ===================")
    print(f"  garde-fou deflation : {'OK' if guard_ok else 'ECHEC'}")
    print(f"  gate directionnel   : {'OK' if dir_ok else 'ECHEC'}")
    print(f"  HARNAIS {'FIABLE (flagge le faux, valide le vrai, robuste aux seeds)' if all_ok else 'DEFAILLANT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
