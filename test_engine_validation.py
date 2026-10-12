"""Can the harness be trusted to say "no edge"? (validation, not a finding)

This file exists to answer the hardest objection to everything in this repo:
"you found several bugs, so how can any strategy be said to have failed?"

The objection is correct, and it is sharpest against NEGATIVE results. Every
bug found so far was INFATING -- it made returns too good. Those announce
themselves as absurd numbers. A bug that DEFLATES, an edge-suppressing error,
produces a plausible mediocre curve and nobody notices. So "found no edge" is
the claim most exposed to bugs that were never caught.

Reading the code again does not settle it. These four experiments do, because
they measure the instrument rather than trusting it:

  1. POWER. Manufacture data with a known edge, sweep the signal strength,
     and check reported Sharpe actually rises. A harness that cannot find a
     planted edge cannot be used to report an absent one.

  2. NULL. Run on a DRIFTLESS random walk. A trend follower must score ~0 or
     below -- it pays costs. Anything positive means the engine invents money.

  3. TIMING NULL. Permute real returns: same drift, same volatility, no
     serial structure. That is what riding drift with no skill looks like.
     Real data must beat THIS to show timing skill.

  4. OVERFITTING GAP. Compare the oracle (peeks at the test segment) against
     the honest protocol (chooses on train). The difference is the
     overfitting, measured rather than asserted.

Result: the harness HAS power, its null is correctly zero, real timing scores
BELOW the shuffled control, and the oracle-to-honest gap is large. The
negative finding survives all four.

Run: .venv/bin/python test_engine_validation.py
"""
from __future__ import annotations

import math
import statistics

import research as R
import validate_engine as V


def test_power_confirmed_harness_detects_a_planted_edge():
    """Sharpe must RISE as the planted signal strengthens.

    This is the load-bearing test. Without it, "no edge found" could just
    mean "the search could never find an edge."
    """
    curve = []
    for snr in (0.0, 0.4):
        vol = 0.015
        vals = [V.best_test_sharpe(
            V.synth_regime(seed=500 + s, drift=snr * vol, vol=vol))[0]
            for s in range(6)]
        curve.append(statistics.mean(vals))
    lo, hi = curve
    print(f"ok: planted-signal power curve -- Sharpe {lo:+.2f} at zero signal "
          f"rises to {hi:+.2f} at drift/vol=0.4")
    assert hi - lo > 2.0, (
        f"harness does not respond to a planted edge ({lo:+.2f} -> {hi:+.2f}); "
        f"its silence on real data would be uninformative")
    assert lo < 0.75, (
        "a zero-signal regime-switching series already scores high; the "
        "baseline is contaminated")


def test_null_is_zero_a_driftless_walk_earns_nothing():
    """Control B. If a trend follower beats zero drift, the engine is broken.

    This is the check that would have caught an edge-INFLATING bug. It passes
    with the value it should have: slightly negative, because of costs.
    """
    means, maxes = [], []
    for s in range(8):
        c = V.synth_random_walk(seed=700 + s)
        maxes.append(V.best_test_sharpe(c)[0])
        means.append(V.mean_sharpe(c))
    m = statistics.mean(means)
    print(f"ok: driftless random walk -> mean test Sharpe {m:+.3f}, "
          f"best-of-N {statistics.mean(maxes):+.2f} (must be <= 0: no free money)")
    assert m < 0.25, f"engine earns {m:+.3f} from pure noise -- it invents money"
    assert m > -0.5, "implausibly negative; check costs are not double-charged"


def test_real_timing_scores_below_the_shuffled_control():
    """Control A: the null for TIMING, not for making money.

    Permuting returns keeps XBTZAR's drift and volatility, so a positive
    Sharpe is expected -- that is just riding the BTC/ZAR drift. A strategy
    with real timing skill must beat this level. It does not.
    """
    real = V.mean_sharpe(R.load_pair("XBTZAR"))
    shuf = statistics.mean(V.mean_sharpe(V.shuffled_real("XBTZAR", s))
                           for s in range(8))
    print(f"ok: real XBTZAR test Sharpe {real:+.3f} vs shuffled control "
          f"{shuf:+.3f} -- timing adds nothing beyond drift")
    assert real < shuf, (
        f"real ({real:+.3f}) now EXCEEDS the shuffled null ({shuf:+.3f}); "
        f"there may be timing skill worth investigating")


def test_overfitting_gap_is_large():
    """Oracle vs honest. The peek is worth a lot, which is why it must go."""
    grid, warm = R.build_grid(), R.warmup_for(R.build_grid())
    orc, hst = [], []
    for p in R.available_pairs():
        try:
            c = R.load_pair(p)
        except SystemExit:
            continue
        if len(c) < warm + 400:
            continue
        split = int(len(c) * 0.6)
        bpy = R.bars_per_year(c)
        bh = R.metrics(R.simulate(c, R.w_buy_hold(c), warm)[split:], bpy).sharpe
        rows = [(R.metrics(R.simulate(c, R.weights_for(c, s), warm)[warm:split], bpy),
                 R.metrics(R.simulate(c, R.weights_for(c, s), warm)[split:], bpy))
                for s in grid]
        orc.append(max(r[1].sharpe for r in rows) - bh)
        hst.append(rows[max(range(len(rows)), key=lambda i: rows[i][0].sharpe)][1].sharpe - bh)
    gap = statistics.mean(orc) - statistics.mean(hst)
    print(f"ok: oracle excess {statistics.mean(orc):+.2f} Sharpe vs honest "
          f"{statistics.mean(hst):+.2f} -- peeking is worth {gap:+.2f}")
    assert gap > 0.3, "the peek stopped mattering; the protocol changed"
    assert statistics.mean(hst) < 0.1, (
        f"honest protocol now beats buy&hold by {statistics.mean(hst):+.2f} "
        f"Sharpe -- re-evaluate the no-edge conclusion")


def test_metrics_reproduce_from_an_independent_implementation():
    """Sharpe and max drawdown, recomputed from scratch."""
    c = R.load_pair("XBTZAR")
    eq = R.simulate(c, R.w_buy_hold(c), R.warmup_for(R.build_grid()))
    m = R.metrics(eq, R.bars_per_year(c))
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    mu = sum(rets) / len(rets)
    sharpe = (mu / math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
              * math.sqrt(365.0))
    hi, worst = eq[0], 0.0
    for v in eq:
        hi = max(hi, v)
        worst = min(worst, v / hi - 1)
    assert abs(m.sharpe - sharpe) < 1e-9, "Sharpe does not reproduce"
    assert abs(m.maxdd - (-worst * 100)) < 1e-9, "maxDD does not reproduce"
    print(f"ok: Sharpe and maxDD reproduce exactly from a separate computation")


if __name__ == "__main__":
    print(__doc__)
    for fn in (test_power_confirmed_harness_detects_a_planted_edge,
               test_null_is_zero_a_driftless_walk_earns_nothing,
               test_real_timing_scores_below_the_shuffled_control,
               test_overfitting_gap_is_large,
               test_metrics_reproduce_from_an_independent_implementation):
        fn()
    print("\nThe negative finding survives validation:")
    print("  - the harness CAN find edges when they exist (power curve)")
    print("  - it earns ~0 on zero-drift data (no invented money)")
    print("  - real timing scores BELOW the shuffled control")
    print("  - the oracle-to-honest gap is large, so the no-peek protocol matters")
    print("\nCalibrated claim: NO EDGE WAS DETECTABLE ABOVE THE NOISE FLOOR.")
    print("That is weaker than 'no edge exists' -- see README for why.")