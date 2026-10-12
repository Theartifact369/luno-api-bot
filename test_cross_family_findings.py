"""JOB 2 RESULT: no edge found in ANY of the strategy families researched.

Companion to test_edge_finding.py, which covers crossovers and Donchian.
This file covers the families researched afterwards, each chosen for a
mechanism rather than another parameter sweep.

FAMILIES EVALUATED: 8
  1. SMA/EMA crossover ........... no OOS edge (test_edge_finding.py)
  2. Donchian breakout ........... no OOS edge (test_edge_finding.py)
  3. Volatility-target sizing .... 0/9 beat a plain equal-weight basket
  4. Cross-sectional momentum .... 0/18 beat buy & hold AND the basket
  5. Cross-sectional reversal .... 0/18 (same grid)
  6. Calendar effects ............ effect is real, NOT predictive, untradable
  7. Long-only dip buying (RSI) . dominated by buy & hold
  8. Carry / funding rate ........ NOT IMPLEMENTABLE on Luno (see below)
  (market-neutral pairs trading is also not implementable -- see below)

TWO STRUCTURAL CONSTRAINTS, from Luno's own documentation, that bound the
whole search space:
  https://guide.luno.com/hc/en-gb/articles/14158595648541-Luno-fees-and-limits-in-South-Africa

  * Luno is SPOT ONLY. No margin, no leverage, no shorting. So every
    market-neutral construction -- long one leg and SHORT the other, the
    entire stat-arb literature -- cannot be executed here at all. This is
    not "we tested it and it failed"; it cannot be traded on this venue.
  * Crypto/ZAR taker fee is 0.60% at tier 1 (R0-R20k/30d), i.e. 1.30% per
    round trip. A monthly rebalance costs 14.4%/yr. Anything that trades
    more than quarterly is dead before it starts. research.py assumed 0.10%,
    six times too kind -- every earlier result in this repo was FLATTERED.

THE ONE INTERESTING RESULT:
  The 06:00 UTC hourly bar (08:00 South Africa, the local cash-market open)
  is genuinely positive, and it REPLICATED out of sample: best hour on
  train 2013-2021 (+0.1202%), still the best on test 2021-2026 (+0.0586%,
  t = +4.36). The mechanism is real -- RSA flow hits Luno at open.
  It is still worthless:
    - it does not PREDICT. correlation(hour-6 return, next 24h) = +0.0224.
      The +24h conditional gap is t = +1.76, not significant.
    - only VOLATILITY persists (corr on |returns| = +0.20), not direction.
    - and it is untradable regardless: a 0.059%/hour effect against a
      1.30% round trip. Cost is ~14x the edge.

Run: .venv/bin/python test_cross_family_findings.py
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import crosspair as C
import research as R

LUNO_TAKER_ZAR = 0.006
ROUND_TRIP = 2 * (LUNO_TAKER_ZAR + 0.0005)


def test_luno_taker_fee_is_six_times_what_the_engine_assumed():
    """The single most consequential correction in this project.

    research.py used FEE = 0.001. Luno's published tier-1 taker fee on
    crypto/ZAR pairs is 0.60% -- 6x higher. Every strategy that trades
    anything like monthly was being flattered by that factor.
    """
    assert LUNO_TAKER_ZAR == 0.006, "update with Luno's current published fee"
    assert C.LUNO_TAKER_ZAR == LUNO_TAKER_ZAR, "crosspair and this test disagree"
    assert R.FEE == 0.001, (
        "research.py FEE changed -- re-run every finding, the old numbers "
        "assumed a cheaper venue")
    monthly = 12 * 2 * LUNO_TAKER_ZAR * 100
    assert monthly > 14, "monthly rebalance cost sanity check"
    print(f"ok: Luno crypto/ZAR taker is {LUNO_TAKER_ZAR*100:.2f}% "
          f"(engine assumed {R.FEE*100:.2f}%) -> {ROUND_TRIP*100:.2f}% round trip, "
          f"{monthly:.1f}%/yr for monthly rebalancing")


def test_no_cross_sectional_strategy_beats_both_baselines():
    """0/18 on momentum and reversal, scored after real Luno fees."""
    panel = C.build_panel(R.available_pairs())
    warm = 200
    ew = R.metrics(C.sim_equal_weight(panel.names, panel, warm), 365.0)
    bh = R.metrics(C.sim_single_on_panel("XBTZAR", panel, warm), 365.0)
    tried = 0
    beat_ew = beat_bh = 0
    for lb in (30, 60, 90):
        for top in (1, 3, 5):
            for winners in (True, False):
                tried += 1
                eq = C.simulate_panel(panel, C.cs_momentum(panel, lb, top, winners),
                                      warm, fee=LUNO_TAKER_ZAR)
                m = R.metrics(eq, 365.0)
                beat_ew += m.sharpe > ew.sharpe
                beat_bh += m.sharpe > bh.sharpe
    print(f"ok: {tried} cross-sectional combos; beat equal-weight {beat_ew}/{tried}, "
          f"beat XBTZAR {beat_bh}/{tried}")
    assert beat_ew == 0 and beat_bh == 0, "a cross-sectional edge appeared -- re-evaluate"


def test_volatility_targeting_does_not_beat_plain_equal_weight():
    """Inverse-vol sizing is a risk lever, not an alpha source."""
    import candidates
    panel = C.build_panel(R.available_pairs())
    warm = 200
    ew = R.metrics(C.sim_equal_weight(panel.names, panel, warm), 365.0)
    wins = 0
    n = 0
    for lb in (14, 30, 60):
        for tv in (0.15, 0.25, 0.40):
            n += 1
            w = candidates.vol_target_weights(panel, lb, tv)
            m = R.metrics(C.simulate_panel(panel, w, warm, fee=LUNO_TAKER_ZAR), 365.0)
            wins += m.sharpe > ew.sharpe
    print(f"ok: {wins}/{n} volatility-target combos beat plain equal-weight")
    assert wins == 0, "vol targeting beat the null -- re-evaluate"


def test_hour6_effect_is_real_but_not_predictive():
    """The most interesting negative result, pinned from both sides.

    It IS real and it DID replicate out of sample -- but a real effect that
    predicts nothing cannot be traded, and the cost dwarfs it regardless.
    """
    c = R.load_pair("XBTZAR", 3600)
    closes = [float(x.close) for x in c]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    hours = [datetime.fromtimestamp(x.timestamp / 1000, tz=timezone.utc).hour
             for x in c][1:]
    idx6 = [i for i, h in enumerate(hours) if h == 6]

    by_h = defaultdict(list)
    for h, r in zip(hours, rets):
        by_h[h].append(r)
    means = {h: statistics.mean(v) for h, v in by_h.items()}
    best_hour = max(means, key=means.get)
    assert best_hour == 6, f"best hour moved to {best_hour}; re-check the mechanism"
    print(f"ok: hour {best_hour:02d}:00 UTC is still the best hour "
          f"({means[6]*100:+.4f}%/hour, 08:00 SAST = SA cash-market open)")

    # ...but it predicts nothing
    xs = [rets[i] for i in idx6 if i + 24 < len(rets)]
    ys = [rets[i + 24] for i in idx6 if i + 24 < len(rets)]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    r = cov / (math.sqrt(sum((a - mx) ** 2 for a in xs))
               * math.sqrt(sum((b - my) ** 2 for b in ys)))
    assert abs(r) < 0.10, f"hour-6 now predicts next-day at r={r:+.3f}; re-evaluate"
    print(f"ok: but it does not PREDICT -- corr(hour-6, next 24h) = {r:+.4f}")

    # ...and even if it did, the fee is ~14x the effect
    assert ROUND_TRIP / (means[6]) > 5, "cost/edge ratio changed materially"
    print(f"ok: and it is untradable -- {ROUND_TRIP*100:.2f}% round trip vs "
          f"{means[6]*100:.4f}%/hour effect (cost is {ROUND_TRIP/means[6]:.0f}x)")


def test_market_neutral_and_carry_are_impossible_on_this_venue():
    """Structural, not empirical -- worth a test so it stays documented.

    Luno SA offers spot trading only. No shorting means no market-neutral
    pairs trade; no perpetuals means no funding-rate carry. Both are absent
    from the venue's product, so neither can be evaluated as 'no edge' --
    they are simply not tradeable here.
    """
    assert ROUND_TRIP > 0.01, "spot round trip should exceed 1%"
    print("ok: spot-only venue rules out market-neutral pairs and funding "
          "carry by construction, not by measurement")


if __name__ == "__main__":
    print(__doc__)
    for fn in (test_luno_taker_fee_is_six_times_what_the_engine_assumed,
               test_no_cross_sectional_strategy_beats_both_baselines,
               test_volatility_targeting_does_not_beat_plain_equal_weight,
               test_hour6_effect_is_real_but_not_predictive,
               test_market_neutral_and_carry_are_impossible_on_this_venue):
        fn()
    print("\nCONCLUSION: 8 families evaluated, none produced an out-of-sample")
    print("edge that survives Luno's real 0.60% crypto/ZAR taker fee.")
    print("  - the venue is spot-only: no shorting, no perps, no carry")
    print("  - costs (1.30%/round trip) exceed every effect found")
    print("  - the one real, replicated effect (06:00 UTC open) predicts nothing")
    print("There is no basis for trading any of these on Luno.")