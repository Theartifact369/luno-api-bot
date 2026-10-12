"""THE ANSWER: no tradable edge was found in SMA/EMA crossovers on Luno ZAR
pairs. This file is that conclusion, written down and made executable.

Every claim below is reproducible with the command printed above it. If a
future change makes any of them fail, the claim was wrong and the
strategy engine should be re-evaluated rather than trusted.

WHAT WAS TESTED
  44 parameter combinations x 16 Luno ZAR pairs = 704 backtests
  (SMA and EMA crossovers over 10 period pairs, Donchian breakouts over 4
   channel pairs, and volatility-targeted crossovers at 2 target vols).
  XBTZAR daily bars 2013-04-13 -> 2026-10-11 (4930 bars); 15 other pairs
  with 781-2652 bars each.

WHAT WAS FOUND
  1. No out-of-sample return edge. Walk-forward (train 750 bars, trade the
     next 250, roll forward) beat buy & hold on 4 of 8 pairs. 4/8 is a coin
     flip. Repeating with other splits gave 5/11, 3/6 and 1/6.

  2. The apparent drawdown improvement is reduced exposure, not foresight.
     Walk-forward max drawdown looked excellent -- lower than buy & hold on
     8/8 pairs. But time-in-market was only 43%, and a constant-fraction
     portfolio with NO signal shows the same effect: halving exposure cuts
     max drawdown from 82.1% to 61.8% on its own. Once drawdown is divided
     by time-in-market, the crossover's advantage INVERTS: lower drawdown
     than buy & hold on only 3 of 8 pairs, worse on 5.

  3. The signal is not just useless, it is slightly backwards. Splitting
     each pair's daily returns into bars the crossover held vs bars it
     avoided: held bars were WORSE than avoided bars on 14 of 16 pairs
     (z = -3.00 against a coin flip). The rule holds through declines and
     sits out rallies.

WHY THE ORIGINAL +74343% WAS NOT AN EDGE
  paper_trader.sell() sweeps realised profit into a 'vault' of BTC held
  forever and marked at the final price. That vault is 'I was long BTC in
  2014'. Separated out: strategy +54689%, vault-on +74585%, buy & hold
  +66090% -- the strategy loses to doing nothing. And 12 trades with a
  Sharpe of 0.36 cannot support any claim.

WHAT THIS DOES NOT PROVE
  That crypto markets are efficient, or that no strategy can work. It
  proves only that this family of rules, on this data, out of sample, on
  these pairs, does not. The honest response is to not trade it.

Run: .venv/bin/python test_edge_finding.py
"""
from __future__ import annotations

import statistics
from operator import attrgetter

import research as R

MIN_PAIRS = 8
# A rule must beat buy & hold on well over half the pairs to be considered.
# 5/8 is inside noise; the rule fails at 4/8.
BEAT_RATE_REQUIRED = 0.60


def test_grid_is_the_one_that_was_searched():
    """44 combos: the search must stay auditable, not quietly grow."""
    g = R.build_grid()
    assert len(g) == 44, f"grid changed to {len(g)} combos; update the docstring"
    kinds = {s["kind"] for s in g}
    assert kinds == {"sma", "ema", "donchian", "voltarget"}, kinds
    print(f"ok: grid is {len(g)} combos across {sorted(kinds)}")


def test_no_out_of_sample_return_edge():
    """Walk-forward must NOT beat buy & hold consistently.

    Guards the headline claim. If this ever passes with a real margin, the
    rules engine deserves another look -- the conclusion is not dogma.
    """
    from stress import walk_chain
    grid, warm, key = R.build_grid(), R.warmup_for(R.build_grid()), attrgetter("sharpe")
    got = {}
    for p in R.available_pairs():
        r = walk_chain(R.load_pair(p), grid, key, warm, 750, 250)
        if r:
            got[p] = r
    assert len(got) >= MIN_PAIRS, f"only {len(got)} pairs had enough data"
    wins = sum(1 for v in got.values() if v[0].sharpe > v[1].sharpe)
    rate = wins / len(got)
    print(f"ok: walk-forward beat buy & hold on {wins}/{len(got)} pairs "
          f"({rate:.0%}) -- required >= {BEAT_RATE_REQUIRED:.0%} to claim an edge")
    assert rate < BEAT_RATE_REQUIRED, (
        f"beat rate {rate:.0%} exceeds threshold -- an edge may now exist, "
        f"re-evaluate before assuming otherwise")


def test_drawdown_advantage_is_reduced_exposure():
    """The 8/8 'lower drawdown' result must be explained by sitting in cash.

    Normalising drawdown by time-in-market is what kills it: the crossover
    must NOT still beat buy & hold once you charge for how little it traded.
    """
    from stress import walk_chain
    grid, warm, key = R.build_grid(), R.warmup_for(R.build_grid()), attrgetter("sharpe")
    got = {}
    for p in R.available_pairs():
        r = walk_chain(R.load_pair(p), grid, key, warm, 750, 250)
        if r:
            got[p] = r
    raw_better = sum(1 for v in got.values() if v[0].maxdd < v[1].maxdd)
    per_better = 0
    tims = []
    for st, bh, _d, _c, _h in got.values():
        tim = max(st.time_in_market, 1e-9)
        tims.append(st.time_in_market)
        if st.maxdd / tim < bh.maxdd:
            per_better += 1
    n = len(got)
    print(f"ok: raw drawdown beats buy & hold on {raw_better}/{n} pairs "
          f"(mean TiM {statistics.mean(tims):.0%})")
    print(f"ok: drawdown-per-unit-exposure beats it on only {per_better}/{n} "
          f"-- the advantage was exposure, not foresight")
    assert per_better <= n / 2, (
        "normalised drawdown advantage survives -- re-examine the conclusion")


def test_held_bars_are_not_better_than_avoided_bars():
    """The sharpest test: does holding beat NOT holding?

    If the crossover has timing value, bars it held must return more than
    bars it sat out. It does the opposite, on 14/16 pairs.
    """
    warm = R.warmup_for(R.build_grid())
    diffs = []
    for p in R.available_pairs():
        c = R.load_pair(p)
        if len(c) < warm + 400:
            continue
        bpy = R.bars_per_year(c)
        w = R.w_sma(c, 50, 200)
        cl = R.closes_of(c)
        held, avoided = [], []
        for i in range(warm + 1, len(c)):
            r = cl[i] / cl[i - 1] - 1
            (held if w[i] > 1e-12 else avoided).append(r)
        if len(held) < 60 or len(avoided) < 60:
            continue
        diffs.append(statistics.mean(held) - statistics.mean(avoided))
    n = len(diffs)
    wins = sum(1 for d in diffs if d > 0)
    se = (n * 0.25) ** 0.5
    z = (wins - n / 2) / se if se else 0.0
    mean_d = statistics.mean(diffs) * R.bars_per_year(R.load_pair("XBTZAR"))
    print(f"ok: held bars beat avoided bars on {wins}/{n} pairs, "
          f"mean diff {mean_d:+.2f}%/yr, z={z:+.2f}")
    assert z < -1.5 and mean_d < 0, (
        "the held-vs-avoided relationship has changed; re-evaluate")


def test_cash_is_the_honest_default():
    """A flat curve must stay the benchmark nobody accidentally beats."""
    import test_research
    c = test_research.synth(500)
    eq = R.simulate(c, R.w_cash(c), 0)
    assert all(abs(e - 1.0) < 1e-15 for e in eq)
    print("ok: cash remains exactly flat -- no strategy can 'beat' it by accident")


if __name__ == "__main__":
    print(__doc__)
    for fn in (test_grid_is_the_one_that_was_searched,
               test_no_out_of_sample_return_edge,
               test_drawdown_advantage_is_reduced_exposure,
               test_held_bars_are_not_better_than_avoided_bars,
               test_cash_is_the_honest_default):
        fn()
    print("\nCONCLUSION: no out-of-sample edge found in SMA/EMA crossovers "
          "or Donchian breakouts across 16 Luno ZAR pairs.\n"
          "  - return edge:      NONE (4/8 pairs out of sample, a coin flip)\n"
          "  - drawdown 'edge':  exposure artefact, inverts when normalised\n"
          "  - signal quality:   BACKWARDS (held < avoided on 14/16 pairs)\n"
          "The rules engine should not be traded on this evidence.")