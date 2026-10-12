"""Correctness guards for crosspair.py -- the portfolio accounting layer.

Run: .venv/bin/python test_crosspair.py

The bug these guard: the simulator set cash = portfolio_value while ALSO
holding units worth portfolio_value. Equity doubled every bar and reported a
CAGR of 1e170%. A backtester that cannot hold value still constant is
incapable of disproving anything.
"""
from __future__ import annotations

import math

import crosspair as C
import research as R


def test_holding_constant_weights_tracks_the_daily_rebalanced_index():
    """No cost, no signal: equity must equal the DAILY-REBALANCED equal-weight
    portfolio, computed independently.

    Constant target weights mean rebalance every bar -- that is a different
    animal from buy-and-hold-once, and the two diverge in a trending market.
    (The first version of this test asserted the buy-and-hold identity and
    failed: it was wrong, not the simulator.)
    """
    names = ["XBTZAR", "ETHZAR", "LTCZAR"]
    panel = C.build_panel(names)
    k = len(names)
    w = [[1.0 / k] * k for _ in range(len(panel.dates))]
    eq = C.simulate_panel(panel, w, 0, fee=0.0, half_spread_bps=0.0)

    # independent recomputation: each bar, portfolio return is the
    # value-weighted average of the individual pair returns
    expect = 1.0
    for i in range(1, len(panel.dates)):
        r = sum(panel.closes[n][i] / panel.closes[n][i - 1] for n in names) / k
        expect *= r
    got = eq[-1] / eq[0]
    assert abs(got - expect) / expect < 1e-9, (
        f"simulator returned {got:.8f}, daily-rebalanced index says {expect:.8f}")
    print(f"ok: constant weights track the daily-rebalanced index exactly "
          f"({expect:.4f}x -- no phantom compounding)")


def test_equity_never_doubles_without_a_reason():
    """The regression: one bar of a flat market must not double the book."""
    # flat prices across every pair -> equity must stay exactly flat
    panel = C.build_panel(["XBTZAR"])
    k = 1
    w = [[1.0] for _ in range(len(panel.dates))]
    eq = C.simulate_panel(panel, w, 0, fee=0.0, half_spread_bps=0.0)
    # XBTZAR is not flat, so instead check the ratio is a real price ratio
    ratio = eq[-1] / eq[0]
    px_ratio = panel.closes["XBTZAR"][-1] / panel.closes["XBTZAR"][0]
    assert abs(ratio - px_ratio) / px_ratio < 1e-9, f"{ratio} vs {px_ratio}"
    print("ok: single-pair equity equals the price ratio (no doubling)")


def test_fees_reduce_equity_and_are_monotone():
    """More cost must never produce more money."""
    panel = C.build_panel(["XBTZAR", "ETHZAR", "LTCZAR", "XRPZAR"])
    k = 4
    w = [[1.0 / k] * k for _ in range(len(panel.dates))]
    free = C.simulate_panel(panel, w, 5, fee=0.0, half_spread_bps=0.0)
    cheap = C.simulate_panel(panel, w, 5, fee=0.001)
    dear = C.simulate_panel(panel, w, 5, fee=C.LUNO_TAKER_ZAR)
    assert cheap[-1] < free[-1], "charging a fee increased equity"
    assert dear[-1] < cheap[-1], "Luno's 0.6% fee beat 0.1%"
    print(f"ok: costs bite monotonically -- free {free[-1]:.4f} > "
          f"0.1% {cheap[-1]:.4f} > 0.6% {dear[-1]:.4f}")


def test_weights_sum_to_one_or_less():
    """Leverage is impossible on Luno spot, so the book must not exceed 1.0."""
    panel = C.build_panel(["XBTZAR", "ETHZAR"])
    for spec in (C.cs_momentum(panel, 30, 1), C.cs_momentum(panel, 30, 2, False)):
        bad = [sum(row) for row in spec if sum(row) > 1.0 + 1e-12]
        assert not bad, f"weights sum above 1.0 ({max(bad)}); would imply shorting"
    print("ok: no strategy ever sums weights above 1.0 (spot-only constraint)")


def test_panel_alignment_has_no_invented_bars():
    """Alignment must INTERSECT dates, never forward-fill a missing close."""
    panel = C.build_panel(["XBTZAR", "ETHZAR", "BCHZAR"])
    for n in panel.names:
        raw = {c.timestamp: float(c.close) for c in R.load_pair(n)}
        for t, px in zip(panel.dates, panel.closes[n]):
            assert t in raw, f"{n} has no bar at {t} -- forward-filled?"
            assert raw[t] == px, f"{n} price at {t} does not match the source"
    print(f"ok: every panel price is a real traded close "
          f"({len(panel.dates)} dates x {len(panel.names)} pairs)")


def test_metrics_on_cross_pair_results_are_finite_and_sane():
    """No inf, no 1e100 CAGR, no negative drawdown."""
    panel = C.build_panel(C.R.available_pairs())
    eq = C.simulate_panel(panel, C.cs_momentum(panel, 30, 3), 200)
    for e in eq:
        assert math.isfinite(e) and e > 0, f"bad equity point {e}"
    m = R.metrics(eq, 365.0)
    assert math.isfinite(m.cagr) and abs(m.cagr) < 1e6, f"implausible CAGR {m.cagr}"
    assert 0 <= m.maxdd <= 100, f"maxDD out of range: {m.maxdd}"
    print(f"ok: cross-pair result is finite and sane "
          f"(CAGR {m.cagr:+.1f}%, maxDD {m.maxdd:.1f}%)")


def test_single_pair_sim_never_goes_short_or_negative():
    """Regression: the fee was charged as an ABSOLUTE amount, so any asset
    priced above 1/fee bought negative units. BTC at R1.39m bought -6,000 BTC
    and printed a 1,305,742% drawdown on a long-only position."""
    for pair in ("XBTZAR", "ETHZAR"):
        eq = C.sim_single(pair, 5)
        assert all(e > 0 for e in eq), f"{pair} produced non-positive equity"
        m = R.metrics(eq, 365.0)
        assert 0 <= m.maxdd <= 100, f"{pair} maxDD {m.maxdd} out of range"
        assert abs(m.maxdd) < 100, f"{pair} cannot lose more than 100% long-only"
    print("ok: single-pair sim is long-only and never exceeds a 100% drawdown")


if __name__ == "__main__":
    for fn in (test_holding_constant_weights_tracks_the_daily_rebalanced_index,
               test_single_pair_sim_never_goes_short_or_negative,
               test_equity_never_doubles_without_a_reason,
               test_fees_reduce_equity_and_are_monotone,
               test_weights_sum_to_one_or_less,
               test_panel_alignment_has_no_invented_bars,
               test_metrics_on_cross_pair_results_are_finite_and_sane):
        fn()
    print("\nall crosspair accounting checks passed")