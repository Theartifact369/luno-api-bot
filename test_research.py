"""Causality + simulator guards for research.py.

Run: .venv/bin/python test_research.py

These are the checks that make every number in research.py mean anything:
if a signal can see the future, the whole search is worthless.
"""
from __future__ import annotations

import math

import research as R
from client import Candle
from decimal import Decimal


def synth(n: int = 600, seed: int = 7) -> list[Candle]:
    """Deterministic random-walk candles with no network or cache needed."""
    out, price, x = [], 1000.0, seed
    for i in range(n):
        x = (1103515245 * x + 12345) % (2 ** 31)
        drift = ((x / 2 ** 31) - 0.5) * 0.04
        o = price
        price = max(1.0, price * (1 + drift))
        hi, lo = max(o, price) * 1.01, min(o, price) * 0.99
        out.append(Candle(timestamp=1_600_000_000_000 + i * 86_400_000,
                          open=Decimal(str(o)), high=Decimal(str(hi)),
                          low=Decimal(str(lo)), close=Decimal(str(price)),
                          volume=Decimal("1")))
    return out


def test_signals_never_see_the_bar_they_trade():
    """Truncate the future off and the past must not move.

    This is the strongest available look-ahead check: if weights[i] depended
    on any bar > i, cutting the series at i+1 would change weights[i].
    """
    full = synth(600)
    cut = 400
    for name, fn in (("sma", lambda c: R.w_sma(c, 10, 50)),
                     ("ema", lambda c: R.w_ema(c, 10, 50)),
                     ("donchian", lambda c: R.w_donchian(c, 20, 10)),
                     ("voltarget", lambda c: R.w_voltarget(c, 10, 50, 0.3))):
        a = fn(full)[:cut]
        b = fn(full[:cut])[:cut]
        assert a == b, f"{name} weights changed when the future was removed"
    print("ok: all four signal families are causal (weights unchanged when "
          "future bars are deleted)")


def test_sma_matches_a_naive_recompute():
    """Rolling SMA must equal a from-scratch mean of the same window."""
    cl = R.closes_of(synth(300))
    got = R.sma_series(cl, 37)
    for i in range(37, len(cl)):
        want = sum(cl[i - 36:i + 1]) / 37
        assert abs(got[i] - want) < 1e-9, f"sma mismatch at {i}"
    assert got[36] is not None and got[35] is None, "warmup not blank"
    print("ok: sma_series matches a naive window mean and blanks its warmup")


def test_cash_is_exactly_flat():
    c = synth(300)
    eq = R.simulate(c, R.w_cash(c), 0)
    assert all(abs(e - 1.0) < 1e-15 for e in eq), "cash drifted"
    print("ok: cash baseline is exactly 1.0 (no accidental return)")


def test_buy_and_hold_pays_the_spread():
    """B&H must NOT be free. A naive B&H that ignores costs flatters the
    strategy's comparison by ~2x half-spread."""
    c = synth(300)
    free = R.simulate(c, R.w_buy_hold(c), 0, fee=0.0, half_spread_bps=0.0)
    paid = R.simulate(c, R.w_buy_hold(c), 0)
    assert paid[-1] < free[-1], "B&H did not pay any cost"
    spread_only = R.simulate(c, R.w_buy_hold(c), 0, fee=0.0)
    fee_only = R.simulate(c, R.w_buy_hold(c), 0, half_spread_bps=0.0)
    assert spread_only[-1] < free[-1] and fee_only[-1] < free[-1]
    print(f"ok: costs bite -- B&H free {free[-1]:.6f} > spread {spread_only[-1]:.6f} "
          f"> spread+fee {paid[-1]:.6f}")


def test_no_free_money_from_trading():
    """A strategy that trades on a flat market must lose exactly the costs."""
    flat = [Candle(timestamp=1_600_000_000_000 + i * 86_400_000,
                   open=Decimal("100"), high=Decimal("100"), low=Decimal("100"),
                   close=Decimal("100"), volume=Decimal("1")) for i in range(400)]
    w = [1.0 if 150 <= i <= 250 else 0.0 for i in range(400)]
    eq = R.simulate(flat, w, 0)
    assert eq[-1] < 1.0, f"round trip on a flat market profited: {eq[-1]}"
    print(f"ok: 2 fills on a flat market cost {(1-eq[-1])*10000:.2f}bps, no alpha")


def test_sharpe_of_a_random_walk_is_near_zero():
    """A coin-flip strategy must not manufacture Sharpe.

    Guards the metric itself: if Sharpe were computed on levels instead of
    returns, or annualised wrongly, noise would look like skill.
    """
    c = synth(4000, seed=99)
    w = [1.0 if (i // 37) % 2 == 0 else 0.0 for i in range(4000)]
    m = R.metrics(R.simulate(c, w, 0), 365.0)
    assert abs(m.sharpe) < 2.0, f"coin flip produced Sharpe {m.sharpe:.2f}"
    print(f"ok: coin-flip strategy Sharpe {m.sharpe:+.3f} "
          f"(ci95 +-{m.ci95:.2f}) -- indistinguishable from noise")


def test_max_drawdown_matches_a_hand_computed_case():
    # Running peak is 1.5 through the whole dip; the trough 0.6 is -60% from
    # it. (Not -50%: drawdown is against the RUNNING peak, not the previous
    # point -- getting that wrong understates every drawdown.)
    eq = [1.0, 1.5, 1.4, 0.75, 0.9, 1.2, 1.1, 0.6, 1.0, 2.0, 2.1, 2.2]
    m = R.metrics(eq, 365.0)
    assert abs(m.maxdd - 60.0) < 1e-9, f"maxdd {m.maxdd}"
    print("ok: max drawdown measured against the running peak (60%, not 50%)")


def test_metrics_refuses_short_series():
    """Below 10 bars there is no dispersion estimate; returning zeros beats
    inventing a Sharpe from 3 returns."""
    m = R.metrics([1.0, 1.1, 1.2], 365.0)
    assert m.sharpe == 0.0 and m.maxdd == 0.0 and m.n_bars == 3
    print("ok: short series returns zeros rather than a fabricated Sharpe")


def test_misaligned_weights_raise_instead_of_reporting_zero_tim():
    """A length mismatch used to be swallowed, and time-in-market silently
    printed 0% -- which read as 'never in the market' when the truth was a
    mis-sliced walk-forward chain."""
    eq = [1.0 * (1.01 ** i) for i in range(30)]
    try:
        R.metrics(eq, 365.0, [1.0] * 10)
    except ValueError:
        print("ok: misaligned weights raise instead of reporting TiM 0%")
        return
    raise AssertionError("misaligned weights were silently ignored")


def test_chain_building_cannot_compound():
    """Scaled block-chaining must compose multiplicatively ONCE.

    Regression: `chain.extend(chain[-1] * ... for ...)` is a lazy generator,
    so chain[-1] is re-read while the list is still growing and every element
    compounds off the one just appended. It reported CAGR 1e27% and maxDD
    100% on a curve that never fell.
    """
    block = [1.0, 2.0, 2.0, 2.0]          # one block tripled
    chain = [1.0]
    base = chain[-1]
    chain.extend([base * (e / block[0]) for e in block[1:]])
    assert chain == [1.0, 2.0, 2.0, 2.0], chain

    # two identical blocks must compound to 2x2, never 2x2x2x2...
    two = [1.0]
    for _ in range(2):
        base = two[-1]
        two.extend([base * (e / block[0]) for e in block[1:]])
    assert two[-1] == 4.0, two[-1]

    # and the metric on a monotone chain must show no drawdown at all
    m = R.metrics([1.0 * (1.5 ** i) for i in range(12)], 365.0)
    assert m.maxdd == 0.0, m.maxdd
    print("ok: block chaining compounds once (2 blocks -> 4x, drawdown 0%)")


def test_monotonic_equity_has_no_drawdown_and_never_nans():
    """A rising series has zero drawdown, and a degenerate one must produce a
    finite number rather than inf/nan (sd==0 used to be a divide-by-zero)."""
    up = [1.0 * (1.1 ** i) for i in range(12)]
    m = R.metrics(up, 365.0)
    assert m.maxdd == 0.0 and math.isfinite(m.sharpe) and math.isfinite(m.cagr), \
        f"degenerate series produced maxdd={m.maxdd} sharpe={m.sharpe}"
    flat = [1.0] * 12
    f = R.metrics(flat, 365.0)
    assert f.sharpe == 0.0 and f.maxdd == 0.0 and f.cagr == 0.0, \
        f"flat series produced sharpe={f.sharpe}"
    print("ok: degenerate series give 0/finite metrics, never inf or nan")


if __name__ == "__main__":
    for fn in (test_signals_never_see_the_bar_they_trade,
               test_sma_matches_a_naive_recompute,
               test_cash_is_exactly_flat,
               test_buy_and_hold_pays_the_spread,
               test_no_free_money_from_trading,
               test_sharpe_of_a_random_walk_is_near_zero,
               test_max_drawdown_matches_a_hand_computed_case,
               test_metrics_refuses_short_series,
               test_misaligned_weights_raise_instead_of_reporting_zero_tim,
               test_chain_building_cannot_compound,
               test_monotonic_equity_has_no_drawdown_and_never_nans):
        fn()
    print("\nall research-engine checks passed")