"""Regression checks for the backtest harness fixes.

Run:  .venv/bin/python test_backtest_harness.py
Each one fails if the defect it guards is reintroduced.
"""
import inspect
from decimal import Decimal

import backtest
from backtest import resample_daily
from client import Candle


def test_cache_is_keyed_by_pair_and_duration():
    """A flat candles_backtest.json served XBTZAR 1h bars to any request."""
    a = backtest._cache_path("XBTZAR", 3600)
    b = backtest._cache_path("XBTZAR", 86400)
    c = backtest._cache_path("ETHZAR", 3600)
    assert a != b and a != c and b != c, "cache collision across pair/duration"
    print("ok: cache keyed by pair+duration ->", a.name, b.name, c.name)


def test_fetch_requests_from_2010_not_2021():
    """START was pinned to 2021, discarding 8 years of XBTZAR daily history."""
    assert backtest.EARLIEST < 1_300_000_000_000, "fetch must start before 2021"
    src = inspect.getsource(backtest.fetch_candles)
    assert "2021" not in src, "fetch_candles still hardcodes a 2021 start"
    print("ok: fetch starts", backtest.EARLIEST, "(2010)")


def test_fetch_loop_cannot_spin_forever():
    """An empty/duplicate page used to re-read the same 1000 bars forever."""
    src = inspect.getsource(backtest.fetch_candles)
    assert "nxt <= since" in src, "missing no-progress guard in pagination loop"
    print("ok: pagination has a no-progress break")


def test_sma_regime_is_all_in_by_default():
    """Volatility scaling capped entries at 15% of cash while the docstring
    said 'all-in', which is what made the crossover look worthless."""
    sig = inspect.signature(backtest.run_backtest)
    assert sig.parameters["sma_size"].default == 1.0, "default must be all-in"
    src = inspect.getsource(backtest.run_backtest)
    sma_branch = src.split("elif sma_regime:")[1].split("elif dc_entry:")[0]
    assert "compute_position_size" not in sma_branch, (
        "SMA regime branch still uses volatility-capped sizing")
    assert "balance_counter / c.close" in sma_branch, "expected an all-in entry"
    print("ok: sma_regime deploys 100% of cash by default")


def test_fills_cross_the_spread():
    """ask == bid == close made every backtest fill free."""
    src = inspect.getsource(backtest.run_backtest)
    assert "spread_bps" in src, "spread parameter missing"
    assert "ask=c.close * (1 + half)" in src, "ask must sit above the close"
    assert "bid=c.close * (1 - half)" in src, "bid must sit below the close"
    print("ok: ticker ask/bid straddle the close")


def test_risk_manager_uses_config():
    """RiskManager() with defaults meant the backtest never tested the
    configured risk limits."""
    src = inspect.getsource(backtest.run_backtest)
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    assert "RiskManager()" not in code, "bare RiskManager() ignores config"
    for field in ("max_daily_loss_pct=config", "max_drawdown_pct=config",
                  "max_consecutive_losses=config"):
        assert field in src, f"config not wired: {field}"
    print("ok: risk limits come from Config")


def test_resample_daily_is_idempotent_on_native_daily_bars():
    """--candles 86400 already returns daily bars; resampling again is a no-op
    only if the function doesn't collapse or duplicate bars."""
    day = 86400000
    base = 1_600_000_000_000 // day * day
    bars = [Candle(timestamp=base + i * day, open=Decimal("1"), high=Decimal("2"),
                   low=Decimal("0.5"), close=Decimal("1.5"), volume=Decimal("10"))
            for i in range(10)]
    out = resample_daily(bars)
    assert len(out) == 10, f"10 daily bars collapsed to {len(out)}"
    assert [c.timestamp for c in out] == [c.timestamp for c in bars], "timestamps drifted"
    assert all(c.high == Decimal("2") and c.low == Decimal("0.5") for c in out), "OHLC mangled"
    print("ok: resample_daily is a no-op on native daily bars")


def test_no_lookahead_in_sma_window():
    """The slow SMA window ended at i+1 while entering at close of i, so the
    signal used the same bar it traded on."""
    src = inspect.getsource(backtest.run_backtest)
    sma_branch = src.split("elif sma_regime:")[1].split("elif dc_entry:")[0]
    assert "closes[-sma_slow:]" in sma_branch or "i - sma_slow + 1" in sma_branch, (
        "slow SMA window still includes the bar being traded")


if __name__ == "__main__":
    for fn in (test_cache_is_keyed_by_pair_and_duration,
               test_fetch_requests_from_2010_not_2021,
               test_fetch_loop_cannot_spin_forever,
               test_sma_regime_is_all_in_by_default,
               test_fills_cross_the_spread,
               test_risk_manager_uses_config,
               test_resample_daily_is_idempotent_on_native_daily_bars,
               test_no_lookahead_in_sma_window):
        fn()
    print("\nall harness checks passed")
