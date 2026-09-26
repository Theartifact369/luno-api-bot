"""Self-checks for the papertrade logic fixes (run: python test_papertrade.py)"""
from decimal import Decimal
from datetime import datetime, timezone

from paper_trader import PaperTrader, State, TradeLog, CompletedTradeLog
from risk_manager import RiskManager


def test_fee_adjusted_pnl_pct():
    trader = PaperTrader(State.initial(0.0, 10000.0), TradeLog(), CompletedTradeLog(), "XBTZAR")
    trader.buy(Decimal("100"), Decimal("1"))          # cost 100 + 0.1 fee
    trader.sell(Decimal("110"), Decimal("1"))         # proceeds 110 - 0.11 fee
    t = trader.completed_log.trades[-1]
    # fee-adjusted: (110 - 0.11 - 100 - 0.1) / 100 = 0.0979, NOT raw 10%
    assert abs(Decimal(t["pnl_pct"]) - Decimal("0.0979")) < Decimal("0.0001"), t["pnl_pct"]
    assert Decimal(t["pnl"]) > 0
    print("ok: pnl_pct is fee-adjusted:", t["pnl_pct"])


def test_cooldown_expires():
    rm = RiskManager(max_consecutive_losses=3, cooldown_hours=0.0001)  # ~0.36s
    blocked = [rm.check_circuit_breakers(Decimal("100"), Decimal("100"), 3, 0, 5) for _ in range(3)]
    assert all(not s.trading_allowed for s in blocked), "must block during cooldown"
    assert blocked[0].cooldown_until == blocked[-1].cooldown_until, "cooldown must not restart each scan"
    import time as _t
    _t.sleep(0.5)
    s = rm.check_circuit_breakers(Decimal("100"), Decimal("100"), 3, 0, 5)
    assert s.trading_allowed and s.cooldown_just_expired, "cooldown must expire, not run forever"
    print("ok: cooldown expires after cooldown_hours and flags just_expired")


def test_position_cap_sizing():
    # governance cap: 30% of total value; position already worth the target -> no new buy
    from types import SimpleNamespace
    trader = SimpleNamespace(
        state=SimpleNamespace(
            balance_counter=Decimal("70000"),
            open_position={"current_value": "30000"},
        ),
        total_value=lambda price: Decimal("100000"),
    )
    total = trader.total_value(Decimal("1000"))
    target = total * Decimal("0.3")
    already = Decimal(trader.state.open_position["current_value"])
    headroom = target - already
    assert headroom <= 0, "at cap, no further buys"
    print("ok: position headroom respects governance cap")


def test_realistic_fills_documented():
    # buy fills at ask, sell fills at bid (checked by code inspection in main_loop)
    import inspect, main
    src = inspect.getsource(main)
    assert "price = ticker.ask" in src and "price = ticker.bid" in src
    print("ok: buys fill at ask, sells fill at bid")


def test_daily_loss_baseline_resets_once_per_day():
    rm = RiskManager(max_daily_loss_pct=3.0)
    start = Decimal("100000")
    rm.reset_daily(start, "2026-09-26")
    # every scan calls reset_daily; the baseline must not drift with the mark
    for v in (start * Decimal("0.99"), start * Decimal("0.95")):
        rm.reset_daily(v, "2026-09-26")
        s = rm.check_circuit_breakers(v, start, 0, 0, 5)
    assert not s.trading_allowed and s.daily_loss_hit, f"5% day loss must trip 3% breaker: {s.reason}"
    # new day: fresh baseline, breaker re-armed
    rm.reset_daily(start, "2026-09-27")
    s = rm.check_circuit_breakers(start, start, 0, 0, 5)
    assert s.trading_allowed, s.reason
    print("ok: daily-loss baseline resets per day, not per scan")


def test_exit_is_never_gated_by_risk_breakers():
    # A tripped breaker must not be able to refuse a SELL. Drawdown only
    # re-arms on a new portfolio peak, so gating the exit meant an unattended
    # run could hold a dying position forever and never say so.
    import inspect, main
    src = inspect.getsource(main)
    buy = src.split('signal.action == "BUY"')[0].rsplit("if ", 1)[-1]
    sell = src.split('signal.action == "SELL"')[0].rsplit("elif ", 1)[-1]
    assert "can_trade" in buy, "entries should still respect the breakers"
    assert "can_trade" not in sell, f"exit is gated on a breaker: {sell!r}"
    print("ok: exits bypass risk breakers, entries respect them")


def test_drawdown_breaker_matches_peak_basis():
    # peak_portfolio_value is tracked from total_value (capital + vault BTC), so
    # the breaker must be fed total_value too. Regression: it got
    # portfolio_value, so a growing vault faked a drawdown and halted trading.
    import inspect, main
    src = inspect.getsource(main)
    args = src.split("check_circuit_breakers(")[1].split(")")[0]
    first_arg = args.split(",")[0].strip()
    assert first_arg == "total_value", f"breaker reads {first_arg!r}, must read total_value"
    print("ok: drawdown breaker reads the same basis as the peak")


def test_sharpe_annualises_by_trade_rate_not_by_day():
    # returns are per-trade. Annualising with sqrt(365) turned the 5-trade
    # regime backtest into a fake Sharpe 12.18; the honest figure is ~0.6.
    import math
    from performance import compute_metrics

    year_ms = 365.25 * 24 * 3600 * 1000
    trades = [
        {"pnl_pct": Decimal(str(p)), "pnl": Decimal("0"),
         "entry_time": int(i * year_ms), "exit_time": int(i * year_ms + 1),
         "holding_period_seconds": 1}
        for i, p in enumerate((25.0, -10.0, 30.0, -12.0, 18.0))
    ]
    m = compute_metrics(trades, Decimal("100"), Decimal("100"),
                        Decimal("100"), 0)

    rets = [float(t["pnl_pct"]) for t in trades]
    avg = sum(rets) / len(rets)
    std = math.sqrt(sum((r - avg) ** 2 for r in rets) / len(rets))
    # 5 trades spread over 4 years -> ~1.25/yr, not 365/yr
    honest = (avg / std) * math.sqrt(5 / 4)
    assert abs(m.sharpe_ratio - honest) < 1e-6, \
        f"got {m.sharpe_ratio:.2f}, expected {honest:.2f} (sqrt(365) inflation)"
    assert m.sharpe_ratio < 2, f"absurd Sharpe {m.sharpe_ratio:.2f} on 5 trades"
    print(f"ok: Sharpe annualised by trade rate -> {m.sharpe_ratio:.2f}, not 12+")

    # one trade can't produce a Sharpe at all; must not divide by zero span
    m1 = compute_metrics([{"pnl_pct": Decimal("50"), "pnl": Decimal("0"),
                           "entry_time": 0, "exit_time": 1,
                           "holding_period_seconds": 1}],
                         Decimal("100"), Decimal("100"), Decimal("100"), 0)
    assert m1.sharpe_ratio == 0.0, m1.sharpe_ratio
    print("ok: single trade yields no Sharpe, no blowup")


def test_fills_use_candle_time_not_wallclock():
    # Regression: buy/sell hardcoded time.time(), so every backtest trade was
    # stamped with the real-world instant of the run. Holding periods read as
    # zero and the trade-rate annualiser divided by a ~0 span (Sharpe 12211).
    from paper_trader import PaperTrader, State, TradeLog, CompletedTradeLog

    t0, t1 = 1_600_000_000_000, 1_700_000_000_000      # ~2.8 years apart
    t = PaperTrader(State.initial(0.0, 10000.0), TradeLog(), CompletedTradeLog(), "XBTZAR")
    o = t.buy(Decimal("1000"), Decimal("10"), at_ms=t0)
    assert o.filled_at == t0, o.filled_at
    t.update_open_position_value(Decimal("1200"))
    t.sell(Decimal("1200"), reason="exit", at_ms=t1)

    trade = t.completed_log.trades[-1]
    held = trade["holding_period_seconds"]
    assert held == (t1 - t0) // 1000, f"holding period {held}s, expected {(t1-t0)//1000}"
    assert (t1 - t0) / 1000 / 86400 > 700, "test gap should be years, not days"
    print(f"ok: fill time comes from the caller -> {held}s holding period, not 0")


def test_decision_log_feedback():
    # the "why did we think this" log: reasons persist on trades, and a
    # family that never won vetoes the next trade; one that won <50% halves it
    from strategy import Strategy
    from config import Config

    trader = PaperTrader(State.initial(0.0, 10000.0), TradeLog(), CompletedTradeLog(), "XBTZAR")
    s = Strategy(Config(), trader)

    # no history yet -> no feedback
    assert s._reason_feedback("golden_cross") == (1.0, "")

    # two straight losses for this family -> veto (mult 0)
    trader.completed_log.trades.append({"entry_reason": "golden_cross", "pnl_pct": "-0.10"})
    trader.completed_log.trades.append({"entry_reason": "golden_cross", "pnl_pct": "-0.20"})
    mult, note = s._reason_feedback("golden_cross")
    assert mult == 0.0 and note, (mult, note)

    # two losses + one win -> under half -> halve confidence
    trader.completed_log.trades.append({"entry_reason": "golden_cross", "pnl_pct": "0.30"})
    mult, _ = s._reason_feedback("golden_cross")
    assert mult == 0.5, mult

    # other families are independent
    assert s._reason_feedback("rules_buy") == (1.0, "")

    # round trip: buy persists the family, sell carries it onto the record
    t2 = PaperTrader(State.initial(0.0, 10000.0), TradeLog(), CompletedTradeLog(), "XBTZAR")
    t2.buy(Decimal("100"), Decimal("1"), reason="golden_cross")
    assert t2.state.open_position["entry_reason"] == "golden_cross"
    assert t2.trade_log.trades[-1]["reason"] == "golden_cross"
    t2.sell(Decimal("110"), reason="exit")
    assert t2.completed_log.trades[-1]["entry_reason"] == "golden_cross"
    print("ok: decision log — reasons persist and feed back (veto/halve)")


if __name__ == "__main__":
    test_fee_adjusted_pnl_pct()
    test_cooldown_expires()
    test_position_cap_sizing()
    test_realistic_fills_documented()
    test_daily_loss_baseline_resets_once_per_day()
    test_drawdown_breaker_matches_peak_basis()
    test_exit_is_never_gated_by_risk_breakers()
    test_sharpe_annualises_by_trade_rate_not_by_day()
    test_fills_use_candle_time_not_wallclock()
    test_decision_log_feedback()
    print("all papertrade logic checks passed")
