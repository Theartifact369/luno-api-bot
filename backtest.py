from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from client import Candle, LunoClient, Ticker
from config import Config
from governance import GovernanceParams
from paper_trader import CompletedTradeLog, PaperTrader, State, TradeLog
from performance import compute_metrics
from risk_manager import RiskManager
from strategy import Strategy

CACHE = Path("candles_backtest.json")
START = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
WARMUP = 200
INITIAL = Decimal("100000")


def fetch_candles(pair: str, duration: int, refresh: bool, config: Config) -> list[Candle]:
    if CACHE.exists() and not refresh:
        return [Candle(**{**c, "close": Decimal(c["close"]), "open": Decimal(c["open"]),
                          "high": Decimal(c["high"]), "low": Decimal(c["low"]), "volume": Decimal(c["volume"])})
                for c in json.loads(CACHE.read_text())]

    client = LunoClient(
        base_url=config.luno_api_base,
        api_key_id=config.luno_api_key_id,
        api_key_secret=config.luno_api_secret,
    )
    seen: dict[int, Candle] = {}
    since = int(START)
    while since < datetime.now(tz=timezone.utc).timestamp() * 1000:
        chunk = client.get_candles(pair, duration, since)
        if not chunk:
            break
        for c in chunk:
            seen[c.timestamp] = c
        since = max(c.timestamp for c in chunk) + duration * 1000
        print(f"fetched {len(seen)} candles...", end="\r", flush=True)
    client.close()

    candles = sorted(seen.values(), key=lambda c: c.timestamp)
    CACHE.write_text(json.dumps([c.__dict__ for c in candles], default=str))
    print(f"\n{len(candles)} candles cached to {CACHE}")
    return candles


def cagr_pct(start_val: Decimal, end_val: Decimal, years: float) -> float:
    if years <= 0 or start_val <= 0:
        return 0.0
    return (float(end_val / start_val) ** (1 / years) - 1) * 100


def run_backtest(candles: list[Candle], config: Config, dca: bool = False, trend: bool = False) -> None:
    if len(candles) < WARMUP + 2:
        print("Not enough candle data")
        return

    trader = PaperTrader(
        State.initial(0.0, float(INITIAL)),
        TradeLog(),
        CompletedTradeLog(),
        config.trading_pair,
    )
    strategy = Strategy(config, trader)
    gp = GovernanceParams()
    strategy.update_governance(gp)
    risk = RiskManager()

    start_i = WARMUP
    first_close = candles[start_i].close
    equity_curve: list[Decimal] = []
    last_day = ""
    prev_portfolio_value = INITIAL

    for i in range(start_i, len(candles)):
        c = candles[i]
        day = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).date().isoformat()
        if day != last_day:
            last_day = day
            trader.reset_daily_counter(day)
            risk.reset_daily(trader.portfolio_value)

        ticker = Ticker(pair=config.trading_pair, ask=c.close, bid=c.close,
                        last_trade=c.close, timestamp=c.timestamp, status="ACTIVE")
        ctx = strategy.build_context(ticker, candles[max(0, i - WARMUP):i + 1])

        signal = None
        if not dca:
            signal = strategy._rules_signal(ctx)

        trader.update_open_position_value(c.close)
        if trader.portfolio_value > trader.state.peak_portfolio_value:
            trader.state.peak_portfolio_value = trader.portfolio_value

        risk_status = risk.check_circuit_breakers(
            trader.portfolio_value,
            trader.state.peak_portfolio_value,
            trader.state.consecutive_losses,
            trader.state.trades_today,
            gp.max_daily_trades,
        )

        if dca:
            # weekly buy, fully deployed over the period, no signals
            if (i - start_i) % (7 * 24) == 0:
                trader.buy(c.close, trader.state.balance_counter / Decimal("1000"))
        elif risk_status.trading_allowed and signal is not None:
            if signal.action == "BUY" and trader.state.trades_today < gp.max_daily_trades:
                if trend and (ctx.sma_long is None or c.close <= ctx.sma_long):
                    pass
                else:
                    size_pct = risk.compute_position_size(trader.state.balance_counter, c.close, ctx.atr)
                    vol = trader.state.balance_counter * size_pct / c.close
                    if trader.buy(c.close, vol):
                        trader.state.trades_today += 1
            elif signal.action == "SELL" and trader.state.open_position is not None:
                pos_vol = Decimal(trader.state.open_position["volume"])
                if trader.sell(c.close, pos_vol * Decimal(str(gp.sell_fraction_pct)), reason=signal.reason):
                    trader.state.trades_today += 1

        equity_curve.append(trader.portfolio_value)
        prev_portfolio_value = trader.portfolio_value

    end = trader.portfolio_value
    bh_end = INITIAL * candles[-1].close / first_close
    years = (candles[-1].timestamp - candles[start_i].timestamp) / (365.25 * 24 * 3600 * 1000)

    metrics = compute_metrics(
        trader.completed_log.last_n(10 ** 9),
        end,
        trader.state.peak_portfolio_value,
        INITIAL,
        trader.state.consecutive_losses,
    )

    peak = INITIAL
    max_dd = Decimal("0")
    for e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)

    hold_return = (bh_end - INITIAL) / INITIAL * 100
    bot_return = (end - INITIAL) / INITIAL * 100

    print("\n=========== BACKTEST: XBTZAR 1h, rules engine (governor+kronos off) ===========")
    print(f"Period: {datetime.fromtimestamp(candles[start_i].timestamp / 1000, tz=timezone.utc).date()} "
          f"-> {datetime.fromtimestamp(candles[-1].timestamp / 1000, tz=timezone.utc).date()} ({years:.1f} yr)")
    print(f"Start price: R{first_close:,.0f}  End price: R{candles[-1].close:,.0f}")
    print(f"\n  Bot:        R{INITIAL:,.0f} -> R{end:,.0f}  ({bot_return:+.2f}%, {cagr_pct(INITIAL, end, years):+.2f}%/yr)")
    print(f"  Buy & hold: R{INITIAL:,.0f} -> R{bh_end:,.0f}  ({hold_return:+.2f}%, {cagr_pct(INITIAL, bh_end, years):+.2f}%/yr)")
    print(f"\n  Trades: {metrics.total_trades}  Win rate: {metrics.win_rate*100:.1f}%  "
          f"Profit factor: {metrics.profit_factor:.2f}")
    print(f"  Max drawdown (equity): {max_dd:.1f}%  Sharpe: {metrics.sharpe_ratio:.2f}  "
          f"Net BTC: {trader.state.balance_base:+.8f}")
    print(f"  Fees: R{trader.state.total_fees:,.2f}")
    verdict = "BEATS buy-and-hold" if bot_return > hold_return else "LOSES to buy-and-hold"
    print(f"\nVERDICT: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest XBTZAR strategy vs buy-and-hold")
    ap.add_argument("--refresh", action="store_true", help="re-download candle data")
    ap.add_argument("--days", type=int, default=0, help="only last N days (0 = all)")
    ap.add_argument("--dca", action="store_true", help="weekly fixed buys, no signals")
    ap.add_argument("--trend", action="store_true", help="only buy dips when price above long SMA")
    args = ap.parse_args()

    config = Config.from_env()
    candles = fetch_candles(config.trading_pair, config.candle_duration, args.refresh, config)
    if args.days:
        cutoff = (datetime.now(tz=timezone.utc).timestamp() - args.days * 86400) * 1000
        candles = [c for c in candles if c.timestamp >= cutoff]
    run_backtest(candles, config, dca=args.dca, trend=args.trend)


if __name__ == "__main__":
    main()
