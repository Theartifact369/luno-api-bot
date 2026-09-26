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
from indicators import compute_rsi
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


def resample_daily(candles: list[Candle]) -> list[Candle]:
    days: list[Candle] = []
    for c in candles:
        day = c.timestamp // 86400000
        if days and days[-1].timestamp // 86400000 == day:
            d = days[-1]
            d.high = max(d.high, c.high)
            d.low = min(d.low, c.low)
            d.close = c.close
            d.volume += c.volume
        else:
            days.append(Candle(timestamp=day * 86400000, open=c.open, high=c.high,
                               low=c.low, close=c.close, volume=c.volume))
    return days


def cagr_pct(start_val: Decimal, end_val: Decimal, years: float) -> float:
    if years <= 0 or start_val <= 0:
        return 0.0
    return (float(end_val / start_val) ** (1 / years) - 1) * 100


def run_backtest(candles: list[Candle], config: Config, dca: bool = False, trend: bool = False,
                 dc_entry: int = 0, dc_exit: int = 0, dc_size: float = 0.95,
                 sma_regime: bool = False, sma_fast: int = 50, sma_slow: int = 200,
                 sma_break_exit: bool = False, mean_rev: bool = False,
                 mr_oversold: int = 30, mr_exit: int = 55, mr_max_hours: int = 168) -> None:
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
    mr_entry_i = 0

    for i in range(start_i, len(candles)):
        c = candles[i]
        day = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).date().isoformat()
        if day != last_day:
            last_day = day
            trader.reset_daily_counter(day)
            risk.reset_daily(trader.total_value(c.close), day)

        ticker = Ticker(pair=config.trading_pair, ask=c.close, bid=c.close,
                        last_trade=c.close, timestamp=c.timestamp, status="ACTIVE")
        ctx = strategy.build_context(ticker, candles[max(0, i - WARMUP):i + 1])

        signal = None
        if not dca:
            signal = strategy._rules_signal(ctx)

        trader.update_open_position_value(c.close)
        total = trader.total_value(c.close)
        if total > trader.state.peak_portfolio_value:
            trader.state.peak_portfolio_value = total

        risk_status = risk.check_circuit_breakers(
            total,
            trader.state.peak_portfolio_value,
            trader.state.consecutive_losses,
            trader.state.trades_today,
            gp.max_daily_trades,
        )

        if dca:
            # weekly fixed-ZAR buys, fully deployed over the period, no signals
            budget = INITIAL / Decimal(str((len(candles) - start_i) // (7 * 24)))
            if (i - start_i) % (7 * 24) == 0 and trader.state.balance_counter >= budget:
                trader.buy(c.close, budget / c.close, at_ms=c.timestamp)
        elif sma_regime:
            # Regime filter: fully invested while SMA_fast > SMA_slow, cash on
            # death cross. Optionally exit earlier when price breaks the fast
            # line (--sma-break-exit) to cut crash drawdowns.
            closes = [x.close for x in candles[max(0, i - sma_slow):i + 1]]
            if len(closes) >= sma_slow:
                sma_fast_ = sum(closes[-sma_fast:]) / Decimal(str(sma_fast))
                sma_slow_ = sum(closes) / Decimal(str(len(closes)))
                in_pos = trader.state.open_position is not None
                if not in_pos and sma_fast_ > sma_slow_:
                    trader.buy(c.close, trader.state.balance_counter / c.close, at_ms=c.timestamp)
                elif in_pos and (sma_fast_ < sma_slow_
                                 or (sma_break_exit and c.close < sma_fast_)):
                    reason = "trend_break" if sma_break_exit and c.close < sma_fast_ else "death_cross"
                    trader.sell(c.close, trader.state.balance_base, reason=reason, at_ms=c.timestamp)
        elif dc_entry:
            # Donchian trend-following: long on N-bar breakout, exit on M-bar low.
            # No profit cap -> winners run; channel exit IS the trailing stop.
            in_pos = trader.state.open_position is not None
            if not in_pos and i >= dc_entry and risk_status.trading_allowed:
                channel_hi = max(c.high for c in candles[i - dc_entry:i])
                if c.close > channel_hi:
                    vol = trader.state.balance_counter * Decimal(str(dc_size)) / c.close
                    if trader.buy(c.close, vol, at_ms=c.timestamp):
                        trader.state.trades_today += 1
            elif in_pos and i >= dc_exit:
                channel_lo = min(c.low for c in candles[i - dc_exit:i])
                if c.close < channel_lo:
                    if trader.sell(c.close, trader.state.balance_base, reason="donchian_exit", at_ms=c.timestamp):
                        trader.state.trades_today += 1
        elif mean_rev:
            # Fade RSI extremes: buy oversold, exit on recovery to mr_exit or
            # mr_max_hours time-stop. Full balance per position, fees included.
            rsi = compute_rsi([x.close for x in candles[max(0, i - 14):i + 1]], 14)
            in_pos = trader.state.open_position is not None
            if not in_pos and rsi is not None and rsi <= Decimal(str(mr_oversold)) and risk_status.trading_allowed:
                if trader.buy(c.close, trader.state.balance_counter / c.close, at_ms=c.timestamp):
                    trader.state.trades_today += 1
                    mr_entry_i = i
            elif in_pos and rsi is not None and (rsi >= Decimal(str(mr_exit)) or i - mr_entry_i >= mr_max_hours):
                reason = "rsi_recovery" if rsi >= Decimal(str(mr_exit)) else "time_stop"
                if trader.sell(c.close, trader.state.balance_base, reason=reason, at_ms=c.timestamp):
                    trader.state.trades_today += 1
        elif risk_status.trading_allowed and signal is not None:
            if signal.action == "BUY" and trader.state.trades_today < gp.max_daily_trades:
                if trend and (ctx.sma_long is None or c.close <= ctx.sma_long):
                    pass
                else:
                    size_pct = risk.compute_position_size(trader.state.balance_counter, c.close, ctx.atr)
                    vol = trader.state.balance_counter * size_pct / c.close
                    if trader.buy(c.close, vol, at_ms=c.timestamp,
                                  reason=signal.family or signal.reason):
                        trader.state.trades_today += 1
            elif signal.action == "SELL" and trader.state.open_position is not None:
                pos_vol = Decimal(trader.state.open_position["volume"])
                if trader.sell(c.close, pos_vol * Decimal(str(gp.sell_fraction_pct)), reason=signal.reason, at_ms=c.timestamp):
                    trader.state.trades_today += 1

        equity_curve.append(trader.total_value(c.close))
        prev_portfolio_value = trader.portfolio_value

    end = trader.total_value(candles[-1].close)
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

    hold_curve = [INITIAL * c.close / first_close for c in candles[start_i:]]
    h_peak, h_max_dd = INITIAL, Decimal("0")
    for e in hold_curve:
        h_peak = max(h_peak, e)
        h_max_dd = max(h_max_dd, (h_peak - e) / h_peak * 100)

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
    print(f"  Max drawdown (equity): {max_dd:.1f}% (buy & hold: {h_max_dd:.1f}%)  "
          f"Sharpe (trade-rate annualised, {metrics.total_trades} obs - not significant): {metrics.sharpe_ratio:.2f}  "
          f"Net BTC: {trader.state.balance_base + trader.state.vault_base:+.8f}")
    print(f"  Vault: {trader.state.vault_base:.8f} BTC (R{trader.state.vault_deposits:,.2f} swept)")
    print(f"  Fees: R{trader.state.total_fees:,.2f}")
    verdict = "BEATS buy-and-hold" if bot_return > hold_return else "LOSES to buy-and-hold"
    print(f"\nVERDICT: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest XBTZAR strategy vs buy-and-hold")
    ap.add_argument("--refresh", action="store_true", help="re-download candle data")
    ap.add_argument("--days", type=int, default=0, help="only last N days (0 = all)")
    ap.add_argument("--dca", action="store_true", help="weekly fixed buys, no signals")
    ap.add_argument("--trend", action="store_true", help="only buy dips when price above long SMA")
    ap.add_argument("--daily", action="store_true", help="resample hourly candles to daily bars")
    ap.add_argument("--sma-regime", action="store_true", help="SMA regime: hold while golden cross")
    ap.add_argument("--sma-fast", type=int, default=50, help="SMA regime fast period")
    ap.add_argument("--sma-slow", type=int, default=200, help="SMA regime slow period")
    ap.add_argument("--sma-break-exit", action="store_true",
                    help="SMA regime: exit when close < fast SMA (not just death cross)")
    ap.add_argument("--mean-rev", action="store_true", help="mean-reversion: buy RSI oversold, sell on recovery")
    ap.add_argument("--mr-oversold", type=int, default=30, help="mean-rev RSI entry threshold")
    ap.add_argument("--mr-exit", type=int, default=55, help="mean-rev RSI exit threshold")
    ap.add_argument("--mr-max-hours", type=int, default=168, help="mean-rev time-stop hours")
    ap.add_argument("--dc-entry", type=int, default=0, help="Donchian entry channel bars (0=off)")
    ap.add_argument("--dc-exit", type=int, default=48, help="Donchian exit channel bars")
    ap.add_argument("--dc-size", type=float, default=0.95, help="Donchian position size fraction")
    args = ap.parse_args()

    config = Config.from_env()
    candles = fetch_candles(config.trading_pair, config.candle_duration, args.refresh, config)
    if args.days:
        cutoff = (datetime.now(tz=timezone.utc).timestamp() - args.days * 86400) * 1000
        candles = [c for c in candles if c.timestamp >= cutoff]
    if args.daily:
        candles = resample_daily(candles)
    run_backtest(candles, config, dca=args.dca, trend=args.trend,
                 dc_entry=args.dc_entry, dc_exit=args.dc_exit, dc_size=args.dc_size,
                 sma_regime=args.sma_regime, sma_fast=args.sma_fast, sma_slow=args.sma_slow,
                 sma_break_exit=args.sma_break_exit, mean_rev=args.mean_rev,
                 mr_oversold=args.mr_oversold, mr_exit=args.mr_exit,
                 mr_max_hours=args.mr_max_hours)


if __name__ == "__main__":
    main()
