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

CACHE_DIR = Path("candles")
WARMUP = 200
INITIAL = Decimal("100000")
# XBTZAR daily history on Luno starts 2013-04-13; 1h starts later. Requesting
# from 2010 costs one extra request and the API just returns what it has.
EARLIEST = datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def _cache_path(pair: str, duration: int) -> Path:
    # keyed by pair+duration: a flat candles_backtest.json silently served
    # XBTZAR 1h bars to a request for ETHZAR 1d bars
    return CACHE_DIR / f"{pair}_{duration}.json"


def fetch_candles(pair: str, duration: int, refresh: bool, config: Config,
                  start_ms: int = int(EARLIEST)) -> list[Candle]:
    cache = _cache_path(pair, duration)
    if cache.exists() and not refresh:
        return [Candle(**{**c, "close": Decimal(c["close"]), "open": Decimal(c["open"]),
                          "high": Decimal(c["high"]), "low": Decimal(c["low"]), "volume": Decimal(c["volume"])})
                for c in json.loads(cache.read_text())]

    client = LunoClient(
        base_url=config.luno_api_base,
        api_key_id=config.luno_api_key_id,
        api_key_secret=config.luno_api_secret,
    )
    seen: dict[int, Candle] = {}
    since = int(start_ms)
    now = datetime.now(tz=timezone.utc).timestamp() * 1000
    while since < now:
        chunk = client.get_candles(pair, duration, since)
        if not chunk:
            break
        for c in chunk:
            seen[c.timestamp] = c
        nxt = max(c.timestamp for c in chunk) + duration * 1000
        # Luno caps a response at 1000 bars. If a page ever returns nothing
        # newer than `since`, the old loop spun forever re-reading one page.
        if nxt <= since:
            print(f"\nno progress at {datetime.fromtimestamp(since/1000, tz=timezone.utc)}")
            break
        since = nxt
        print(f"fetched {len(seen)} candles...", end="\r", flush=True)
    client.close()

    candles = sorted(seen.values(), key=lambda c: c.timestamp)
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps([c.__dict__ for c in candles], default=str))
    print(f"\n{len(candles)} candles cached to {cache}")
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
                 mr_oversold: int = 30, mr_exit: int = 55, mr_max_hours: int = 168,
                 sma_size: float = 1.0, spread_bps: float = 5.0) -> None:
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
    # was RiskManager() with all defaults, ignoring the configured limits --
    # the backtest never tested the risk settings the bot actually runs
    risk = RiskManager(
        max_daily_loss_pct=config.risk_max_daily_loss_pct,
        max_drawdown_pct=config.risk_max_drawdown_pct,
        max_consecutive_losses=config.risk_max_consecutive_losses,
        cooldown_hours=config.risk_cooldown_hours,
        volatility_position_sizing=config.risk_volatility_sizing,
    )

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

        # Crossing the spread costs money. Live XBTZAR bid/ask is ~0.005%, but
        # ask==bid==close made every backtest fill free, flattering the edge.
        # half = spread charged on the side you trade.
        half = Decimal(str(spread_bps)) / Decimal("20000")
        ticker = Ticker(pair=config.trading_pair, ask=c.close * (1 + half),
                        bid=c.close * (1 - half),
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
            # Regime filter: invested while SMA_fast > SMA_slow, cash on
            # death cross. ALL-IN / ALL-OUT -- volatility scaling here was
            # capped at base_position_size_pct (30%), and with ATR/price >3%
            # on 66% of daily bars it halved to 15%, so this deployed ~10% of
            # capital while the docstring claimed "all-in". Sizing is a knob
            # (--sma-size); the default now matches the strategy's intent.
            closes = [x.close for x in candles[max(0, i - sma_slow + 1):i + 1]]
            if len(closes) >= sma_slow:
                sma_fast_ = sum(closes[-sma_fast:]) / Decimal(str(sma_fast))
                sma_slow_ = sum(closes) / Decimal(str(len(closes)))
                in_pos = trader.state.open_position is not None
                if not in_pos and sma_fast_ > sma_slow_:
                    if sma_size >= 1:
                        vol = trader.state.balance_counter / c.close
                    else:
                        vol = trader.state.balance_counter * Decimal(str(sma_size)) / c.close
                    trader.buy(c.close, vol, at_ms=c.timestamp, reason="golden_cross")
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

    # Strategy-only equity: the vault holds BTC forever, so total_value()
    # folds "I was long BTC in 2014" into what looks like strategy alpha.
    # Reported separately, never in the headline. See test_edge_finding.py.
    strat_end = trader.portfolio_value
    vault_end = trader.state.vault_base * candles[-1].close

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
    strat_return = (strat_end - INITIAL) / INITIAL * 100

    bar_h = (candles[1].timestamp - candles[0].timestamp) / 3600000 if len(candles) > 1 else 1
    bar_name = f"{bar_h:g}h" if bar_h < 48 else f"{bar_h/24:g}d"
    print(f"\n=========== BACKTEST: {config.trading_pair} {bar_name}, "
          f"rules engine (governor+kronos off) ===========")
    print(f"Spread: {spread_bps}bps per fill" + (f"  |  SMA size: {sma_size:.0%}" if sma_regime else ""))
    print(f"Period: {datetime.fromtimestamp(candles[start_i].timestamp / 1000, tz=timezone.utc).date()} "
          f"-> {datetime.fromtimestamp(candles[-1].timestamp / 1000, tz=timezone.utc).date()} ({years:.1f} yr)")
    print(f"Start price: R{first_close:,.0f}  End price: R{candles[-1].close:,.0f}")
    print(f"\n  Strategy only:  R{INITIAL:,.0f} -> R{strat_end:,.0f}  "
          f"({strat_return:+.2f}%, {cagr_pct(INITIAL, strat_end, years):+.2f}%/yr)  <- HEADLINE")
    print(f"  + vault (BTC): R{vault_end:,.0f}  ({bot_return - strat_return:+.2f}% of the "
          f"total, and that is just holding BTC since 2014)")
    print(f"  Buy & hold:     R{INITIAL:,.0f} -> R{bh_end:,.0f}  "
          f"({hold_return:+.2f}%, {cagr_pct(INITIAL, bh_end, years):+.2f}%/yr)")
    print(f"\n  Trades: {metrics.total_trades}  Win rate: {metrics.win_rate*100:.1f}%  "
          f"Profit factor: {metrics.profit_factor:.2f}")
    print(f"  Max drawdown (equity, incl. vault): {max_dd:.1f}% (buy & hold: {h_max_dd:.1f}%)")
    print(f"  Fees: R{trader.state.total_fees:,.2f}")
    print(f"  Vault: {trader.state.vault_base:.8f} BTC (R{trader.state.vault_deposits:,.2f} swept)")

    # A verdict has to carry its own sample size and refuse to call a result
    # significant when the trade count cannot support one. "BEATS buy-and-hold"
    # on 12 trades is how the +74343% headline happened.
    MIN_TRADES = 30
    n = metrics.total_trades
    beats = strat_return > hold_return
    if n < MIN_TRADES:
        verdict = ("NO VERDICT -- n=%d trades, below the %d needed to distinguish "
                   "skill from luck" % (n, MIN_TRADES))
    elif beats:
        verdict = f"BEATS buy-and-hold on strategy return alone (n={n} trades)"
    else:
        verdict = f"LOSES to buy-and-hold (n={n} trades)"
    print(f"\nVERDICT: {verdict}")
    print("NOTE: this is a single in-sample pass with parameters chosen by hand. "
          "It is not evidence of an edge.")
    print("      For the out-of-sample verdict see: .venv/bin/python research.py walk")


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
    ap.add_argument("--sma-size", type=float, default=1.0,
                    help="fraction of cash to deploy per entry (1.0 = all-in, the strategy's intent)")
    ap.add_argument("--mean-rev", action="store_true", help="mean-reversion: buy RSI oversold, sell on recovery")
    ap.add_argument("--mr-oversold", type=int, default=30, help="mean-rev RSI entry threshold")
    ap.add_argument("--mr-exit", type=int, default=55, help="mean-rev RSI exit threshold")
    ap.add_argument("--mr-max-hours", type=int, default=168, help="mean-rev time-stop hours")
    ap.add_argument("--dc-entry", type=int, default=0, help="Donchian entry channel bars (0=off)")
    ap.add_argument("--dc-exit", type=int, default=48, help="Donchian exit channel bars")
    ap.add_argument("--dc-size", type=float, default=0.95, help="Donchian position size fraction")
    ap.add_argument("--spread-bps", type=float, default=5.0,
                    help="half-spread charged per fill in bps (live XBTZAR ~0.5)")
    ap.add_argument("--candles", type=int, default=0,
                    help="fetch native candles of this duration in seconds (86400=daily). "
                         "Default uses config.candle_duration.")
    args = ap.parse_args()

    config = Config.from_env()
    duration = args.candles or config.candle_duration
    candles = fetch_candles(config.trading_pair, duration, args.refresh, config)
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
                 mr_max_hours=args.mr_max_hours, sma_size=args.sma_size,
                 spread_bps=args.spread_bps)


if __name__ == "__main__":
    main()
