from __future__ import annotations

import asyncio
import logging
import logging.handlers
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import plotext as plt

from client import Candle, LunoClient
from config import Config
from governance import GovernanceState, GovernorAgent
from paper_trader import PaperTrader, State, TradeLog, CompletedTradeLog
from performance import compute_metrics
from risk_manager import RiskManager, RiskStatus
from strategy import Strategy

logger = logging.getLogger(__name__)
console = Console()

log_configured = False


def setup_logging(config: Config) -> None:
    global log_configured
    if log_configured:
        return
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_dir / "trader.log"),
        when="midnight",
        backupCount=30,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(config.log_level.upper())
    root.addHandler(handler)

    if not config.headless:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(sh)

    log_configured = True


def chart_height(term_height: int) -> int:
    # reserve for ticker(3) + status(3) + ~2 table rows; shrink reserve on small terms
    reserve = 26 if term_height >= 40 else max(14, term_height // 2)
    return max(6, min(30, term_height - reserve))


def build_price_chart(candles: list[Candle], width: int = 80, height: int = 18) -> str:
    if len(candles) < 2:
        return "Collecting candles..."
    candles = candles[-48:]
    opens = [float(c.open) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    closes = [float(c.close) for c in candles]
    vols = [float(c.volume) for c in candles]
    labels = [time.strftime("%H:%M", time.gmtime(c.timestamp)) for c in candles]
    ticks = list(range(len(candles)))
    n = len(candles)
    tick_pos = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    import plotext._core as _pltcore

    _pltcore._figure._active = _pltcore._figure._master
    plt.clear_figure()
    plt.theme("dark")
    plt.limit_size(False, False)
    plt.plotsize(max(20, width), height)
    plt.subplots(2, 1)
    plt.title("Price (1h)")
    plt.candlestick(ticks, {"Open": opens, "High": highs, "Low": lows, "Close": closes})
    plt.xticks(tick_pos, [labels[i] for i in tick_pos])
    plt.subplot(2, 1)
    plt.title("Volume")
    plt.ylim(0, max(vols) * 1.3 or 1.0)
    plt.bar(ticks, vols, color="blue")
    plt.xticks(tick_pos, [labels[i] for i in tick_pos])
    return plt.build()


def make_layout(
    ticker_str: str,
    portfolio_table: Table,
    perf_table: Table,
    risk_table: Table,
    gov_table: Table,
    chart_panel: str,
    status_line: str,
    chart_h: int,
) -> Layout:
    grid = Table.grid(padding=(1, 2))
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_row(portfolio_table, perf_table)
    grid.add_row(risk_table, gov_table)
    layout = Layout()
    layout.split_column(
        Layout(Panel(ticker_str, title="Market Data"), size=3),
        Layout(Text.from_ansi(chart_panel), size=chart_h),
        Layout(grid, ratio=1),
        Layout(Panel(status_line, title="Status"), size=3),
    )
    return layout


def build_portfolio_table(trader: PaperTrader, config: Config, gp) -> Table:
    t = Table(title="Portfolio", show_header=False)
    s = trader.state
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="yellow")
    pnl_pct = Decimal("0")
    if s.total_trades > 0:
        pnl_pct = (trader.portfolio_value - Decimal(str(config.initial_balance_counter))) / Decimal(
            str(config.initial_balance_counter)
        ) * 100
    t.add_row("Balance (ZAR)", f"{s.balance_counter:.2f}")
    t.add_row("Balance (BTC)", f"{s.balance_base:.8f}")
    net_btc = s.balance_base - Decimal(str(config.initial_balance_base))
    t.add_row("Net BTC (stacked)", f"{net_btc:+.8f}")
    t.add_row("Portfolio Value", f"{trader.portfolio_value:.2f}")
    t.add_row("Total P&L", f"{pnl_pct:.2f}%" if pnl_pct != 0 else "0.00%")
    t.add_row("Trades Total", str(s.total_trades))
    t.add_row("Trades Today", f"{s.trades_today} / {gp.max_daily_trades}")
    t.add_row("Open Position", str(s.open_position is not None))
    t.add_row("Fees Paid", f"{s.total_fees:.2f}")
    return t


def build_perf_table(metrics) -> Table:
    t = Table(title="Performance", show_header=False)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    if metrics.total_trades == 0:
        t.add_row("Status", "No trades yet")
        return t
    t.add_row("Win Rate", f"{metrics.win_rate * 100:.1f}%")
    t.add_row("Total P&L", f"{metrics.total_pnl:.2f}")
    t.add_row("Sharpe", f"{metrics.sharpe_ratio:.2f}")
    t.add_row("Max Drawdown", f"{metrics.max_drawdown_pct:.1f}%")
    t.add_row("Current Drawdown", f"{metrics.current_drawdown_pct:.1f}%")
    t.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
    t.add_row("Avg Return/Trade", f"{metrics.avg_return_per_trade:.2f}%")
    t.add_row("Consecutive Losses", str(metrics.consecutive_losses))
    t.add_row("Best Trade", f"{metrics.best_trade_pct:.2f}%")
    t.add_row("Worst Trade", f"{metrics.worst_trade_pct:.2f}%")
    t.add_row("Avg Holding Period", f"{metrics.avg_holding_period_minutes:.1f}m")
    return t


def build_risk_table(risk_manager: RiskManager, status: RiskStatus) -> Table:
    t = Table(title="Risk Status", show_header=False)
    t.add_column("Circuit Breaker", style="cyan")
    t.add_column("Status", style="magenta")
    t.add_row("Daily Loss Limit", f"{risk_manager.max_daily_loss_pct}%")
    t.add_row("Max Drawdown", f"{risk_manager.max_drawdown_pct}%")
    t.add_row("Consecutive Losses", str(risk_manager.max_consecutive_losses))
    if status.cooldown_active:
        remaining = max(0, status.cooldown_until - time.time())
        t.add_row("Cooldown", Text(f"Cooldown {remaining:.0f}s", style="yellow"))
    t.add_row("Can Trade", Text("YES" if status.trading_allowed else "NO", style="green" if status.trading_allowed else "red"))
    if status.reason:
        t.add_row("Reason", status.reason)
    return t


def build_gov_table(gp, governor: GovernorAgent, gov_state: GovernanceState) -> Table:
    t = Table(title="Governance", show_header=False)
    t.add_column("Parameter", style="cyan")
    t.add_column("Value", style="blue")
    t.add_row("Stop Loss", f"{gp.stop_loss_pct * 100:.1f}%")
    t.add_row("Take Profit", f"{gp.take_profit_pct * 100:.1f}%")
    t.add_row("Max Position", f"{gp.max_position_size_pct * 100:.1f}%")
    t.add_row("RSI OB/OS", f"{gp.rsi_overbought:.0f}/{gp.rsi_oversold:.0f}")
    t.add_row("Max/Day", str(gp.max_daily_trades))
    t.add_row("Regime", gp.regime)
    t.add_row("Reflections", str(gov_state.reflection_count))
    t.add_row("LLM", governor.llm_provider if governor.llm_provider != "none" else "rules-only")
    if gp.governor_notes:
        t.add_row("Notes", gp.governor_notes[:60])
    return t


config = Config.from_env()
state = State.load(config.state_file, config.initial_balance_base, config.initial_balance_counter)
trade_log = TradeLog.load(config.state_file.replace(".json", "_trades.json"))
completed_log = CompletedTradeLog.load(config.completed_trades_file or config.state_file.replace(".json", "_completed.json"))
gov_state = GovernanceState.load(config.governance_state_file)

trader = PaperTrader(state, trade_log, completed_log, config.trading_pair)
strategy = Strategy(config, trader)
risk_manager = RiskManager(
    max_daily_loss_pct=config.risk_max_daily_loss_pct,
    max_drawdown_pct=config.risk_max_drawdown_pct,
    max_consecutive_losses=config.risk_max_consecutive_losses,
    cooldown_hours=config.risk_cooldown_hours,
    volatility_position_sizing=config.risk_volatility_sizing,
    base_position_size_pct=config.max_position_size_pct,
    max_position_size_pct=max(config.max_position_size_pct, config.max_position_size_pct),
    min_position_size_pct=0.05,
)
governor = GovernorAgent(
    llm_provider=config.llm_provider,
    llm_api_key=config.llm_api_key,
    llm_model=config.llm_model,
    reflection_interval_trades=config.governor_reflection_interval_trades,
    reflection_interval_seconds=config.governor_reflection_interval_seconds,
)

strategy.update_governance(gov_state.params)
client = LunoClient(config.luno_api_base, config.luno_api_key_id, config.luno_api_secret)


async def main_loop() -> None:
    if trader.state.total_trades == 0:
        console.print("[bold green]Luno AI Trading Bot Started (Paper Trading Mode)[/bold green]")
        console.print(f"Pair: {config.trading_pair} | Initial Balance: {config.initial_balance_counter}")
        console.print(f"Scan interval: {config.scan_interval_seconds}s")
        console.print(f"LLM: {config.llm_provider or 'none (rules-only)'}")
        console.print(f"Governor: {'enabled' if config.llm_provider != 'none' else 'disabled'}\n")

    while True:
        try:
            ticker = client.get_ticker(config.trading_pair)
            if ticker.status != "ACTIVE":
                console.print(f"[yellow]Market status: {ticker.status}[/yellow]")
                await asyncio.sleep(config.scan_interval_seconds)
                continue

            now = datetime.now(tz=timezone.utc)
            since_ms = int((now.timestamp() - 86400 * 3) * 1000)
            candles = client.get_candles(config.trading_pair, config.candle_duration, since_ms)
            ctx = strategy.build_context(ticker, candles)

            today = now.strftime("%Y-%m-%d")
            trader.reset_daily_counter(today)
            trader.update_open_position_value(ticker.last_trade)
            if trader.portfolio_value > state.peak_portfolio_value:
                state.peak_portfolio_value = trader.portfolio_value
            risk_manager.reset_daily(trader.portfolio_value)

            gp = strategy.governance
            metrics = compute_metrics(
                trader.completed_log.trades,
                trader.portfolio_value,
                state.peak_portfolio_value,
                Decimal(str(config.initial_balance_counter)),
                state.consecutive_losses,
            )

            risk_status = risk_manager.check_circuit_breakers(
                trader.portfolio_value,
                state.peak_portfolio_value,
                state.consecutive_losses,
                trader.state.trades_today,
                gp.max_daily_trades,
            )

            signal = await strategy.analyze(ctx)
            can_trade = risk_status.trading_allowed

            ticker_text = (
                f"[bold]Pair:[/bold] {config.trading_pair}  "
                f"[bold]Ask:[/bold] {ticker.ask}  "
                f"[bold]Bid:[/bold] {ticker.bid}  "
                f"[bold]Last:[/bold] {ticker.last_trade}  "
                f"[bold]RSI:[/bold] {ctx.rsi:.1f}" if ctx.rsi is not None else "N/A"
            )

            signal_text = f"[bold]{signal.action}[/bold] ({signal.confidence:.0%}) - {signal.reason}"
            portfolio_table = build_portfolio_table(trader, config, gp)
            perf_table = build_perf_table(metrics)
            risk_table = build_risk_table(risk_manager, risk_status)
            gov_table = build_gov_table(gp, governor, gov_state)

            status = (
                f"Last scan: {now.strftime('%H:%M:%S')} | "
                f"Signal: {signal_text} | "
                f"Can trade: {'YES' if can_trade else 'NO'} | "
                f"Next: in {config.scan_interval_seconds}s"
            )

            term_size = console.size
            chart_h = chart_height(term_size.height)
            chart_panel = build_price_chart(candles, term_size.width, chart_h)
            layout = make_layout(
                ticker_text, portfolio_table, perf_table, risk_table, gov_table, chart_panel, status, chart_h
            )
            console.clear()
            console.print(layout)

            if can_trade and signal.action == "BUY" and trader.state.trades_today < gp.max_daily_trades:
                price = (ticker.ask + ticker.bid) / 2 if ticker.ask and ticker.bid else ticker.last_trade
                size_pct = risk_manager.compute_position_size(
                    trader.state.balance_counter, price, ctx.atr
                )
                vol = (trader.state.balance_counter * size_pct) / price

                order = trader.buy(price, vol)
                if order:
                    trader.state.trades_today += 1
                    _save_state()

            elif can_trade and signal.action == "SELL" and trader.state.open_position is not None:
                price = (ticker.ask + ticker.bid) / 2 if ticker.ask and ticker.bid else ticker.last_trade
                pos_vol = Decimal(trader.state.open_position["volume"])
                sell_vol = pos_vol * Decimal(str(gp.sell_fraction_pct))
                order = trader.sell(price, sell_vol, reason=signal.reason)
                if order:
                    trader.state.trades_today += 1
                    _save_state()

            if governor.should_reflect(gov_state, trader.state.total_trades, time.time()):
                recent_trades = trader.completed_log.last_n(5)
                summary = "\n".join(
                    f"  {datetime.fromtimestamp(t['entry_time'], tz=timezone.utc).isoformat()} "
                    f"entry={t['entry_price']} exit={t['exit_price']} pnl={t['pnl_pct']}% "
                    f"reason={t.get('exit_reason', '')}"
                    for t in recent_trades
                )

                market_lines = []
                if ctx.rsi is not None:
                    market_lines.append(f"RSI: {ctx.rsi:.1f}")
                if ctx.atr is not None:
                    market_lines.append(f"ATR: {ctx.atr:.2f}")
                market_summary = " | ".join(market_lines) if market_lines else "N/A"

                new_params = await governor.reflect(
                    gov_state, metrics, gp, summary, market_summary
                )
                if new_params != gp:
                    strategy.update_governance(new_params)
                    _save_state()

            _save_state()

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        await asyncio.sleep(config.scan_interval_seconds)


def _save_state() -> None:
    trader.state.save(config.state_file)
    trade_log.save(config.state_file.replace(".json", "_trades.json"))
    completed_log.save(config.completed_trades_file or config.state_file.replace(".json", "_completed.json"))
    gov_state.save(config.governance_state_file)


async def main() -> None:
    setup_logging(config)
    try:
        await main_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        _save_state()
        client.close()
        logger.info("State saved. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
