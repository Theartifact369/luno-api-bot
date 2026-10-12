from __future__ import annotations

import asyncio
import io
import logging
import math
import logging.handlers
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

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


def _frame_metrics(term_height: int) -> tuple[int, int]:
    """(chart_h, boxes_h): chart gets the majority; fixed rows = ticker(4) + status(3)."""
    box_h = max(8, term_height // 4)
    chart_h = max(6, min(40, term_height - 7 - box_h))
    box_h = term_height - 7 - chart_h  # exact fit; boxes absorb the remainder
    return chart_h, max(6, box_h)


def _plot_fig(width: int, height: int, draw) -> str:
    import plotext._core as _pltcore

    _pltcore._figure._active = _pltcore._figure._master
    plt.clear_figure()
    plt.theme("dark")
    plt.limit_size(False, False)
    plt.plotsize(max(20, width), height)
    draw()
    return plt.build()


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round y-axis ticks: ~target evenly spaced values spanning [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 10 ** math.floor(math.log10(span) - 1)
    for m in (1, 2, 5, 10):
        if span / (step * m) <= target:
            step *= m
            break
    v = math.ceil(lo / step) * step
    ticks = []
    while v <= hi + 1e-9:
        ticks.append(v)
        v += step
    return ticks or [lo]


def build_price_chart(candles: list[Candle], width: int = 80, height: int = 18) -> str:
    if len(candles) < 2:
        return "Collecting candles..."
    # ~2 columns per candle so bodies stay readable on narrow panes
    # (16 cols budgeted for the y labels + plot borders)
    n = max(14, min(48, (width - 16) // 2))
    candles = candles[-n:]
    opens = [float(c.open) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    closes = [float(c.close) for c in candles]
    vols = [float(c.volume) for c in candles]
    labels = [time.strftime("%H:%M", time.gmtime(c.timestamp)) for c in candles]
    ticks = list(range(len(candles)))
    n = len(candles)
    tick_pos = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    # volume = double its old share: 2/5 of chart height (= 2/3 of price height)
    vol_h = max(4, height * 2 // 5)
    price_h = max(4, height - vol_h)

    def price_fig() -> None:
        plt.title("Price (1h)")
        plt.candlestick(ticks, {"Open": opens, "High": highs, "Low": lows, "Close": closes})
        # compact y labels: "1,368,000" instead of raw "1368000.0"
        yts = _nice_ticks(min(lows), max(highs))
        plt.yticks(yts, [f"{t:,.0f}" for t in yts])
        plt.xticks(tick_pos, [labels[i] for i in tick_pos])

    def vol_fig() -> None:
        plt.title("Volume")
        plt.ylim(0, max(vols) * 1.3 or 1.0)
        # half-block fill: 2x vertical resolution, much less blocky than █
        plt.bar(ticks, vols, color="blue", marker="hd")
        plt.xticks(tick_pos, [labels[i] for i in tick_pos])

    return _plot_fig(width, price_h, price_fig) + "\n" + _plot_fig(width, vol_h, vol_fig)


# --- interactive TUI: live account panel + scrolling paper-trade stats --------
_stats_lines: list[str] = []
_account_lines: list[str] = []
_stats_offset = 0
_account_offset = 0
_refresh = False
_scroll_delta = 0
_acct_scroll_delta = 0
_quit = False
_wake: asyncio.Event | None = None
_saved_termios: list = []
_frame: dict = {}


def _split_widths(term_w: int) -> tuple[int, int]:
    # left column holds the account table only; paper stats take the rest
    left_w = max(40, term_w // 2)
    return left_w, term_w - left_w


def _capture_lines(renderable, width: int) -> list[str]:
    # force_terminal: keep ANSI colors (semantic green/red) in captured lines
    buf = io.StringIO()
    c = Console(file=buf, width=width, height=10_000, legacy_windows=False,
                force_terminal=True, color_system="standard")
    c.print(renderable)
    return buf.getvalue().rstrip("\n").splitlines()


def _draw_frame() -> None:
    global _stats_offset, _account_offset
    term = console.size
    chart_h, box_h = _frame_metrics(term.height)
    acct_h = box_h - 2
    max_acct = max(0, len(_account_lines) - acct_h)
    _account_offset = min(max(_account_offset, 0), max_acct)
    acct_visible = _account_lines[_account_offset:_account_offset + acct_h]
    max_off = max(0, len(_stats_lines) - acct_h)
    _stats_offset = min(max(_stats_offset, 0), max_off)
    visible = _stats_lines[_stats_offset:_stats_offset + acct_h]
    acct_label = (f"{_account_offset + 1}-{min(_account_offset + acct_h, len(_account_lines))}"
                  f"/{len(_account_lines)}") if _account_lines else "0/0"
    label = (f"{_stats_offset + 1}-{min(_stats_offset + acct_h, len(_stats_lines))}"
             f"/{len(_stats_lines)}") if _stats_lines else "0/0"
    left_w, right_w = _split_widths(term.width)
    left = Panel(
        Text.from_ansi("\n".join(acct_visible)),
        title=f"Luno Account (live) ({acct_label}, arrows)", title_align="left", height=box_h,
    )
    right = Panel(
        Text.from_ansi("\n".join(visible)),
        title=f"Papertrade Stats ({label}, j/k)", title_align="left", height=box_h,
    )
    outer = Table.grid(expand=True)
    outer.add_column()
    outer.add_column()
    outer.columns[0].width = left_w
    outer.columns[1].width = right_w
    outer.add_row(left, right)
    console.clear()
    # one renderable, exactly term.height tall: no scroll drift between refreshes
    chart_lines = _frame["chart"].splitlines()
    chart_lines = (chart_lines + [""] * chart_h)[:chart_h]
    frame = Group(
        Panel(_frame["ticker"], title="Market Data"),
        Text.from_ansi("\n".join(chart_lines)),
        outer,
        Panel(_frame["status"], title="Status"),
    )
    console.print(frame, height=term.height, crop=True)


def _start_key_reader() -> None:
    global _wake
    loop = asyncio.get_running_loop()
    ev = asyncio.Event()
    _wake = ev
    fd = sys.stdin.fileno()
    _saved_termios.append(termios.tcgetattr(fd))
    tty.setcbreak(fd)

    def reader() -> None:
        global _refresh, _scroll_delta, _acct_scroll_delta, _quit
        while True:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                return
            wake = True
            if ch in ("r", "R"):
                _refresh = True
            elif ch in ("j", "J"):
                _scroll_delta += 3
            elif ch in ("k", "K"):
                _scroll_delta -= 3
            elif ch == "\x1b":
                try:
                    seq = sys.stdin.read(2)
                except Exception:
                    seq = ""
                if seq == "[A":
                    _acct_scroll_delta -= 3
                elif seq == "[B":
                    _acct_scroll_delta += 3
                else:
                    wake = False
            elif ch in ("q", "Q"):
                _quit = True
            else:
                wake = False
            if wake:
                loop.call_soon_threadsafe(ev.set)

    threading.Thread(target=reader, daemon=True).start()


async def _wait_wake() -> None:
    if _wake is None:
        await asyncio.sleep(config.scan_interval_seconds)
        return
    if _refresh or _scroll_delta or _acct_scroll_delta or _quit:
        return  # key landed while we were busy: handle it next loop turn
    _wake.clear()
    try:
        await asyncio.wait_for(_wake.wait(), timeout=config.scan_interval_seconds)
    except asyncio.TimeoutError:
        pass


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
    net_btc = s.balance_base + s.vault_base - Decimal(str(config.initial_balance_base))
    t.add_row("Net BTC (stacked)", f"{net_btc:+.8f}")
    t.add_row("Vault (BTC)", f"{s.vault_base:.8f}")
    if trader.last_price is not None:
        t.add_row("Vault (ZAR)", f"{s.vault_base * trader.last_price:.2f}")
    t.add_row("Portfolio Value", f"{trader.portfolio_value:.2f}")
    pnl_style = "" if pnl_pct == 0 else ("green" if pnl_pct > 0 else "red")
    if pnl_style:
        t.add_row("Total P&L", Text(f"{pnl_pct:.2f}%", style=pnl_style))
    else:
        t.add_row("Total P&L", "0.00%")
    t.add_row("Trades Total", str(s.total_trades))
    t.add_row("Trades Today", f"{s.trades_today} / {gp.max_daily_trades}")
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
    pnl_style = "green" if metrics.total_pnl > 0 else ("red" if metrics.total_pnl < 0 else "")
    t.add_row("Total P&L", Text(f"{metrics.total_pnl:.2f}", style=pnl_style) if pnl_style else f"{metrics.total_pnl:.2f}")
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


TICKER_ASSETS = ("XBT", "ETH", "USDT", "ZAR", "GRAM", "XRP", "SAND")
# GRAM was renamed TON on Luno: the bar shows "GRAM" but prices come from TON pairs
PAIR_ALIAS = {"GRAM": "TON"}


_prev_prices: dict[str, Decimal] = {}  # last scan's prices, for direction arrows


def build_ticker_bar(tickers: list[dict]) -> str:
    zar = {tk["pair"][:-3]: Decimal(tk["last_trade"]) for tk in tickers if tk["pair"].endswith("ZAR")}
    xbt = zar.get("XBT")
    parts = []
    for a in TICKER_ASSETS:
        base = PAIR_ALIAS.get(a, a)
        if a == "ZAR":
            p = Decimal("1")
        else:
            p = zar.get(base)
            if p is None and xbt is not None:
                # no ZAR pair -> derive from the asset's XBT pair (e.g. SANDXBT)
                for tk in tickers:
                    if tk["pair"] == base + "XBT":
                        p = Decimal(tk["last_trade"]) * xbt
                        break
            if p is None and xbt is not None:
                # still nothing -> fiat cross, e.g. TON (GRAM) only trades as TONMYR:
                # TONMYR x (XBTZAR / XBTMYR)
                for tk in tickers:
                    pair = tk["pair"]
                    if pair.startswith(base) and len(pair) > len(base) and not pair.endswith("XBT"):
                        fiat = pair[len(base):]
                        btc_fiat = next((Decimal(t["last_trade"]) for t in tickers
                                         if t["pair"] == "XBT" + fiat), None)
                        if btc_fiat:
                            p = Decimal(tk["last_trade"]) * xbt / btc_fiat
                            break
        prev = _prev_prices.get(base)
        if p is not None:
            _prev_prices[base] = p
        arrow, color, body = "", "", None
        if p is not None and prev is not None and p != prev:
            arrow = "▲" if p > prev else "▼"
            color = "green" if p > prev else "red"
        if p is None:
            parts.append(f"[bold]{a}[/bold] —")
        else:
            body = f"R{p:,.0f}" if p >= 1000 else "R" + f"{p:.4f}".rstrip("0").rstrip(".")
            if color:
                parts.append(f"[{color}][bold]{a}[/bold] {arrow} {body}[/{color}]")
            else:
                parts.append(f"[bold]{a}[/bold] {body}")
    return "  ".join(parts)


def build_account_table(client: LunoClient, config: Config, tickers: list[dict]) -> Table:
    """Account activity, not holdings.

    This panel used to print live balances and a ZAR total. A balance sheet
    leaks the operator's real net worth into every screenshot, and that is not
    what a trading log is for. Trade history shows what the account did.
    """
    t = Table(title="Trade History (live)", show_header=False)
    t.add_column("When", style="cyan")
    t.add_column("Side", style="magenta")
    t.add_column("Detail", style="yellow")
    if not config.luno_api_key_id:
        t.add_row("Status", Text("no LUNO_API_KEY_ID in .env", style="red"))
        return t
    try:
        trades = client.get_account_trades(config.trading_pair)
    except Exception as e:
        t.add_row("Status", Text(f"API error: {str(e)[:80]}", style="red"))
        return t
    if not trades:
        t.add_row("Status", Text(f"no trades on {config.trading_pair}", style="dim"))
        return t
    for tr in sorted(trades, key=lambda x: int(x["timestamp"]), reverse=True)[:12]:
        when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(tr["timestamp"]) / 1000))
        side = "BUY" if tr.get("is_buy") else "SELL"
        detail = f"{Decimal(str(tr['price'])):,.2f} x {Decimal(str(tr['volume'])):.8f}"
        t.add_row(when, Text(side, style="green" if side == "BUY" else "red"), detail)
    return t


def build_orders_table(client: LunoClient, config: Config) -> Table:
    t = Table(title="Orders / Recent Fills", show_header=True)
    t.add_column("Type", style="cyan")
    t.add_column("Side", style="magenta")
    t.add_column("Price", justify="right", style="yellow")
    t.add_column("Volume", justify="right")
    if not config.luno_api_key_id:
        return t
    rows: list[tuple[str, str, str, str]] = []
    try:
        for o in client.get_pending_orders(config.trading_pair):
            price = Decimal(str(o.get("limit_price") or o.get("counter") or 0))
            vol = Decimal(str(o.get("limit_volume") or o.get("base") or 0))
            rows.append(("pending", o.get("type", "?"), f"{price:,.2f}", f"{vol:.8f}"))
        fills = client.get_account_trades(config.trading_pair)
        for tr in sorted(fills, key=lambda x: int(x["timestamp"]), reverse=True)[:5]:
            when = time.strftime("%d %H:%M", time.gmtime(int(tr["timestamp"]) / 1000))
            rows.append((when, "BUY" if tr["is_buy"] else "SELL",
                         f"{Decimal(str(tr['price'])):,.2f}", f"{Decimal(str(tr['volume'])):.8f}"))
    except Exception as e:
        t.add_row("error", str(e)[:60], "", "")
    if not rows:
        t.add_row("-", "no pending orders or fills", "-", "-")
    for typ, side, price, vol in rows:
        side_style = {"BUY": "green", "BID": "green", "SELL": "red", "ASK": "red"}.get(side, "")
        t.add_row(typ, Text(side, style=side_style) if side_style else side, price, vol)
    return t


def build_position_table(trader: PaperTrader) -> Table | None:
    # live unrealized P&L for the open paper position (positions-panel pattern)
    pos = trader.state.open_position
    if pos is None:
        return None
    entry = Decimal(pos["entry_price"])
    vol = Decimal(pos["volume"])
    cur_value = Decimal(pos.get("current_value") or 0)
    mark = cur_value / vol if vol else Decimal("0")
    pnl_zar = cur_value - entry * vol
    pnl_pct = Decimal(pos.get("unrealized_pnl_pct") or 0) * 100
    color = "green" if pnl_zar >= 0 else "red"
    t = Table(title="Open Position", show_header=False)
    t.add_column(style="cyan")
    t.add_column()
    t.add_row("Size (BTC)", f"{vol:.8f}")
    t.add_row("Entry", f"R{entry:,.2f}")
    t.add_row("Mark", f"R{mark:,.2f}")
    t.add_row("Unrealized", Text(f"R{pnl_zar:,.2f} ({pnl_pct:+.2f}%)", style=color))
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
if trader.state.vault_deposits < 1:
    from client import LunoClient
    client = LunoClient(config.luno_api_base, config.luno_api_key_id, config.luno_api_secret)
    try:
        ticker = client.get_ticker(config.trading_pair)
        client.close()
        zar_amount = Decimal(str(config.initial_balance_counter)) * Decimal(str(config.initial_vault_pct))
        btc = trader.initial_vault_allocation(zar_amount, ticker.last_trade)
        logger.info("Initial vault allocation: R%.2f -> %.8f BTC at R%.2f", float(zar_amount), float(btc), float(ticker.last_trade))
    except Exception as e:
        logger.warning("Initial vault allocation failed: %s", e)
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
    global _stats_offset, _account_offset, _refresh, _scroll_delta, _acct_scroll_delta, _quit
    if trader.state.total_trades == 0:
        console.print("[bold green]Luno AI Trading Bot Started (Paper Trading Mode)[/bold green]")
        console.print(f"Pair: {config.trading_pair} | Initial Balance: {config.initial_balance_counter}")
        console.print(f"Scan interval: {config.scan_interval_seconds}s")
        console.print(f"LLM: {config.llm_provider or 'none (rules-only)'}")
        console.print(f"Governor: {'LLM' if config.llm_provider != 'none' else 'rules (no API key needed)'}\n")

    if sys.stdin.isatty():
        _start_key_reader()

    last_scan = 0.0
    import signal
    def _on_resize(signum, frame):
        global _refresh
        _refresh = True
    signal.signal(signal.SIGWINCH, _on_resize)
    while True:
        due = _refresh or (time.time() - last_scan) >= config.scan_interval_seconds
        if due:
            _refresh = False
            last_scan = time.time()
        elif _scroll_delta or _acct_scroll_delta:
            _stats_offset += _scroll_delta
            _scroll_delta = 0
            _account_offset += _acct_scroll_delta
            _acct_scroll_delta = 0
            _draw_frame()
            await _wait_wake()
            if _quit:
                raise KeyboardInterrupt
            continue
        else:
            await _wait_wake()
            if _quit:
                raise KeyboardInterrupt
            continue
        try:
            ticker = client.get_ticker(config.trading_pair)
            if ticker.status != "ACTIVE":
                logger.warning("market status %s - skipping scan", ticker.status)
                continue

            now = datetime.now(tz=timezone.utc)
            since_ms = int((now.timestamp() - 86400 * 3) * 1000)
            candles = client.get_candles(config.trading_pair, config.candle_duration, since_ms)
            ctx = strategy.build_context(ticker, candles)

            daily_candles: list[Candle] | None = None
            if config.strategy == "regime":
                # regime needs 200 daily bars; Luno serves them directly at duration=86400
                daily_since = int((now.timestamp() - 86400 * 230) * 1000)
                daily_candles = client.get_candles(config.trading_pair, 86400, daily_since)

            today = now.strftime("%Y-%m-%d")
            trader.reset_daily_counter(today)
            trader.last_price = ticker.last_trade
            trader.update_open_position_value(ticker.last_trade)
            total_value = trader.total_value(ticker.last_trade)
            if total_value > state.peak_portfolio_value:
                state.peak_portfolio_value = total_value
            risk_manager.reset_daily(total_value, today)

            gp = strategy.governance
            metrics = compute_metrics(
                trader.completed_log.trades,
                total_value,
                state.peak_portfolio_value,
                Decimal(str(config.initial_balance_counter)),
                state.consecutive_losses,
            )

            risk_status = risk_manager.check_circuit_breakers(
                total_value,
                state.peak_portfolio_value,
                state.consecutive_losses,
                trader.state.trades_today,
                gp.max_daily_trades,
            )
            if risk_status.cooldown_just_expired:
                state.consecutive_losses = 0  # cooldown served; re-arm on new losses only

            signal = await strategy.analyze(ctx, daily_candles)
            can_trade = risk_status.trading_allowed
            # ponytail: one line per scan so headless runs are reviewable from logs
            logger.info(
                "scan ts=%s signal=%s can_trade=%s total_value=%s",
                now.strftime("%Y-%m-%dT%H:%M:%SZ"), signal, can_trade,
                trader.total_value(ticker.last_trade),
            )

            rsi_text = f"{ctx.rsi:.1f}" if ctx.rsi is not None else "N/A"
            ticker_text = (
                f"[bold]Pair:[/bold] {config.trading_pair}  "
                f"[bold]Ask:[/bold] {ticker.ask}  "
                f"[bold]Bid:[/bold] {ticker.bid}  "
                f"[bold]Last:[/bold] {ticker.last_trade}  "
                f"[bold]RSI:[/bold] {rsi_text}"
            )

            signal_text = f"[bold]{signal.action}[/bold] ({signal.confidence:.0%}) - {signal.reason}"
            portfolio_table = build_portfolio_table(trader, config, gp)
            perf_table = build_perf_table(metrics)
            risk_table = build_risk_table(risk_manager, risk_status)
            gov_table = build_gov_table(gp, governor, gov_state)
            try:
                tickers = client.get_tickers()
            except Exception:
                tickers = []  # ponytail: bar shows dashes, account shows unvalued balances
            account_table = build_account_table(client, config, tickers)
            orders_table = build_orders_table(client, config)

            status = (
                f"[bold yellow]PAPER[/bold yellow] | {now.strftime('%H:%M:%S')} UTC | "
                f"Last scan: {now.strftime('%H:%M:%S')} | "
                f"Signal: {signal_text} | "
                f"Can trade: {'YES' if can_trade else 'NO'} | "
                "r refresh | j/k paper | arrows account | q quit"
            )

            term_size = console.size
            left_w, right_w = _split_widths(term_size.width)
            lines: list[str] = []
            pos_table = build_position_table(trader)
            paper_tables = [portfolio_table, orders_table, perf_table, risk_table, gov_table]
            if pos_table is not None:
                paper_tables.insert(0, pos_table)
            for tb in paper_tables:
                tb.expand = True
                lines.extend(_capture_lines(tb, right_w - 4))
                lines.append("")
            _stats_lines[:] = lines
            _account_lines[:] = _capture_lines(account_table, left_w - 4)
            _stats_offset = 0
            _frame.clear()
            _frame.update(
                ticker=build_ticker_bar(tickers) + "\n" + ticker_text,
                chart=build_price_chart(candles, term_size.width, _frame_metrics(term_size.height)[0]),
                status=status,
            )
            _draw_frame()

            if can_trade and signal.action == "BUY" and trader.state.trades_today < gp.max_daily_trades:
                price = ticker.ask if ticker.ask else ticker.last_trade  # realistic fill: cross the spread
                if config.strategy == "regime":
                    size_pct = risk_manager.compute_position_size(
                        trader.state.balance_counter, price, ctx.atr
                    )
                    order = trader.buy(price, trader.state.balance_counter * size_pct / price,
                                       reason=signal.family or signal.reason)
                else:
                    size_pct = risk_manager.compute_position_size(
                        trader.state.balance_counter, price, ctx.atr
                    )
                    # governance cap is a share of TOTAL value; buy only the headroom
                    # the open position doesn't already occupy
                    total = trader.total_value(price)
                    target = total * Decimal(str(size_pct))
                    already = (Decimal(str(trader.state.open_position["current_value"]))
                               if trader.state.open_position else Decimal("0"))
                    headroom = target - already
                    order = trader.buy(price, headroom / price,
                                       reason=signal.family or signal.reason) if headroom > 0 else None
                if order:
                    trader.state.trades_today += 1
                    _save_state()

            elif signal.action == "SELL" and trader.state.open_position is not None:
                # no can_trade gate: a risk breaker must never trap us in a
                # position. Drawdown only re-arms on a new peak, so gating the
                # exit meant a death cross could be refused indefinitely while
                # the strategy had no other way out.
                price = ticker.bid if ticker.bid else ticker.last_trade  # realistic fill: cross the spread
                pos_vol = Decimal(trader.state.open_position["volume"])
                sell_vol = pos_vol if config.strategy == "regime" else pos_vol * Decimal(str(gp.sell_fraction_pct))
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
        finally:
            # in finally so every exit path sleeps: a bare `continue` above
            # (non-ACTIVE market) used to skip this and hot-loop the API
            await _wait_wake()

        if _quit:
            raise KeyboardInterrupt


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
    finally:
        if _saved_termios:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_termios[0])
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
