from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: Decimal = Decimal("0")
    total_pnl_pct: float = 0.0
    avg_return_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_holding_period_minutes: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0


def compute_metrics(
    completed_trades: list[dict[str, Any]],
    current_value: Decimal,
    peak_value: Decimal,
    initial_balance: Decimal,
    consecutive_losses: int,
) -> PerformanceMetrics:
    m = PerformanceMetrics()
    m.total_trades = len(completed_trades)

    if m.total_trades == 0:
        m.current_drawdown_pct = float(
            (peak_value - current_value) / peak_value * 100
        ) if peak_value > 0 else 0.0
        return m

    returns: list[float] = []
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    total_holding = 0
    best = -999.0
    worst = 999.0

    for t in completed_trades:
        pnl_pct = float(t.get("pnl_pct", 0)) * 100
        returns.append(pnl_pct)
        if pnl_pct >= 0:
            m.winning_trades += 1
            gross_profit += Decimal(str(pnl_pct))
            if pnl_pct > best:
                best = pnl_pct
        else:
            m.losing_trades += 1
            gross_loss += abs(Decimal(str(pnl_pct)))
            if pnl_pct < worst:
                worst = pnl_pct

        hp = t.get("holding_period_seconds", 0)
        total_holding += hp

    m.total_pnl = sum(
        Decimal(str(t.get("pnl", "0"))) for t in completed_trades
    )

    initial_f = float(initial_balance)
    m.total_pnl_pct = float(m.total_pnl) / initial_f * 100 if initial_f > 0 else 0.0
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0
    m.avg_return_per_trade = sum(returns) / len(returns) if returns else 0.0

    win_returns = [r for r in returns if r >= 0]
    loss_returns = [r for r in returns if r < 0]
    m.avg_win = sum(win_returns) / len(win_returns) if win_returns else 0.0
    m.avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0.0

    m.profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (
        float(gross_profit) if gross_profit > 0 else 0.0
    )

    avg_r = sum(returns) / len(returns)
    variance = sum((r - avg_r) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance) if variance > 0 else 1.0
    m.sharpe_ratio = (avg_r / std_dev) * math.sqrt(365) if std_dev > 0 else 0.0

    peak = float(peak_value)
    current = float(current_value)
    m.max_drawdown_pct = 0.0
    running_peak = initial_f
    for t in completed_trades:
        pnl_val = float(t.get("pnl", "0"))
        running_peak = max(running_peak, running_peak + pnl_val)
        dd = (running_peak - (running_peak + pnl_val)) / running_peak * 100
        if dd > m.max_drawdown_pct:
            m.max_drawdown_pct = dd

    m.current_drawdown_pct = max(0.0, (peak - current) / peak * 100) if peak > 0 else 0.0
    m.consecutive_losses = consecutive_losses

    m.avg_holding_period_minutes = (total_holding / len(returns) / 60) if returns else 0.0
    m.best_trade_pct = best if best != -999.0 else 0.0
    m.worst_trade_pct = worst if worst != 999.0 else 0.0

    return m
