from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass
class Order:
    order_id: str
    pair: str
    side: str
    price: Decimal
    volume: Decimal
    filled_at: int
    fee: Decimal = Decimal("0")


@dataclass
class CompletedTrade:
    buy_order_id: str
    sell_order_id: str
    pair: str
    entry_price: Decimal
    exit_price: Decimal
    volume: Decimal
    entry_time: int
    exit_time: int
    buy_fee: Decimal
    sell_fee: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    holding_period_seconds: int
    exit_reason: str = "signal"


@dataclass
class TradeLog:
    trades: list[dict[str, Any]] = field(default_factory=list)

    def add(self, order: Order) -> None:
        self.trades.append({
            "order_id": order.order_id,
            "pair": order.pair,
            "side": order.side,
            "price": str(order.price),
            "volume": str(order.volume),
            "fee": str(order.fee),
            "filled_at": order.filled_at,
            "filled_at_iso": datetime.fromtimestamp(
                order.filled_at / 1000, tz=timezone.utc
            ).isoformat(),
        })

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.trades, indent=2))

    @classmethod
    def load(cls, path: str) -> TradeLog:
        try:
            data = json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
        tl = cls()
        tl.trades = data
        return tl


@dataclass
class CompletedTradeLog:
    trades: list[dict[str, Any]] = field(default_factory=list)

    def add(self, trade: CompletedTrade) -> None:
        self.trades.append(asdict(trade))

    def last_n(self, n: int) -> list[dict[str, Any]]:
        return self.trades[-n:]

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.trades, indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> CompletedTradeLog:
        try:
            data = json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
        tl = cls()
        tl.trades = data
        return tl


@dataclass
class State:
    balance_base: Decimal
    balance_counter: Decimal
    trades_today: int
    last_trade_date: str
    open_position: dict[str, Any] | None
    total_trades: int
    total_fees: Decimal
    peak_portfolio_value: Decimal = Decimal("0")
    consecutive_losses: int = 0

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def initial(cls, base: float, counter: float) -> State:
        return cls(
            balance_base=Decimal(str(base)),
            balance_counter=Decimal(str(counter)),
            trades_today=0,
            last_trade_date="",
            open_position=None,
            total_trades=0,
            total_fees=Decimal("0"),
            peak_portfolio_value=Decimal(str(base + counter)),
            consecutive_losses=0,
        )

    @classmethod
    def load(cls, path: str, base: float, counter: float) -> State:
        try:
            data = json.loads(Path(path).read_text())
            return cls(
                balance_base=Decimal(str(data["balance_base"])),
                balance_counter=Decimal(str(data["balance_counter"])),
                trades_today=data["trades_today"],
                last_trade_date=data["last_trade_date"],
                open_position=data.get("open_position"),
                total_trades=data.get("total_trades", 0),
                total_fees=Decimal(str(data.get("total_fees", "0"))),
                peak_portfolio_value=Decimal(str(data.get("peak_portfolio_value", str(base + counter)))),
                consecutive_losses=data.get("consecutive_losses", 0),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return cls.initial(base, counter)


class PaperTrader:
    def __init__(self, state: State, trade_log: TradeLog, completed_log: CompletedTradeLog, pair: str) -> None:
        self.state = state
        self.trade_log = trade_log
        self.completed_log = completed_log
        self.pair = pair
        self._order_counter = 0

    @property
    def portfolio_value(self) -> Decimal:
        pos_value = Decimal("0")
        if self.state.open_position is not None:
            pos_value = Decimal(str(self.state.open_position.get("current_value", "0")))
        return self.state.balance_counter + pos_value

    def update_open_position_value(self, current_price: Decimal) -> None:
        if self.state.open_position is not None:
            vol = Decimal(self.state.open_position["volume"])
            self.state.open_position["current_value"] = str(vol * current_price)
            entry = Decimal(self.state.open_position["entry_price"])
            change = (current_price - entry) / entry
            self.state.open_position["unrealized_pnl_pct"] = str(change)

    def buy(self, price: Decimal, volume: Decimal) -> Order | None:
        volume = volume.copy_abs()
        cost = price * volume
        fee = cost * Decimal("0.001")
        total_cost = cost + fee
        if total_cost > self.state.balance_counter:
            max_vol = (self.state.balance_counter * Decimal("0.999")) / price
            if max_vol < Decimal("0.0001"):
                return None
            volume = max_vol
            cost = price * volume
            fee = cost * Decimal("0.001")
            total_cost = cost + fee

        self._order_counter += 1
        order = Order(
            order_id=f"paper_{int(time.time() * 1000)}_{self._order_counter}",
            pair=self.pair,
            side="BUY",
            price=price,
            volume=volume,
            filled_at=int(time.time() * 1000),
            fee=fee,
        )
        self.state.balance_counter -= total_cost
        self.state.balance_base += volume

        pos = self.state.open_position
        if pos is not None:
            old_vol = Decimal(pos["volume"])
            old_price = Decimal(pos["entry_price"])
            new_vol = old_vol + volume
            avg_price = (old_price * old_vol + price * volume) / new_vol
            pos["entry_price"] = str(avg_price)
            pos["volume"] = str(new_vol)
            pos["buy_fee"] = str(Decimal(pos.get("buy_fee", "0")) + fee)
            pos["current_value"] = str(new_vol * price)
            pos["unrealized_pnl_pct"] = str((price - avg_price) / avg_price)
        else:
            self.state.open_position = {
                "entry_price": str(price),
                "volume": str(volume),
                "entry_time": order.filled_at,
                "buy_fee": str(fee),
                "buy_order_id": order.order_id,
                "current_value": str(price * volume),
                "unrealized_pnl_pct": "0",
            }
        self.state.total_fees += fee
        self.state.total_trades += 1
        self.trade_log.add(order)
        return order

    def sell(self, price: Decimal, volume: Decimal | None = None, reason: str = "signal") -> Order | None:
        if self.state.open_position is None:
            return None
        if volume is None:
            volume = self.state.balance_base
        if volume <= Decimal("0") or volume > self.state.balance_base:
            return None

        proceeds = price * volume
        fee = proceeds * Decimal("0.001")
        net_proceeds = proceeds - fee

        self._order_counter += 1
        order = Order(
            order_id=f"paper_{int(time.time() * 1000)}_{self._order_counter}",
            pair=self.pair,
            side="SELL",
            price=price,
            volume=volume,
            filled_at=int(time.time() * 1000),
            fee=fee,
        )

        entry_price = Decimal(self.state.open_position["entry_price"])
        pos_vol = Decimal(self.state.open_position["volume"])
        buy_fee = Decimal(self.state.open_position.get("buy_fee", "0"))
        entry_time = self.state.open_position["entry_time"]
        buy_order_id = self.state.open_position.get("buy_order_id", "")

        actual_cost = entry_price * volume
        fee_share = buy_fee * volume / pos_vol
        pnl_actual = net_proceeds - actual_cost - fee_share
        pnl_pct = (price - entry_price) / entry_price

        completed = CompletedTrade(
            buy_order_id=buy_order_id,
            sell_order_id=order.order_id,
            pair=self.pair,
            entry_price=entry_price,
            exit_price=price,
            volume=volume,
            entry_time=entry_time,
            exit_time=order.filled_at,
            buy_fee=fee_share,
            sell_fee=fee,
            pnl=pnl_actual,
            pnl_pct=pnl_pct,
            holding_period_seconds=int((order.filled_at - entry_time) / 1000),
            exit_reason=reason,
        )

        was_profitable = pnl_actual > 0

        self.state.balance_base -= volume
        self.state.balance_counter += net_proceeds
        if volume >= pos_vol - Decimal("1e-8"):
            self.state.open_position = None
        else:
            pos = self.state.open_position
            pos["volume"] = str(pos_vol - volume)
            pos["buy_fee"] = str(buy_fee - fee_share)
            pos["current_value"] = str((pos_vol - volume) * price)
            pos["unrealized_pnl_pct"] = str((price - entry_price) / entry_price)
        self.state.total_fees += fee
        self.state.total_trades += 1
        self.state.consecutive_losses = 0 if was_profitable else self.state.consecutive_losses + 1

        current_value = self.portfolio_value
        if current_value > self.state.peak_portfolio_value:
            self.state.peak_portfolio_value = current_value

        self.trade_log.add(order)
        self.completed_log.add(completed)
        return order

    def reset_daily_counter(self, today: str) -> None:
        if self.state.last_trade_date != today:
            self.state.trades_today = 0
            self.state.last_trade_date = today
