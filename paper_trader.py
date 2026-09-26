from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    entry_reason: str = ""  # the signal family that opened this position (decision log)


@dataclass
class TradeLog:
    trades: list[dict[str, Any]] = field(default_factory=list)

    def add(self, order: Order, reason: str = "") -> None:
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
            "reason": reason,
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
    vault_base: Decimal = Decimal("0")       # BTC locked from realized wins, never traded
    vault_deposits: Decimal = Decimal("0")   # ZAR total ever swept into the vault

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
                vault_base=Decimal(str(data.get("vault_base", "0"))),
                vault_deposits=Decimal(str(data.get("vault_deposits", "0"))),
            )
        except FileNotFoundError:
            return cls.initial(base, counter)
        except (json.JSONDecodeError, KeyError) as e:
            # corrupt state: move it aside, never silently overwrite a real
            # account with a fresh one (the next _save_state would clobber it)
            backup = f"{path}.corrupt"
            Path(path).replace(backup)
            logger.error(
                "%s is corrupt (%s) - moved to %s, starting a fresh account. "
                "Recover the old state from the backup before trusting this run.",
                path, e, backup,
            )
            return cls.initial(base, counter)


class PaperTrader:
    def __init__(self, state: State, trade_log: TradeLog, completed_log: CompletedTradeLog, pair: str) -> None:
        self.state = state
        self.trade_log = trade_log
        self.completed_log = completed_log
        self.pair = pair
        self._order_counter = 0
        self.last_price: Decimal | None = None  # stashed by tick loop for display

    @property
    def portfolio_value(self) -> Decimal:
        pos_value = Decimal("0")
        if self.state.open_position is not None:
            pos_value = Decimal(str(self.state.open_position.get("current_value", "0")))
        return self.state.balance_counter + pos_value

    def total_value(self, price: Decimal) -> Decimal:
        # trading capital + vault BTC marked at price
        return self.portfolio_value + self.state.vault_base * price

    def _sweep_profit_to_vault(self, zar_amount: Decimal, price: Decimal) -> Decimal:
        # ponytail: vault = realized wins auto-converted to BTC at market, never sold.
        # Trading capital stays at baseline; wealth compounds in sats. 0.1% Luno fee.
        amt = min(zar_amount, self.state.balance_counter)
        if amt <= 0:
            return Decimal("0")
        vol = (amt * Decimal("0.999")) / price
        self.state.balance_counter -= amt
        self.state.vault_base += vol
        self.state.vault_deposits += amt
        return vol

    def update_open_position_value(self, current_price: Decimal) -> None:
        if self.state.open_position is not None:
            vol = Decimal(self.state.open_position["volume"])
            self.state.open_position["current_value"] = str(vol * current_price)
            entry = Decimal(self.state.open_position["entry_price"])
            change = (current_price - entry) / entry
            self.state.open_position["unrealized_pnl_pct"] = str(change)

    def buy(self, price: Decimal, volume: Decimal, at_ms: int | None = None,
            reason: str = "") -> Order | None:
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
        # at_ms: the backtest replays historical candles, so the fill time must
        # be the candle's, not wall-clock. Stamping now collapsed every backtest
        # trade into the same instant and made holding periods read as zero.
        fill_ms = int(time.time() * 1000) if at_ms is None else at_ms
        order = Order(
            order_id=f"paper_{fill_ms}_{self._order_counter}",
            pair=self.pair,
            side="BUY",
            price=price,
            volume=volume,
            filled_at=fill_ms,
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
                "entry_reason": reason,
            }
        self.state.total_fees += fee
        self.state.total_trades += 1
        self.trade_log.add(order, reason=reason)
        return order

    def sell(self, price: Decimal, volume: Decimal | None = None, reason: str = "signal",
             at_ms: int | None = None) -> Order | None:
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
        fill_ms = int(time.time() * 1000) if at_ms is None else at_ms
        order = Order(
            order_id=f"paper_{fill_ms}_{self._order_counter}",
            pair=self.pair,
            side="SELL",
            price=price,
            volume=volume,
            filled_at=fill_ms,
            fee=fee,
        )

        entry_price = Decimal(self.state.open_position["entry_price"])
        pos_vol = Decimal(self.state.open_position["volume"])
        buy_fee = Decimal(self.state.open_position.get("buy_fee", "0"))
        entry_time = self.state.open_position["entry_time"]
        buy_order_id = self.state.open_position.get("buy_order_id", "")
        entry_reason = self.state.open_position.get("entry_reason", "")

        actual_cost = entry_price * volume
        fee_share = buy_fee * volume / pos_vol
        pnl_actual = net_proceeds - actual_cost - fee_share
        # fee-adjusted return on cost basis (was raw price return — overstated metrics)
        pnl_pct = pnl_actual / actual_cost if actual_cost else Decimal("0")

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
            entry_reason=entry_reason,
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

        current_value = self.total_value(price)
        if current_value > self.state.peak_portfolio_value:
            self.state.peak_portfolio_value = current_value

        if pnl_actual > 0:
            self._sweep_profit_to_vault(pnl_actual, price)

        self.trade_log.add(order)
        self.completed_log.add(completed)
        return order

    def reset_daily_counter(self, today: str) -> None:
        if self.state.last_trade_date != today:
            self.state.trades_today = 0
            self.state.last_trade_date = today
