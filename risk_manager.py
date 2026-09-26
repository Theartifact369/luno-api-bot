from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


@dataclass
class RiskStatus:
    trading_allowed: bool = True
    reason: str = ""
    daily_loss_hit: bool = False
    max_drawdown_hit: bool = False
    consecutive_loss_hit: bool = False
    cooldown_active: bool = False
    cooldown_until: float = 0.0
    cooldown_just_expired: bool = False


class RiskManager:
    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,
        max_drawdown_pct: float = 15.0,
        max_consecutive_losses: int = 5,
        cooldown_hours: float = 4.0,
        volatility_position_sizing: bool = True,
        base_position_size_pct: float = 0.3,
        max_position_size_pct: float = 0.5,
        min_position_size_pct: float = 0.1,
    ) -> None:
        self.max_daily_loss_pct = Decimal(str(max_daily_loss_pct))
        self.max_drawdown_pct = Decimal(str(max_drawdown_pct))
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_hours = Decimal(str(cooldown_hours))
        self.volatility_position_sizing = volatility_position_sizing
        self.base_position_size_pct = Decimal(str(base_position_size_pct))
        self.max_position_size_pct = Decimal(str(max_position_size_pct))
        self.min_position_size_pct = Decimal(str(min_position_size_pct))
        self.status = RiskStatus()

        self._daily_start_value: Decimal | None = None
        self._daily_date: str = ""
        # persisted on the instance: a cooldown set once must expire on its own,
        # not restart every scan (the old behavior blocked trading forever)
        self._cooldown_until: float | None = None

    def reset_daily(self, portfolio_value: Decimal, today: str) -> None:
        # date-gated: called every scan, so an ungated reset re-zeroed the
        # baseline each minute and the daily-loss breaker could never fire
        if today != self._daily_date:
            self._daily_date = today
            self._daily_start_value = portfolio_value

    def check_circuit_breakers(
        self,
        portfolio_value: Decimal,
        peak_value: Decimal,
        consecutive_losses: int,
        trades_today: int,
        max_daily_trades: int,
    ) -> RiskStatus:
        self.status = RiskStatus(trading_allowed=True, reason="")

        if self._daily_start_value is not None:
            daily_loss = (self._daily_start_value - portfolio_value) / self._daily_start_value * Decimal("100")
            if daily_loss >= self.max_daily_loss_pct:
                self.status.trading_allowed = False
                self.status.daily_loss_hit = True
                self.status.reason = f"Daily loss limit ({self.max_daily_loss_pct}%) hit: {daily_loss:.2f}%"
                return self.status

        if peak_value > 0:
            drawdown = (peak_value - portfolio_value) / peak_value * Decimal("100")
            if drawdown >= self.max_drawdown_pct:
                self.status.trading_allowed = False
                self.status.max_drawdown_hit = True
                self.status.reason = f"Max drawdown ({self.max_drawdown_pct}%) hit: {drawdown:.2f}%"
                return self.status

        if consecutive_losses >= self.max_consecutive_losses:
            now = datetime.now(tz=timezone.utc).timestamp()
            if self._cooldown_until is None:
                self._cooldown_until = now + float(self.cooldown_hours) * 3600
            if now < self._cooldown_until:
                remaining = (self._cooldown_until - now) / 3600
                self.status.trading_allowed = False
                self.status.cooldown_active = True
                self.status.cooldown_until = self._cooldown_until
                self.status.consecutive_loss_hit = True
                self.status.reason = (
                    f"Cooldown after {consecutive_losses} consecutive losses "
                    f"({remaining:.1f}h remaining)"
                )
                return self.status
            # cooldown served: clear the counter (caller resets consecutive_losses)
            self._cooldown_until = None
            self.status.cooldown_just_expired = True

        return self.status

    def compute_position_size(
        self,
        portfolio_value: Decimal,
        current_price: Decimal,
        atr: Decimal | None = None,
    ) -> Decimal:
        if self.volatility_position_sizing and atr is not None and atr > 0:
            volatility_ratio = atr / current_price
            if volatility_ratio > Decimal("0.03"):
                size = self.base_position_size_pct * Decimal("0.5")
            elif volatility_ratio > Decimal("0.02"):
                size = self.base_position_size_pct * Decimal("0.75")
            else:
                size = self.base_position_size_pct
        else:
            size = self.base_position_size_pct

        size = max(self.min_position_size_pct, min(size, self.max_position_size_pct))
        return size
