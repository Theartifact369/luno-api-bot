from __future__ import annotations

from decimal import Decimal
from typing import Sequence


def compute_rsi(prices: Sequence[Decimal], period: int = 14) -> Decimal | None:
    if len(prices) < period + 1:
        return None
    gains, losses = Decimal("0"), Decimal("0")
    for i in range(-period, 0):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


def compute_ema(prices: Sequence[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(prices)
    if len(prices) < period:
        return result
    multiplier = Decimal("2") / Decimal(str(period + 1))
    ema = sum(prices[:period]) / Decimal(str(period))
    result[period - 1] = ema
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * multiplier + ema
        result[i] = ema
    return result


def compute_macd(
    prices: Sequence[Decimal], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[Decimal | None], list[Decimal | None], list[Decimal | None]]:
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    macd_line: list[Decimal | None] = [None] * len(prices)
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    signal_line = compute_ema(
        [m for m in macd_line if m is not None],
        signal,
    )
    sig_idx = 0
    padded_signal: list[Decimal | None] = [None] * len(prices)
    histogram: list[Decimal | None] = [None] * len(prices)
    for i in range(len(prices)):
        if macd_line[i] is not None:
            if sig_idx < len(signal_line) and signal_line[sig_idx] is not None:
                padded_signal[i] = signal_line[sig_idx]
                histogram[i] = macd_line[i] - padded_signal[i]
            sig_idx += 1
    return macd_line, padded_signal, histogram


def compute_sma(prices: Sequence[Decimal], period: int) -> list[Decimal | None]:
    result: list[Decimal | None] = [None] * len(prices)
    if len(prices) < period:
        return result
    window_sum = sum(prices[:period])
    result[period - 1] = window_sum / period
    for i in range(period, len(prices)):
        window_sum += prices[i] - prices[i - period]
        result[i] = window_sum / period
    return result
