"""Vault self-check: realized wins sweep to BTC, losses never touch the vault."""
from decimal import Decimal

from paper_trader import CompletedTradeLog, PaperTrader, State, TradeLog


def run() -> None:
    s = State.initial(0.0, 100000.0)
    t = PaperTrader(s, TradeLog(), CompletedTradeLog(), "XBTZAR")

    # Win: buy 0.01 @ 1,000,000, sell @ 1,100,000 -> pnl ~ +9,879 ZAR swept
    t.buy(Decimal("1000000"), Decimal("0.01"))
    t.sell(Decimal("1100000"), reason="take_profit")
    assert s.vault_base > 0, "win must sweep profit into vault"
    assert s.vault_deposits > 0
    assert s.vault_deposits * Decimal("0.999") / Decimal("1100000") == s.vault_base
    vault_after_win = s.vault_base

    # Loss: vault untouched
    t.buy(Decimal("1000000"), Decimal("0.005"))
    t.sell(Decimal("900000"), reason="stop_loss")
    assert s.vault_base == vault_after_win, "loss must not change vault"

    # Vault excluded from trading capital but counted in total_value
    assert t.portfolio_value < 100000  # profit left the capital
    assert t.total_value(Decimal("1100000")) == t.portfolio_value + s.vault_base * Decimal("1100000")

    # Round-trip persistence
    s.save("/tmp/vault_state_test.json")
    s2 = State.load("/tmp/vault_state_test.json", 0.0, 100000.0)
    assert s2.vault_base == s.vault_base and s2.vault_deposits == s.vault_deposits

    print(f"vault ok: {s.vault_base:.8f} BTC from R{s.vault_deposits:,.2f} swept")


if __name__ == "__main__":
    run()
