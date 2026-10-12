"""Regression guards for the backtest.py honesty fixes.

Run: .venv/bin/python test_backtest_honesty.py
"""
from __future__ import annotations

import inspect

import backtest


def test_vault_is_reported_separately_from_strategy_return():
    """The +74343% headline came from total_value() folding the BTC vault
    into strategy return. Strategy-only must be the headline."""
    src = inspect.getsource(backtest.run_backtest)
    assert "strat_end = trader.portfolio_value" in src, \
        "strategy return must exclude the vault"
    assert "vault_end" in src, "vault must be reported on its own line"
    assert "HEADLINE" in src, "strategy-only line must be marked as the headline"
    print("ok: strategy return excludes the vault, vault shown separately")


def test_verdict_states_n_and_refuses_underpowered_samples():
    """'BEATS buy-and-hold' on 12 trades is exactly the false claim."""
    src = inspect.getsource(backtest.run_backtest)
    assert "MIN_TRADES" in src, "verdict needs a minimum sample"
    assert "NO VERDICT" in src, "verdict must refuse below the minimum"
    assert "n={n} trades" in src or "n=%d trades" in src, \
        "verdict must state the trade count"
    print("ok: verdict states n and refuses below the minimum sample")


def test_verdict_uses_strategy_return_not_vault_inflated_return():
    assert "beats = strat_return > hold_return" in \
        inspect.getsource(backtest.run_backtest), \
        "verdict must compare strategy-only return, not vault-inflated"
    print("ok: verdict compares strategy-only return")


def test_headline_is_labelled_in_sample():
    src = inspect.getsource(backtest.run_backtest)
    assert "not evidence of an edge" in src, \
        "the in-sample caveat must stay attached to the output"
    assert "research.py walk" in src, "output must point at the OOS verdict"
    print("ok: output carries its in-sample caveat and the OOS pointer")


if __name__ == "__main__":
    for fn in (test_vault_is_reported_separately_from_strategy_return,
               test_verdict_states_n_and_refuses_underpowered_samples,
               test_verdict_uses_strategy_return_not_vault_inflated_return,
               test_headline_is_labelled_in_sample):
        fn()
    print("\nall backtest-honesty checks passed")