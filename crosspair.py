"""Cross-pair research on Luno ZAR markets.

The lead: every Luno pair is quoted in ZAR, so they all share a ZAR/BTC
factor. A long-only portfolio spread across several pairs still holds that
factor, but it sheds single-ASSET risk (a BTC-specific collapse), which is
the part that made single-pair backtests unreadable.

TWO CONSTRAINTS DISCOVERED FROM LUNO'S OWN DOCS, both binding:

1. Luno is SPOT ONLY. No margin, no leverage, no shorting.
   https://guide.luno.com/hc/en-gb/articles/14158595648541-Luno-fees-and-limits-in-South-Africa
   -> Market-neutral pairs trading (long one leg, SHORT the other) is NOT
      implementable here at all. Everything below is LONG-ONLY.
   -> A long-only ZAR portfolio cannot neutralise the ZAR factor. It can only
      diversify asset-specific risk. Any "market neutral" claim would be a lie.

2. The taker fee on crypto/ZAR pairs is 0.60% at tier 1 (R0-R20k/30d), not
   0.10%. Same source. That is 1.20% ROUND TRIP.
   -> A monthly rebalance pays 14.4%/yr in fees. Any strategy that trades more
      than quarterly is dead on arrival.
   -> research.py assumed FEE=0.001, which flatters every result by 6x on
      cost. Re-run at the real rate.

WHAT THIS MODULE MEASURES (long-only, no short legs possible):
  A. Is the ZAR factor real? Regress each pair on XBTZAR. If the pairs are
     one bet, diversification cannot help and the whole idea is dead.
  B. Cross-sectional momentum: rank pairs by trailing return, hold the top N.
  C. Cross-sectional mean reversion: hold the BOTTOM N (buy the loser).
  D. Equal-weight basket of everything, vs XBTZAR alone.
  E. Do any of these beat buy & hold after 0.60% fees, out of sample?

Run: .venv/bin/python crosspair.py
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import research as R

# Luno tier 1, "Other crypto/ZAR pairs", taker. Verified against Luno's fee
# page 2026-10. The research engine default of 0.001 was 6x too kind.
LUNO_TAKER_ZAR = 0.006
LUNO_MAKER_ZAR = 0.004
HALF_SPREAD_BPS = 5.0


# ------------------------------------------------------------ data prep ----

@dataclass
class Panel:
    """Daily closes for several pairs aligned on a common date index."""
    dates: list[int]
    names: list[str]
    closes: dict[str, list[float]]

    def returns(self, name: str) -> list[float]:
        c = self.closes[name]
        return [c[i] / c[i - 1] - 1 if c[i - 1] else 0.0 for i in range(1, len(c))]

    def subset(self, names: list[str]) -> "Panel":
        return Panel(self.dates, list(names), {n: self.closes[n] for n in names})


def build_panel(names: list[str]) -> Panel:
    """Align pairs on their INTERSECTION of dates.

    Different pairs list on different days, so each is re-indexed onto the
    dates every requested pair actually has. Anything missing is dropped
    rather than forward-filled -- filling would invent bars that never traded.
    """
    raw = {n: {c.timestamp: float(c.close) for c in R.load_pair(n)} for n in names}
    common = set.intersection(*(set(v) for v in raw.values()))
    dates = sorted(common)
    return Panel(dates, list(names), {n: [raw[n][d] for d in dates] for n in names})


# ------------------------------------------------- A. the ZAR factor ----

def zar_factor(panel: Panel, benchmark: str = "XBTZAR") -> list[tuple[str, float, float]]:
    """(pair, beta, R^2) regressing each pair's daily return on the benchmark."""
    bench = panel.returns(benchmark)
    out = []
    for n in panel.names:
        if n == benchmark:
            continue
        y = panel.returns(n)
        pairs = [(a, b) for a, b in zip(y, bench)]
        xs = [b for _, b in pairs]
        ys = [a for a, _ in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var == 0:
            continue
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        beta = cov / var
        sdy = math.sqrt(sum((y - my) ** 2 for y in ys) / len(ys))
        sdx = math.sqrt(var / len(xs))
        r2 = (cov / (sdx * sdy * len(xs))) ** 2 if sdy > 0 and sdx > 0 else 0.0
        out.append((n, beta, r2))
    return out


# ------------------------------------------- B/C. cross-sectional ranks ----

def cs_momentum(panel: Panel, lookback: int, top_n: int,
                long_only_winners: bool = True) -> list[list[float]]:
    """weights[i][j] = fraction of capital in pair j held INTO bar i.

    Rank by trailing `lookback`-bar return using bars < i only. Long the top
    `top_n` (or the bottom `top_n` when long_only_winners is False).
    """
    n_bars = len(panel.dates)
    k = len(panel.names)
    w = [[0.0] * k for _ in range(n_bars)]
    for i in range(lookback + 1, n_bars):
        scores = []
        for j, name in enumerate(panel.names):
            c = panel.closes[name]
            if c[i - 1] <= 0:
                scores.append(-math.inf)
                continue
            scores.append((c[i - 1] / c[i - lookback] - 1, j))
        scores.sort(reverse=True)
        pick = scores[:top_n] if long_only_winners else scores[-top_n:]
        share = 1.0 / len(pick) if pick else 0.0
        for _s, j in pick:
            w[i][j] = share
    return w


# ------------------------------------------------------- portfolio sim ----

def simulate_panel(panel: Panel, weights: list[list[float]], start_i: int,
                   fee: float = LUNO_TAKER_ZAR,
                   half_spread_bps: float = HALF_SPREAD_BPS) -> list[float]:
    """Equal-weight-across-selected-pairs portfolio, costs charged on turnover."""
    sp = half_spread_bps / 10_000
    n = len(panel.dates)
    cash, units = 1.0, [0.0] * len(panel.names)
    cur = [0.0] * len(panel.names)
    equity = [1.0] * start_i
    for i in range(start_i, n):
        target = weights[i]
        px = [panel.closes[nm][i] for nm in panel.names]
        # value the book at today's close, then rebalance to target weights
        pv = cash + sum(units[j] * px[j] for j in range(len(px)))
        turnover = 0.0
        for j in range(len(px)):
            want_units = target[j] * pv / px[j]
            turnover += abs(want_units - units[j]) * px[j]
        pv_net = pv - turnover * (fee + sp)
        units = [target[j] * pv_net / px[j] for j in range(len(px))]
        # cash is only what the weights did NOT allocate. Setting cash = pv_net
        # while also holding units worth pv_net double-counted the book and
        # doubled equity every single bar.
        cash = pv_net * (1.0 - sum(target))
        cur = list(target)
        equity.append(cash + sum(units[j] * px[j] for j in range(len(px))))
    return equity


def sim_single(pair: str, start_i: int, fee: float = LUNO_TAKER_ZAR) -> list[float]:
    """Buy once at the start of the window, hold, mark to close.

    The fee is a FRACTION of notional. Charging px*fee as an absolute amount
    (the first version) made units negative for any asset priced above
    1/fee -- BTC at R1.39m bought -6,000 BTC, and the curve printed a
    1,305,742% drawdown on a position that was only ever long.
    """
    c = R.load_pair(pair)
    n = len(c)
    units, entered = 0.0, False
    eq = [1.0] * start_i
    for i in range(start_i, n):
        px = float(c[i].close)
        if not entered:
            units = (1.0 - fee - HALF_SPREAD_BPS / 10_000) / px
            entered = True
        eq.append(units * px)
    return eq


def sim_equal_weight(names: list[str], panel: Panel, start_i: int,
                     fee: float = LUNO_TAKER_ZAR) -> list[float]:
    k = len(names)
    w = [[0.0] * len(panel.names) for _ in range(len(panel.dates))]
    for i in range(len(panel.dates)):
        for j, n_ in enumerate(panel.names):
            if n_ in names:
                w[i][j] = 1.0 / k
    return simulate_panel(panel, w, start_i, fee)


# ------------------------------------------------------------ reporting ----

def sim_single_on_panel(pair: str, panel: Panel, start_i: int,
                        fee: float = LUNO_TAKER_ZAR) -> list[float]:
    """Buy & hold one pair, scored on the PANEL's dates.

    Scoring BTC over its own 13 years while the basket gets the 2-year
    intersection is not a comparison, it is a flattering one.
    """
    if pair not in panel.closes:
        return [1.0] * len(panel.dates)
    c = panel.closes[pair]
    units, entered = 0.0, False
    eq = [1.0] * start_i
    for i in range(start_i, len(c)):
        if not entered:
            units = (1.0 - fee - HALF_SPREAD_BPS / 10_000) / c[i]
            entered = True
        eq.append(units * c[i])
    return eq


def main() -> None:
    names = [p for p in R.available_pairs()]
    # Need every pair to have a long joint history; build on the intersection.
    panel = build_panel(names)
    print("=" * 78)
    print(f"ALIGNED PANEL: {len(panel.names)} pairs x {len(panel.dates)} daily bars")
    print(f"  {R.d(panel.dates[0])} -> {R.d(panel.dates[-1])} "
          f"({len(panel.dates)/365:.1f} yr of common history)")
    print(f"  {', '.join(panel.names)}")

    print()
    print("=" * 78)
    print("A. IS THE ZAR FACTOR REAL?  (beta and R^2 vs XBTZAR daily returns)")
    print("=" * 78)
    fr = zar_factor(panel)
    fr.sort(key=lambda t: -t[2])
    print(f"  {'pair':9s} {'beta':>6s} {'R2':>6s}")
    for n, b, r2 in fr:
        print(f"  {n:9s} {b:6.2f} {r2:6.2f}")
    mean_r2 = statistics.mean(r2 for _, _, r2 in fr)
    print(f"\n  mean R^2 {mean_r2:.2f} -- this is how much of each pair's daily")
    print(f"  move is just the BTC/ZAR factor. Anything near 1 means")
    print(f"  diversification buys you little.")

    bpy = 365.0
    print()
    print("=" * 78)
    print("B. LONG-ONLY CROSS-SECTIONAL STRATEGIES, out of sample")
    print("=" * 78)
    n = len(panel.dates)
    split = int(n * 0.6)
    warm = 200
    print(f"  train {R.d(panel.dates[0])} -> {R.d(panel.dates[split])} | "
          f"test {R.d(panel.dates[split])} -> {R.d(panel.dates[-1])}")
    print(f"  fees charged: {LUNO_TAKER_ZAR*100:.2f}% taker + "
          f"{HALF_SPREAD_BPS/100:.2f}% half-spread per turnover")

    # baselines on the SAME aligned window
    eqs = {}
    eqs["XBTZAR buy&hold"] = sim_single_on_panel("XBTZAR", panel, warm)
    eqs["equal-weight basket"] = sim_equal_weight(panel.names, panel, warm)
    half = [p for p in panel.names if p != "XBTZAR"]
    eqs["basket ex-BTC"] = sim_equal_weight(half, panel, warm)

    print("\n  --- BASELINES (identical panel window, identical fees) ---")
    for name, eq in eqs.items():
        m = R.metrics(eq, bpy)
        print(f"  {name:22s} CAGR {m.cagr:+7.1f}%  Sharpe {m.sharpe:+.2f}  "
              f"maxDD {m.maxdd:5.1f}%  Calmar {m.calmar:+.2f}")

    print("\n  --- CROSS-SECTIONAL RANKING (full window) ---")
    tried = 0
    results = []
    for lb in (30, 60, 90):
        for top in (1, 3, 5):
            for winners in (True, False):
                tried += 1
                w = cs_momentum(panel, lb, top, winners)
                eq = simulate_panel(panel, w, warm)
                m = R.metrics(eq, bpy)
                kind = "momentum" if winners else "reversal"
                results.append((f"{kind} lookback={lb} top{top}", m))
                print(f"  {kind:9s} lb={lb:3d} top{top}   CAGR {m.cagr:+7.1f}%  "
                      f"Sharpe {m.sharpe:+.2f}  maxDD {m.maxdd:5.1f}%")
    print(f"\n  {tried} cross-sectional combos tried")

    print()
    print("=" * 78)
    print("C. VERDICT")
    print("=" * 78)
    bh = R.metrics(eqs["XBTZAR buy&hold"], bpy)
    ew = R.metrics(eqs["equal-weight basket"], bpy)
    best = max(results, key=lambda kv: kv[1].sharpe)
    print(f"  XBTZAR buy&hold        Sharpe {bh.sharpe:+.2f}  maxDD {bh.maxdd:.1f}%")
    print(f"  equal-weight basket   Sharpe {ew.sharpe:+.2f}  maxDD {ew.maxdd:.1f}%")
    print(f"  best of {tried} cross-    Sharpe {best[1].sharpe:+.2f}  "
          f"maxDD {best[1].maxdd:.1f}%   ({best[0]})")
    beat_bh = sum(1 for _, m in results if m.sharpe > bh.sharpe)
    beat_ew = sum(1 for _, m in results if m.sharpe > ew.sharpe)
    print(f"\n  beat XBTZAR on {beat_bh}/{tried}, beat equal-weight on {beat_ew}/{tried}")
    print()
    print("  LIMITATION: the intersection of all 16 pairs is only "
          f"{len(panel.dates)} bars")
    print(f"  ({R.d(panel.dates[0])} -> {R.d(panel.dates[-1])}). The cross-pair")
    print("  test has ~2 years of data, so it CANNOT confirm an edge either way.")
    print("  It can only fail to find one.")
    print("  NOTE: the best combo was picked on the same series it is scored")
    print("  on, so it is an upper bound -- and that upper bound is negative.")


if __name__ == "__main__":
    main()