"""Remaining candidate families, measured honestly.

Families already ruled out elsewhere and NOT repeated here:
  - SMA/EMA crossover, Donchian ....... test_edge_finding.py (no OOS edge)
  - cross-sectional momentum/reversal .. crosspair.py (0/18 beat both baselines)

Families this file covers:
  1. VOLATILITY TARGETING on an equal-weight basket. The mechanism is real
     and widely used: size positions inversely to realised volatility so
     every position risks a similar amount. It is NOT a directional signal,
     so it is not a re-run of the crossover failure.
  2. CALENDAR EFFECTS (day-of-week, month-of-year) on XBTZAR hourly bars.
     Crypto has no exchange close, no settlement window and no earnings
     calendar, so any calendar structure would have to come from systematic
     flows -- scheduled weekend maintenance, month-end rebalancing, or
     Asia-hours liquidity. Cheap to test, and the literature is sceptical.
  3. Single-pair long-only dip-buying (RSI-style), the one mean-reversion
     shape that is implementable without shorting.

Every result is scored against buy & hold on the SAME window at Luno's REAL
0.60% crypto/ZAR taker fee.

Run: .venv/bin/python candidates.py
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

import crosspair as C
import research as R

FEE = C.LUNO_TAKER_ZAR


# ------------------------------------------- 1. volatility targeting ----

def vol_target_weights(panel: C.Panel, lookback: int, target_vol: float,
                       per_pair_cap: float = 1.0) -> list[list[float]]:
    """Equal-weight the basket, then scale the whole book to a vol target.

    weights[i][j] proportional to (1/k) * min(cap, target/(sigma_j * sqrt(annual)))
    then renormalised so total <= 1 (spot only, no leverage).
    """
    n_bars, k = len(panel.dates), len(panel.names)
    w = [[0.0] * k for _ in range(n_bars)]
    for i in range(n_bars):
        if i < lookback + 1:
            continue
        raw = []
        for j, name in enumerate(panel.names):
            c = panel.closes[name]
            sigma = R.realized_vol(c, i - 1, lookback)
            if sigma is None or sigma <= 0:
                raw.append(0.0)
                continue
            ann = sigma * math.sqrt(R.bars_per_year(
                [type("X", (), {"timestamp": 0, "close": 0})()] * 1) if False else 365.0)
            raw.append(min(per_pair_cap, target_vol / ann) / k)
        tot = sum(raw)
        if tot <= 1.0 + 1e-12:
            w[i] = raw
        else:
            w[i] = [x / tot for x in raw]
    return w


def main() -> None:
    names = R.available_pairs()
    panel = C.build_panel(names)
    bpy = 365.0
    warm = 200

    print("=" * 78)
    print("1. VOLATILITY-TARGETED EQUAL-WEIGHT BASKET")
    print("=" * 78)
    print(f"  panel {len(panel.dates)} bars {R.d(panel.dates[0])} -> "
          f"{R.d(panel.dates[-1])}, {len(panel.names)} pairs, fee {FEE*100:.2f}%")

    ew = R.metrics(C.sim_equal_weight(panel.names, panel, warm), bpy)
    bh = R.metrics(C.sim_single_on_panel("XBTZAR", panel, warm), bpy)
    print(f"\n  baseline  XBTZAR B&H     Sharpe {bh.sharpe:+.2f}  maxDD {bh.maxdd:5.1f}%")
    print(f"  baseline  equal-weight   Sharpe {ew.sharpe:+.2f}  maxDD {ew.maxdd:5.1f}%")

    tried, rows = 0, []
    for lookback in (14, 30, 60):
        for tv in (0.15, 0.25, 0.40):
            tried += 1
            w = vol_target_weights(panel, lookback, tv)
            eq = C.simulate_panel(panel, w, warm, fee=FEE)
            m = R.metrics(eq, bpy)
            rows.append((f"voltarget lb={lookback} tv={tv:.0%}", m))
            print(f"  lb={lookback:3d} tv={tv:.0%}   CAGR {m.cagr:+7.1f}%  "
                  f"Sharpe {m.sharpe:+.2f}  maxDD {m.maxdd:5.1f}%  "
                  f"Calmar {m.calmar:+.2f}")
    best = max(rows, key=lambda kv: kv[1].sharpe)
    beat = sum(1 for _, m in rows if m.sharpe > ew.sharpe)
    print(f"\n  {tried} combos tried; {beat}/{tried} beat plain equal-weight")
    print(f"  best {best[0]} Sharpe {best[1].sharpe:+.2f} "
          f"(plain equal-weight {ew.sharpe:+.2f})")

    # ------------------------------------------------ 2. calendar ----
    print()
    print("=" * 78)
    print("2. CALENDAR EFFECTS -- XBTZAR hourly bars")
    print("=" * 78)
    hourly = "candles/XBTZAR_3600.json"
    try:
        hc = R.load_pair("XBTZAR", 3600)
    except SystemExit:
        print(f"  no {hourly}; skipping")
        hc = None
    if hc is not None and len(hc) > 500:
        from datetime import datetime, timezone
        closes = [float(c.close) for c in hc]
        rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        stamps = [hc[i].timestamp for i in range(1, len(hc))]
        dow = defaultdict(list)
        hour = defaultdict(list)
        for ts, r in zip(stamps, rets):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            dow[dt.weekday()].append(r)
            hour[dt.hour].append(r)
        names_dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print(f"  {len(rets):,} hourly returns from "
              f"{R.d(hc[0].timestamp)} to {R.d(hc[-1].timestamp)}")
        print(f"\n  {'day':4s} {'n':>7s} {'mean':>10s} {'t-stat':>8s}")
        all_m = statistics.mean(rets)
        all_s = statistics.stdev(rets)
        for d in sorted(dow):
            v = dow[d]
            t = (statistics.mean(v) - all_m) / (all_s / math.sqrt(len(v)))
            print(f"  {names_dow[d]:4s} {len(v):7d} {statistics.mean(v)*100:+9.4f}% {t:+8.2f}")
        print(f"\n  {'hour':>4s} {'n':>7s} {'mean':>10s} {'t-stat':>8s}")
        for h in sorted(hour):
            v = hour[h]
            t = (statistics.mean(v) - all_m) / (all_s / math.sqrt(len(v)))
            print(f"  {h:4d} {len(v):7d} {statistics.mean(v)*100:+9.4f}% {t:+8.2f}")
        max_t = max(abs((statistics.mean(v) - all_m) / (all_s / math.sqrt(len(v))))
                    for v in list(dow.values()) + list(hour.values()))
        print(f"\n  largest |t| across {len(dow)+len(hour)} buckets: {max_t:.2f}")
        print("  with ~33 buckets, a |t| of 2.0+ is EXPECTED by chance alone.")
        print(f"  verdict: {'SUSPECT' if max_t > 3.5 else 'nothing survives multiple testing'}")

    # ------------------------------------------ 3. single-pair dip buy ----
    print()
    print("=" * 78)
    print("3. LONG-ONLY DIP BUYING (RSI) on XBTZAR daily")
    print("=" * 78)
    c = R.load_pair("XBTZAR")
    cl = R.closes_of(c)
    for oversold in (25, 30, 35):
        w = []
        for i in range(len(cl)):
            if i < 15:
                w.append(0.0)
                continue
            seg = cl[max(0, i - 14):i + 1]
            gains = sum(max(0.0, seg[j] - seg[j - 1]) for j in range(1, len(seg)))
            losses = sum(max(0.0, seg[j - 1] - seg[j]) for j in range(1, len(seg)))
            val = 100.0 if losses == 0 else 100 - 100 / (1 + gains / losses)
            w.append(1.0 if val <= oversold else 0.0)
        eq = C.simulate_panel(C.build_panel(["XBTZAR"]),
                              [[w[i]] for i in range(len(w))], 200, fee=FEE)
        m = R.metrics(eq, bpy)
        print(f"  RSI<{oversold}   CAGR {m.cagr:+7.1f}%  Sharpe {m.sharpe:+.2f}  "
              f"maxDD {m.maxdd:5.1f}%  in-market "
              f"{sum(1 for x in w[200:] if x > 0)/len(w[200:]):.0%}")
    print(f"  XBTZAR B&H   CAGR {R.metrics(C.sim_single('XBTZAR', 200), bpy).cagr:+7.1f}%")


if __name__ == "__main__":
    main()