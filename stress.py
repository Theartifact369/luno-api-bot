"""Is the drawdown reduction real, or is it just 'hold cash more'?

The walk-forward found the crossover never beat buy & hold on return
(Sharpe +0.065, 4/8 pairs) but cut drawdown on 8/8. Before believing that,
it has to survive the boring disproofs:

  * DCA buys every day and is ALWAYS 100% invested -- it has B&H's return
    risk with none of the timing freedom. If the crossover cannot beat DCA,
    it certainly cannot beat holding.
  * Vol-targeting mostly works by holding LESS. If a constant 50% position
    does as well, the crossover signal is decoration.
  * Walk-forward on multiple train/test splits and objectives.
  * Bigger drawdown cut should cost return. Free lunch = artefact.
"""
from __future__ import annotations

import statistics
from operator import attrgetter

import research as R


def walk_chain(candles, grid, key, warm, train_bars, test_bars):
    """Pure-OOS equity chain; returns (strategy_metrics, bh_metrics)."""
    n, bpy = len(candles), R.bars_per_year(candles)
    if n < warm + train_bars + test_bars + 10:
        return None
    eqs = {R.label(s): R.simulate(candles, R.weights_for(candles, s), warm) for s in grid}
    bh = R.simulate(candles, R.w_buy_hold(candles), warm)
    dca = R.simulate(candles, R.w_dca(candles, 30), warm)
    # The honest null for "a crossover beats holding": hold a FIXED fraction.
    # No signal, no timing, always invested at the same weight.
    half = R.simulate(candles, [0.5] * len(candles), warm)
    chain, bh_chain, dca_chain, half_chain, chosen = [1.0], [1.0], [1.0], [1.0], []
    # chain_w[0] pairs with chain[0]: the block's opening weight. metrics()
    # ignores weights whose length differs from the equity series, which
    # silently reported time-in-market 0%.
    chain_w: list[float] = [0.0]
    i = warm + train_bars
    while i + test_bars <= n:
        best = max(grid, key=lambda s: key(R.metrics(eqs[R.label(s)][warm:i], bpy)))
        w = R.weights_for(candles, best)
        chosen.append(R.label(best))
        seg = eqs[R.label(best)][i:i + test_bars]
        bseg, dseg, hseg = bh[i:i + test_bars], dca[i:i + test_bars], half[i:i + test_bars]
        chain_w.extend(w[i:i + test_bars][1:])
        a, b, c, h = chain[-1], bh_chain[-1], dca_chain[-1], half_chain[-1]
        chain.extend([a * (e / seg[0]) for e in seg[1:]])
        bh_chain.extend([b * (e / bseg[0]) for e in bseg[1:]])
        dca_chain.extend([c * (e / dseg[0]) for e in dseg[1:]])
        half_chain.extend([h * (e / hseg[0]) for e in hseg[1:]])
        i += test_bars
    if len(chain) < 60:
        return None
    return (R.metrics(chain, bpy, chain_w), R.metrics(bh_chain, bpy),
            R.metrics(dca_chain, bpy), chosen, R.metrics(half_chain, bpy))


def constant_weight(candles, frac, warm):
    return R.simulate(candles, [frac] * len(candles), warm)


def main() -> None:
    grid = R.build_grid()
    warm = R.warmup_for(grid)
    key = attrgetter("sharpe")

    print("=" * 78)
    print("A. constant-fraction baselines (no signal at all), full OOS window")
    print("=" * 78)
    print(f"{'pair':9s} {'constant':>9s} | {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'TiM':>5s}")
    rows = []
    for p in R.available_pairs():
        c = R.load_pair(p)
        if len(c) < warm + 400:
            continue
        bpy = R.bars_per_year(c)
        # simulate() returns one point per bar INCLUDING warmup, so weights
        # must be the full series -- slicing here silently misaligned them.
        bh = R.metrics(R.simulate(c, R.w_buy_hold(c), warm), bpy, R.w_buy_hold(c))
        for frac in (1.0, 0.75, 0.5):
            m = R.metrics(constant_weight(c, frac, warm), bpy, [frac] * len(c))
            rows.append((p, frac, m, bh))
            tag = " (=B&H)" if frac == 1.0 else ""
            print(f"{p:9s} {frac:>8.0%}  | {m.cagr:+6.1f}% {m.sharpe:+7.2f} "
                  f"{m.maxdd:6.1f}% {m.time_in_market:4.0f}%{tag}")
    print()
    for frac in (1.0, 0.75, 0.5):
        sub = [m for _, f, m, _ in rows if f == frac]
        bh_m = [bh for _, _, _, bh in rows]
        print(f"  mean {frac:.0%} position: CAGR {statistics.mean(x.cagr for x in sub):+6.1f}%  "
              f"Sharpe {statistics.mean(x.sharpe for x in sub):+.3f}  "
              f"maxDD {statistics.mean(x.maxdd for x in sub):5.1f}%  "
              f"(B&H maxDD {statistics.mean(x.maxdd for x in bh_m):5.1f}%)")

    print()
    print("=" * 78)
    print("B. walk-forward: crossover vs B&H vs DCA (DCA is always 100% invested)")
    print("=" * 78)
    for train_bars, test_bars in ((750, 250), (500, 250), (1000, 250), (750, 500)):
        got = {}
        for p in R.available_pairs():
            r = walk_chain(R.load_pair(p), grid, key, warm, train_bars, test_bars)
            if r:
                got[p] = r
        if not got:
            continue
        s = [v[0] for v in got.values()]
        b = [v[1] for v in got.values()]
        d = [v[2] for v in got.values()]
        hf = [v[4] for v in got.values()]
        print(f"  train {train_bars:4d} / test {test_bars:3d} ({len(got)} pairs): "
              f"Sharpe strat {statistics.mean(x.sharpe for x in s):+.3f}  "
              f"B&H {statistics.mean(x.sharpe for x in b):+.3f}  "
              f"DCA {statistics.mean(x.sharpe for x in d):+.3f}  "
              f"half-cash {statistics.mean(x.sharpe for x in hf):+.3f}")
        print(f"  {'':22s} beats B&H {sum(1 for x, y in zip(s, b) if x.sharpe > y.sharpe):2d}/{len(s)}"
              f"  beats DCA {sum(1 for x, y in zip(s, d) if x.sharpe > y.sharpe):2d}/{len(s)}"
              f"  beats half-cash {sum(1 for x, y in zip(s, hf) if x.sharpe > y.sharpe):2d}/{len(s)}"
              f"  <- the null that matters")
        print(f"  {'':22s} maxDD  strat {statistics.mean(x.maxdd for x in s):5.1f}%  "
              f"B&H {statistics.mean(x.maxdd for x in b):5.1f}%  "
              f"half-cash {statistics.mean(x.maxdd for x in hf):5.1f}%  |  "
              f"lower DD than half-cash "
              f"{sum(1 for x, y in zip(s, hf) if x.maxdd < y.maxdd)}/{len(s)}  "
              f"TiM {statistics.mean(x.time_in_market for x in s):.0%}")

    print()
    print("=" * 78)
    print("C. does the drawdown cut cost return? (a free lunch would be an artefact)")
    print("=" * 78)
    got = {}
    for p in R.available_pairs():
        r = walk_chain(R.load_pair(p), grid, key, warm, 750, 250)
        if r:
            got[p] = r
    s = [v[0] for v in got.values()]
    b = [v[1] for v in got.values()]
    hf = [v[4] for v in got.values()]
    print(f"  strategy CAGR median {statistics.median(x.cagr for x in s):+6.1f}%  vs "
          f"B&H median {statistics.median(x.cagr for x in b):+6.1f}%  vs "
          f"half-cash {statistics.median(x.cagr for x in hf):+6.1f}%")
    print(f"  strategy Calmar mean {statistics.mean(x.calmar for x in s):+.3f}  vs "
          f"B&H {statistics.mean(x.calmar for x in b):+.3f}  vs "
          f"half-cash {statistics.mean(x.calmar for x in hf):+.3f}")
    wins = [(x, y, h) for x, y, h in zip(s, b, hf) if x.maxdd < y.maxdd]
    print(f"  on the {len(wins)} pairs with lower drawdown than B&H:")
    print(f"    CAGR  strat {statistics.mean(x.cagr for x, _, _ in wins):+6.1f}%  "
          f"B&H {statistics.mean(y.cagr for _, y, _ in wins):+6.1f}%  "
          f"half-cash {statistics.mean(h.cagr for _, _, h in wins):+6.1f}%")
    print(f"    Sharpe strat {statistics.mean(x.sharpe for x, _, _ in wins):+.3f}  "
          f"B&H {statistics.mean(y.sharpe for _, y, _ in wins):+.3f}  "
          f"half-cash {statistics.mean(h.sharpe for _, _, h in wins):+.3f}")
    print(f"    -> the drawdown cut is paid for out of return: "
          f"{'CONFIRMED' if statistics.mean(x.cagr for x, _, _ in wins) < statistics.mean(y.cagr for _, y, _ in wins) else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    main()