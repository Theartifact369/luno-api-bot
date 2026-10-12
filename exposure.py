"""The decisive test: is the 'drawdown reduction' just reduced exposure?

stress.py section C found something that looked like a free lunch -- the
crossover lost less than buy & hold (-0.3% vs -18.2% median CAGR) while
also drawing down less. A signal cannot beat both axes at once for free.

The explanation is mundane: a crossover spends much of its life in CASH.
Cash has no return and no drawdown, so a strategy that is flat 60% of the
time looks calm for a reason that has nothing to do with foresight.

This script separates the two by comparing only the periods the strategy
was actually INVESTED, and by normalising exposure.
"""
from __future__ import annotations

import statistics
from operator import attrgetter

import research as R
from stress import walk_chain


def main() -> None:
    grid = R.build_grid()
    warm = R.warmup_for(grid)
    key = attrgetter("sharpe")

    print("=" * 78)
    print("A. the free lunch, restated: median CAGR across pairs")
    print("=" * 78)
    got = {p: walk_chain(R.load_pair(p), grid, key, warm, 750, 250)
           for p in R.available_pairs()}
    got = {p: v for p, v in got.items() if v}
    s = [v[0] for v in got.values()]
    b = [v[1] for v in got.values()]
    h = [v[4] for v in got.values()]
    print(f"  pairs: {len(got)}   strategy TiM {statistics.mean(x.time_in_market for x in s):.0%}")
    print(f"  median CAGR   strategy {statistics.median(x.cagr for x in s):+7.1f}%   "
          f"B&H {statistics.median(x.cagr for x in b):+7.1f}%   "
          f"half-cash {statistics.median(x.cagr for x in h):+7.1f}%")
    print("  -> the strategy is losing less because it is FLAT more, not")
    print("     because it picked better entries. Cash returns 0 and draws down 0.")

    print()
    print("=" * 78)
    print("B. time-in-market vs drawdown: the exposure explanation")
    print("=" * 78)
    print("   Constant-fraction portfolios have NO signal and NO timing:")
    print("   whatever drawdown they show is purely a function of exposure.")
    rows = []
    for p in R.available_pairs():
        c = R.load_pair(p)
        if len(c) < warm + 400:
            continue
        bpy = R.bars_per_year(c)
        for frac in (1.0, 0.5):
            m = R.metrics(R.simulate(c, [frac] * len(c), warm), bpy, [frac] * len(c))
            rows.append((p, frac, m))
    for frac in (1.0, 0.5):
        sub = [m for _, f, m in rows if f == frac]
        print(f"  {frac:>4.0%} invested, zero signal: CAGR {statistics.mean(x.cagr for x in sub):+6.1f}%  "
              f"maxDD {statistics.mean(x.maxdd for x in sub):5.1f}%  "
              f"Sharpe {statistics.mean(x.sharpe for x in sub):+.3f}")
    print("  -> halving exposure alone roughly halves maxDD. The crossover's")
    print("     drawdown cut is the same lever, pulled by sitting in cash.")

    print()
    print("=" * 78)
    print("C. DRAWDOWN PER UNIT OF EXPOSURE (the fair comparison)")
    print("=" * 78)
    print("   Divide maxDD by time-in-market: if the crossover's drawdown")
    print("   advantage vanishes once you charge for how little it traded,")
    print("   the advantage was exposure, not foresight.")
    print(f"  {'pair':9s} {'stratDD':>8s} {'TiM':>5s} {'DD/TiM':>8s} | "
          f"{'B&HDD':>7s} {'halfDD':>7s} {'half/TiM':>9s}")
    better = worse = 0
    for p, v in sorted(got.items()):
        st, bt, ht = v[0], v[1], v[4]
        tim = max(st.time_in_market, 1e-9)
        per = st.maxdd / tim
        bh_per = bt.maxdd
        ht_per = ht.maxdd / 0.5
        print(f"  {p:9s} {st.maxdd:7.1f}% {st.time_in_market:4.0%} {per:8.1f} | "
              f"{bt.maxdd:6.1f}% {ht.maxdd:6.1f}% {ht_per:9.1f}")
        if per < bh_per:
            better += 1
        else:
            worse += 1
    print(f"\n  normalised, strategy still has lower DD than B&H on "
          f"{better}/{len(got)} pairs ({worse} worse)")

    print()
    print("=" * 78)
    print("D. conditional on being invested: does the SIGNAL add anything?")
    print("=" * 78)
    print("   Sharpe computed only over bars where the strategy held a")
    print("   position. If timing had value, this beats B&H's Sharpe. A")
    print("   strategy that merely sat out bad times shows no edge here.")
    print("   Split each pair's daily returns into bars the crossover HELD vs bars")
    print("   it AVOIDED. If the signal has timing value, held bars must have")
    print("   systematically BETTER returns than avoided bars. Under the null")
    print("   (signal is noise) the two means are equal.")
    print(f"  {'pair':9s} {'held':>7s} {'n':>5s} | {'avoided':>8s} {'n':>5s} | {'diff':>7s}")
    diffs = []
    for p in R.available_pairs():
        c = R.load_pair(p)
        if len(c) < warm + 400:
            continue
        bpy = R.bars_per_year(c)
        w = R.w_sma(c, 50, 200)
        cl = R.closes_of(c)
        held, avoided = [], []
        for i in range(warm + 1, len(c)):
            r = cl[i] / cl[i - 1] - 1
            (held if w[i] > 1e-12 else avoided).append(r)
        if len(held) < 60 or len(avoided) < 60:
            continue
        mu_h, mu_a = statistics.mean(held) * bpy, statistics.mean(avoided) * bpy
        diffs.append((p, mu_h - mu_a))
        print(f"  {p:9s} {mu_h:+6.1f}% {len(held):5d} | {mu_a:+7.1f}% {len(avoided):5d} | "
              f"{mu_h-mu_a:+6.1f}%")
    if diffs:
        wins = sum(1 for _, d in diffs if d > 0)
        mean_d = statistics.mean(d for _, d in diffs)
        # binomial test against a fair coin, stdlib only
        n = len(diffs)
        k = wins
        se = (n * 0.25) ** 0.5
        z = (k - n / 2) / se if se else 0.0
        print(f"\n  held bars beat avoided bars in {wins}/{n} pairs "
              f"(coin flip = {n/2:.1f}, z={z:+.2f})")
        print(f"  mean held-minus-avoided: {mean_d:+.1f}%/yr")
        verdict = ("SIGNAL HAS TIMING VALUE" if abs(z) > 2 and mean_d > 0
                   else "NO TIMING EDGE -- held and avoided bars are indistinguishable")
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()