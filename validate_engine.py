"""Can this harness detect an edge AT ALL? (positive control + placebo)

The objection: "you found several bugs, so how can you claim a strategy
failed?" It is correct, and it is worst for NEGATIVE results specifically.

Every bug found so far was INFATING -- it made returns too good
(portfolio value counted twice, a tautological t-stat). Those announce
themselves as absurd numbers. A bug that DEFLATES, an edge-suppressing
error, produces a plausible-looking mediocre curve and nobody notices. So
"found no edge" is the claim most exposed to bugs I never caught.

Inspecting the code again does not settle it. These two experiments do:

  1. POSITIVE CONTROL. Manufacture data with a KNOWN, exploitable edge and
     run the real search on it. If the search cannot recover an edge that is
     planted in front of it, then its failure to find one in real data means
     nothing. This measures the harness's POWER.

  2. PLACEBO / PERMUTATION. Shuffle the real return series to destroy any
     structure, then run the same search many times. Each run's best-of-N
     test Sharpe is the largest number a search would report on data that is
     pure noise. This is the noise FLOOR -- the bar a real result must clear
     to mean anything.

Read them together:
  power fails      -> the negative result is meaningless (harness is broken)
  placebo floor low, real result below it -> genuinely no edge
  placebo floor high, real result at/below it -> no edge, and you now know
                          how much searching 44 combos costs you in false hope

Run: .venv/bin/python validate_engine.py
"""
from __future__ import annotations

import math
import random
import statistics
from decimal import Decimal

import research as R
from client import Candle


# ------------------------------------------------------- data builders ----

def synth_regime(n: int = 3000, seed: int = 1, regime_len: int = 120,
                 drift: float = 0.006, vol: float = 0.015) -> list[Candle]:
    """Price series with a KNOWN exploitable structure.

    Drift alternates sign every `regime_len` bars: a sustained uptrend, then a
    sustained downtrend, forever. Any trend-follower can capture part of it.
    This is the ground truth the search is supposed to find -- `drift/vol` is
    the signal-to-noise knob the power curve sweeps.
    """
    rnd = random.Random(seed)
    out, price = [], 1000.0
    for i in range(n):
        d = drift if (i // regime_len) % 2 == 0 else -drift
        price *= (1 + d + rnd.gauss(0, vol))
        price = max(1e-6, price)
        out.append(Candle(timestamp=1_500_000_000_000 + i * 86_400_000,
                          open=Decimal(str(price)), high=Decimal(str(price * 1.01)),
                          low=Decimal(str(price * 0.99)), close=Decimal(str(price)),
                          volume=Decimal("1")))
    return out


def synth_random_walk(n: int = 3000, seed: int = 2,
                      vol: float = 0.03) -> list[Candle]:
    """Pure driftless random walk: by construction, NO edge exists here."""
    rnd = random.Random(seed)
    out, price = [], 1000.0
    for i in range(n):
        price = max(1e-6, price * (1 + rnd.gauss(0, vol)))
        out.append(Candle(timestamp=1_500_000_000_000 + i * 86_400_000,
                          open=Decimal(str(price)), high=Decimal(str(price * 1.01)),
                          low=Decimal(str(price * 0.99)), close=Decimal(str(price)),
                          volume=Decimal("1")))
    return out


def shuffled_real(pair: str, seed: int) -> list[Candle]:
    """Real XBTZAR prices with the daily returns randomly permuted.

    NOTE: this is NOT a null for "can the strategy make money". Shuffling
    preserves the multiset of returns, so the mean drift survives -- it is a
    POSITIVE-DRIFT random walk. The first version of this placebo mislabelled
    it "pure noise" and so reported +0.78 Sharpe as if it were a null, which
    invited the false conclusion that the engine was broken.

    What it correctly IS: the null for TIMING. Same drift, same volatility
    distribution, zero serial dependence. Any Sharpe the strategy reports here
    is what you get from riding drift with a lucky-looking chart -- so it is
    the floor a real timing skill must beat.
    """
    c = R.load_pair(pair)
    closes = [float(x.close) for x in c]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    rng = random.Random(seed)
    rng.shuffle(rets)
    out, price = [], closes[0]
    stamps = [x.timestamp for x in c]
    for i, r in enumerate(rets):
        price = max(1e-9, price * (1 + r))
        out.append(Candle(timestamp=stamps[i + 1], open=Decimal(str(price)),
                          high=Decimal(str(price * 1.01)),
                          low=Decimal(str(price * 0.99)),
                          close=Decimal(str(price)), volume=Decimal("1")))
    return out


# ------------------------------------------- small fast grid for probing ----

PROBE_GRID = [
    {"kind": "sma", "fast": f, "slow": s}
    for f, s in [(10, 50), (20, 100), (30, 150), (50, 200), (10, 30), (20, 60),
                 (5, 40), (40, 120), (15, 80), (60, 240)]
] + [
    {"kind": "ema", "fast": f, "slow": s}
    for f, s in [(10, 50), (20, 100), (30, 150), (50, 200), (10, 30), (20, 60)]
]


def best_test_sharpe(candles: list[Candle], train_frac: float = 0.6,
                     fee: float = R.FEE) -> tuple[float, float]:
    """(best-of-N test Sharpe, best-of-N TRAIN Sharpe).

    Deliberately mirrors research.py's protocol so the number is comparable:
    choose on train, report on test.
    """
    warm = R.warmup_for(PROBE_GRID)
    n = len(candles)
    split = int(n * train_frac)
    bpy = R.bars_per_year(candles)
    rows = []
    for spec in PROBE_GRID:
        eq = R.simulate(candles, R.weights_for(candles, spec), warm, fee=fee)
        rows.append((R.label(spec),
                     R.metrics(eq[warm:split], bpy),
                     R.metrics(eq[split:], bpy)))
    best_tr = max(rows, key=lambda r: r[1].sharpe)
    return best_tr[2].sharpe, best_tr[1].sharpe


def mean_sharpe(candles: list[Candle], train_frac: float = 0.6,
                fee: float = R.FEE) -> float:
    """Mean test Sharpe across all specs -- less prone to selection noise
    than the max, so it is the honest summary of 'did anything work'."""
    warm = R.warmup_for(PROBE_GRID)
    n = len(candles)
    split = int(n * train_frac)
    bpy = R.bars_per_year(candles)
    vals = []
    for spec in PROBE_GRID:
        eq = R.simulate(candles, R.weights_for(candles, spec), warm, fee=fee)
        vals.append(R.metrics(eq[split:], bpy).sharpe)
    return statistics.mean(vals)


# ---------------------------------------------------------- experiments ----

def positive_control(seeds: int = 10) -> None:
    """Power curve: sweep signal-to-noise and watch the harness respond.

    A single pass/fail threshold is a bad instrument -- it depends entirely
    on where the threshold is drawn, and it gave a false "LOW POWER" verdict
    on data whose edge was simply too weak to owe anyone a detection.

    The graded version asks the real question: as the planted edge gets
    stronger, does reported Sharpe rise? A harness with power must show a
    monotone response. One without stays flat no matter how easy you make it.
    """
    print("=" * 78)
    print("1. POSITIVE CONTROL -- power curve")
    print("=" * 78)
    print("  Synthetic alternating up/downtrend. drift/vol is the planted")
    print("  signal-to-noise. A harness with power must respond to it.\n")
    print(f"  {'drift/vol':>10s} {'mean best-TEST':>15s} {'mean mean-TEST':>15s} "
          f"{'detected':>10s}")
    curve = []
    for snr in (0.0, 0.1, 0.2, 0.4, 0.8):
        vol = 0.015
        drift = snr * vol
        bests, means, det = [], [], 0
        for s in range(seeds):
            c = synth_regime(seed=500 + s, drift=drift, vol=vol)
            bte, _ = best_test_sharpe(c)
            mt = mean_sharpe(c)
            bests.append(bte)
            means.append(mt)
            det += bte > 0.5
        curve.append((snr, statistics.mean(bests), statistics.mean(means), det))
        print(f"  {snr:10.1f} {statistics.mean(bests):15.2f} "
              f"{statistics.mean(means):15.2f} {det:7d}/{seeds}")

    lo = curve[0][1]
    hi = curve[-1][1]
    mono = curve[-1][1] > curve[0][1]
    print(f"\n  Sharpe at snr=0.0: {lo:+.2f}   at snr=0.8: {hi:+.2f}   "
          f"delta {hi-lo:+.2f}")
    if mono and (hi - lo) > 1.0:
        print("  -> POWER CONFIRMED: reported Sharpe rises with planted signal.")
        print("     The harness recognises an edge when one exists, so its")
        print("     silence on real data is informative rather than vacuous.")
    else:
        print("  -> POWER NOT ESTABLISHED. A 'no edge' finding from this")
        print("     harness would be uninformative in either direction.")


def placebo(n_shuffles: int = 12) -> None:
    print()
    print("=" * 78)
    print("2. PLACEBO -- what does this search report on pure NOISE?")
    print("=" * 78)
    print("  Real XBTZAR returns, randomly permuted: same distribution,")
    print("  no serial structure. Any 'edge' here is selection noise.\n")

    print("  Control A: real XBTZAR returns PERMUTED. Same drift and")
    print("  volatility, zero serial structure. This is the null for TIMING:")
    print("  it shows what the strategy earns by riding drift with no skill.")
    print("  Control B: a DRIFTLESS random walk -- the null for making money")
    print("  at all. A trend follower here must score <= 0 (it pays costs).\n")

    shuf_max, shuf_mean = [], []
    for seed in range(n_shuffles):
        c = shuffled_real("XBTZAR", seed)
        bte, _ = best_test_sharpe(c)
        shuf_max.append(bte)
        shuf_mean.append(mean_sharpe(c))

    rw_max, rw_mean = [], []
    for seed in range(n_shuffles):
        c = synth_random_walk(seed=seed)
        rw_max.append(best_test_sharpe(c)[0])
        rw_mean.append(mean_sharpe(c))

    real_c = R.load_pair("XBTZAR")
    real_max, real_mean = best_test_sharpe(real_c), mean_sharpe(real_c)

    n = len(shuf_max)
    a_mean, a_sd = statistics.mean(shuf_max), statistics.stdev(shuf_max)
    print(f"  {'control':22s} {'best-of-N':>11s} {'sd':>6s} {'max':>7s} "
          f"{'mean-across-specs':>19s}")
    print(f"  {'A: shuffled XBTZAR':22s} {a_mean:+11.2f} {a_sd:6.2f} "
          f"{max(shuf_max):+7.2f} {statistics.mean(shuf_mean):+19.3f}")
    print(f"  {'B: driftless walk':22s} {statistics.mean(rw_max):+11.2f} "
          f"{statistics.stdev(rw_max):6.2f} {max(rw_max):+7.2f} "
          f"{statistics.mean(rw_mean):+19.3f}")
    print(f"  {'REAL XBTZAR':22s} {real_max[0]:+11.2f} {'':6s} {'':7s} "
          f"{real_mean:+19.3f}")

    z = (real_max[0] - a_mean) / a_sd if a_sd else 0.0
    zmean = ((real_mean - statistics.mean(shuf_mean))
             / (statistics.stdev(shuf_mean) or 1))
    print()
    print(f"  best-of-{len(PROBE_GRID)}: real sits {z:+.2f} sd vs control A")
    print(f"  mean-across-specs: real sits {zmean:+.2f} sd vs control A")
    print()
    if real_max[0] <= max(shuf_max):
        print("  -> real data scores BELOW what shuffling produces. The")
        print("     apparent edge is entirely riding drift; timing adds nothing.")
    else:
        print("  -> real EXCEEDS the shuffled control; investigate.")
    if statistics.mean(rw_mean) > 0.25:
        print("  *** control B FAILED: a trend follower beat zero drift.")
        print("  *** either the engine credits free money or costs are wrong.")


def sensitivity() -> None:
    print()
    print("=" * 78)
    print("3. OVERFITTING GAP -- oracle (peeks at test) vs honest (train only)")
    print("=" * 78)
    print("  'oracle' picks the best spec AFTER seeing the test segment.")
    print("  'honest' picks it on train, as you would have to in real life.")
    print("  The gap between them IS the overfitting, measured directly.\n")
    grid = R.build_grid()
    warm = R.warmup_for(grid)
    pairs = R.available_pairs()
    orc = hst = tot = 0
    orc_sh, hst_sh = [], []
    for p in pairs:
        try:
            c = R.load_pair(p)
        except SystemExit:
            continue
        n = len(c)
        if n < warm + 400:
            continue
        split = int(n * 0.6)
        bpy = R.bars_per_year(c)
        bh = R.metrics(R.simulate(c, R.w_buy_hold(c), warm)[split:], bpy).sharpe
        rows = []
        for s in grid:
            eq = R.simulate(c, R.weights_for(c, s), warm)
            rows.append((R.metrics(eq[warm:split], bpy),
                         R.metrics(eq[split:], bpy)))
        honest = rows[max(range(len(rows)), key=lambda i: rows[i][0].sharpe)][1]
        oracle = max(r[1].sharpe for r in rows)
        tot += 1
        orc += oracle > bh
        hst += honest.sharpe > bh
        orc_sh.append(oracle - bh)
        hst_sh.append(honest.sharpe - bh)
    print(f"  {'spec chosen by':22s} {'beats B&H':>10s} {'of':>4s} "
          f"{'mean excess Sharpe':>20s}")
    print(f"  {'oracle (peeks test)':22s} {orc:10d} {tot:4d} "
          f"{statistics.mean(orc_sh):+20.2f}")
    print(f"  {'honest (train only)':22s} {hst:10d} {tot:4d} "
          f"{statistics.mean(hst_sh):+20.2f}")
    print()
    print(f"  gap from removing the peek: {statistics.mean(orc_sh) - statistics.mean(hst_sh):+.2f} Sharpe")
    print(f"  The oracle row is the number a naive backtester would report.")
    print(f"  The honest row is what you could actually have gotten.")


def independent_metrics_check() -> None:
    print()
    print("=" * 78)
    print("4. INDEPENDENT RECOMPUTE -- are the metrics themselves right?")
    print("=" * 78)
    c = R.load_pair("XBTZAR")
    warm = R.warmup_for(R.build_grid())
    eq = R.simulate(c, R.w_buy_hold(c), warm)
    m = R.metrics(eq, R.bars_per_year(c))

    # Sharpe from scratch: population-style stdev, written independently
    rets = []
    for i in range(1, len(eq)):
        rets.append(eq[i] / eq[i - 1] - 1)
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    sharpe_manual = mu / sd * math.sqrt(365.0)

    # maxDD from scratch, scanning forwards
    hi, worst = eq[0], 0.0
    for v in eq:
        if v > hi:
            hi = v
        dd = (v - hi) / hi
        if dd < worst:
            worst = dd

    print(f"  Sharpe  engine {m.sharpe:+.6f}   manual {sharpe_manual:+.6f}   "
          f"diff {abs(m.sharpe-sharpe_manual):.2e}")
    print(f"  maxDD   engine {m.maxdd:.6f}%   manual {-worst*100:.6f}%   "
          f"diff {abs(m.maxdd-(-worst*100)):.2e}")
    assert abs(m.sharpe - sharpe_manual) < 1e-9, "Sharpe mismatch"
    assert abs(m.maxdd - (-worst * 100)) < 1e-9, "maxDD mismatch"
    print("  -> metrics reproduce from an independent implementation")


def main() -> None:
    positive_control()
    placebo()
    sensitivity()
    independent_metrics_check()


if __name__ == "__main__":
    main()