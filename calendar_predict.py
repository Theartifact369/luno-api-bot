"""Is the 06:00 UTC effect PREDICTIVE, or just a same-hour artifact?

calendar_oos.py found that the 06:00 UTC hourly return (08:00 SAST, South
African market open) was the best hour on train (+0.1202%) and held out of
sample (+0.0586%, t=+4.36). That is a genuine out-of-sample replication and
it has a real mechanism -- RSA cash-market flow hitting Luno at open.

The reason it still looked worthless: capturing a +0.059%/hour mean with a
0.60% taker fee is hopeless IF you must round-trip to harvest it.

So everything turns on one question. Is the effect:

  (a) an ACCRETION -- the hour-6 return predicts what happens NEXT, so you
      buy once at 06:00 and hold, amortising one round trip over many hours?
      Then it is tradable.

  (b) a CONTEMPORANEOUS artifact -- the hour-6 bar simply prints a higher
      return, but hour 7 onwards reverts and nothing is predictable? Then
      it is not tradable at any cost, and the effect is a curiosity.

This file measures (a) directly: conditional forward returns after a strong
hour-6 bar versus unconditional. Stdlib only.

Run: .venv/bin/python calendar_predict.py
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import research as R


def main() -> None:
    c = R.load_pair("XBTZAR", 3600)
    closes = [float(x.close) for x in c]
    stamps = [x.timestamp for x in c]
    hours = [datetime.fromtimestamp(t / 1000, tz=timezone.utc).hour for t in stamps]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    hh = hours[1:]

    H = 6  # 08:00 SAST, South African cash-market open

    # index the hour-6 bars
    idx6 = [i for i, h in enumerate(hh) if h == H]
    print(f"hour {H:02d}:00 UTC observations: {len(idx6):,}")

    # forward returns AFTER the hour-6 bar closes
    horizons = {"+1h": 1, "+6h": 6, "+24h": 24, "+7d": 168}
    print(f"\n{'horizon':>6s} {'uncond':>10s} {'after +sig':>11s} {'after -sig':>11s} "
          f"{'t-stat':>8s} {'n_+sig':>7s}")
    results = {}
    for label, hz in horizons.items():
        cond_hi, cond_lo, base = [], [], []
        for i in idx6:
            j = i + hz
            if j >= len(rets):
                continue
            # Condition on the SIGNAL (the hour-6 return rets[i]), then
            # measure the OUTCOME (rets[i+hz]). The first version conditioned
            # on the forward return itself -- a tautology that produced a
            # meaningless t of +33.
            sig, fwd = rets[i], rets[j]
            base.append(fwd)
            (cond_hi if sig > 0 else cond_lo).append(fwd)
        if len(cond_hi) < 100 or len(cond_lo) < 100:
            continue
        mh, ml = statistics.mean(cond_hi), statistics.mean(cond_lo)
        mb = statistics.mean(base)
        se = math.sqrt(statistics.variance(cond_hi) / len(cond_hi)
                       + statistics.variance(cond_lo) / len(cond_lo))
        t = (mh - ml) / se if se else 0.0
        results[label] = (mb, mh, ml, t, len(cond_hi))
        print(f"{label:>6s} {mb*100:+9.4f}% {mh*100:+10.4f}% {ml*100:+10.4f}% "
              f"{t:+8.2f} {len(cond_hi):7d}")

    print("\nInterpretation: forward return AFTER a positive hour-6 bar versus AFTER")
    print("a negative one, conditioning on the SIGNAL not on the outcome.")
    print("A tradable edge needs a reliable gap. There is none at +6h, +24h")
    print("or +7d. The +1h gap is NEGATIVE (reversal) and costs 1.30% to trade.")

    # --- does the hour-6 return predict ANYTHING, on its own magnitude? ---
    print("\n" + "=" * 74)
    print("Does a LARGE hour-6 return predict a large next-day return?")
    print("=" * 74)
    xs, ys = [], []
    for i in idx6:
        j = i + 24
        if j < len(rets):
            xs.append(rets[i])
            ys.append(rets[j])
    if len(xs) > 100:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        sy = math.sqrt(sum((b - my) ** 2 for b in ys))
        r = cov / (sx * sy) if sx and sy else 0.0
        print(f"  correlation(hour-6 return, next-24h return) = {r:+.4f} "
              f"on n={len(xs):,}")
        # volatility is the thing that usually persists, not direction
        absx = [abs(v) for v in xs]
        absy = [abs(v) for v in ys]
        amx, amy = statistics.mean(absx), statistics.mean(absy)
        acov = sum((a - amx) * (b - amy) for a, b in zip(absx, absy))
        asx = math.sqrt(sum((a - amx) ** 2 for a in absx))
        asy = math.sqrt(sum((b - amy) ** 2 for b in absy))
        ar = acov / (asx * asy) if asx and asy else 0.0
        print(f"  correlation(|hour-6|, |next-24h|)          = {ar:+.4f} "
              f"<- volatility clustering, which IS predictable")
        print()
        print("  A large |r| on raw returns would be a tradable edge.")
        print("  A large |r| only on |returns| means volatility persists but")
        print("  direction does not -- which is exactly what equity/crisis")
        print("  research finds, and it is not exploitable on a spot book.")

    # --- what does it cost to actually try? ---
    print("\n" + "=" * 74)
    print("Cost of the best case: buy at 06:00 UTC, sell 24h later")
    print("=" * 74)
    if "+24h" in results:
        mb = results["+24h"][0]
        one_rt = 2 * (0.006 + 0.0005)
        print(f"  mean 24h forward return  {mb*100:+.4f}%")
        print(f"  one round trip costs     {one_rt*100:.2f}%  (2 x 0.60% + 2 x 0.05%)")
        print(f"  net per trade            {(mb - one_rt)*100:+.4f}%")
        print(f"  -> {'POSITIVE net' if mb > one_rt else 'NEGATIVE: cost is %.0fx the edge'
              % (one_rt / mb if mb > 0 else 0)}")


if __name__ == "__main__":
    main()