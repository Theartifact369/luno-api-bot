"""Does the 06:00 UTC (08:00 SAST) hourly effect survive out of sample?

candidates.py found hour 06:00 UTC has t=+4.26 across 118,293 hourly returns
(+0.0956% mean). With 31 buckets tested, Bonferroni needs |t| ~ 2.7 for a
family-wise 5% threshold, so this one survives.

The proposed mechanism is real and specific to this exchange: 06:00 UTC is
08:00 in South Africa, the local cash market opens, and RSA retail/instilled
flow hits Luno at exactly that hour. Crypto trades 24/7 but the PEOPLE
trading it in ZAR have a working day. If that is the mechanism, it should be
an ACCRETION (drift at one hour), not a pattern that only exists in-sample.

This file is the disproof-first test:
  A. Split chronologically. Choose the hour on TRAIN, trade it on TEST only.
  B. Compare against EVERY other hour, not just a random one -- picking the
     best of 24 by hand after seeing test data is the trap.
  C. Subtract the unconditional hourly mean, because "hour 6 was good in
     sample" might just mean the sample had a rising market.

Run: .venv/bin/python calendar_oos.py
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import research as R

SAST_OFFSET = 2  # South Africa Standard Time


def hourly_frame(pair: str = "XBTZAR"):
    """(timestamps, closes, hourly returns) from cached hourly bars."""
    c = R.load_pair(pair, 3600)
    closes = [float(x.close) for x in c]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    stamps = [c[i].timestamp for i in range(1, len(c))]
    return stamps, closes, rets


def bucket(stamps, rets, keyfn):
    out = defaultdict(list)
    for ts, r in zip(stamps, rets):
        out[keyfn(datetime.fromtimestamp(ts / 1000, tz=timezone.utc))].append(r)
    return out


def main() -> None:
    stamps, closes, rets = hourly_frame()
    n = len(rets)
    print(f"XBTZAR hourly: {n:,} returns, "
          f"{R.d(stamps[0])} -> {R.d(stamps[-1])}")

    by_hour = bucket(stamps, rets, lambda dt: dt.hour)
    overall = statistics.mean(rets)
    print(f"unconditional mean hourly return {overall*100:+.4f}%")

    # ---------------- A. chronological split, hour chosen on TRAIN ----------
    split = int(n * 0.6)
    tr_st = stamps[:split]
    tr_rets = rets[:split]
    te_st = stamps[split:]
    te_rets = rets[split:]

    tr_by_hour = bucket(tr_st, tr_rets, lambda dt: dt.hour)
    te_by_hour = bucket(te_st, te_rets, lambda dt: dt.hour)

    scored = sorted(
        ((h, statistics.mean(v)) for h, v in tr_by_hour.items() if len(v) > 50),
        key=lambda t: -t[1])
    best_h, best_mu = scored[0]
    print(f"\nA. train window {R.d(tr_st[0])} -> {R.d(tr_st[-1])}")
    print(f"   best hour on TRAIN: {best_h:02d}:00 UTC "
          f"({(best_h + SAST_OFFSET) % 24:02d}:00 SAST), mean {best_mu*100:+.4f}%")

    te_mu = statistics.mean(te_by_hour[best_h])
    te_all = statistics.mean(te_rets)
    sd_te = statistics.stdev(te_by_hour[best_h])
    t_te = (te_mu - te_all) / (sd_te / math.sqrt(len(te_by_hour[best_h])))
    print(f"   its TEST mean {te_mu*100:+.4f}% vs all-hours TEST mean "
          f"{te_all*100:+.4f}%")
    print(f"   test t-stat {t_te:+.2f} on n={len(te_by_hour[best_h]):,}")

    # ---------------- B. every hour, train vs test side by side ----------
    print("\nB. every hour: train mean, test mean, and the excess")
    print(f"   {'hr':>3s} {'SAST':>5s} {'train':>10s} {'test':>10s} "
          f"{'test t':>8s} {'n_test':>8s}")
    rows = []
    for h in range(24):
        if h not in tr_by_hour or h not in te_by_hour:
            continue
        tm = statistics.mean(tr_by_hour[h])
        em = statistics.mean(te_by_hour[h])
        v = te_by_hour[h]
        t = (em - te_all) / (statistics.stdev(v) / math.sqrt(len(v)))
        rows.append((h, tm, em, t, len(v)))
        print(f"   {h:3d} {(h+SAST_OFFSET)%24:5d} {tm*100:+9.4f}% {em*100:+9.4f}% "
              f"{t:+8.2f} {len(v):8d}")

    # rank correlation between train and test: the direct falsification test.
    # A real hour effect shows up in BOTH windows -> high correlation.
    # A fluke found by scanning 24 buckets -> near zero.
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    corr = cov / (math.sqrt(sum((a - mx) ** 2 for a in xs))
                  * math.sqrt(sum((b - my) ** 2 for b in ys)))
    print(f"\n   train/test correlation of hourly means: r = {corr:+.3f}")
    print(f"   (24 buckets -> |r| under ~0.4 is what noise alone produces)")

    # ---------------- C. tradability: is the edge larger than the cost? ---
    print("\nC. is the effect even tradable?")
    gross = te_mu
    net = gross - (0.006 + 0.0005)   # Luno taker + half-spread, one way
    per_hour_equity = (1 + net)
    # 8766 hours/yr, but you can only trade the ~1 bar per hour you select
    print(f"   test mean gross {gross*100:+.4f}%/hour")
    print(f"   after Luno 0.60% taker + 0.05% half-spread: "
          f"{net*100:+.4f}%/hour")
    print(f"   -> {'TRADABLE' if net > 0 else 'NOT tradable: cost exceeds the effect'}")
    print()
    print("   Note: trading one hour per day means 8766 round trips/yr, each")
    print("   paying ~0.65% twice. The cost is the dominant term, not alpha.")


if __name__ == "__main__":
    main()