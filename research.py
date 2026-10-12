"""Honest edge research: does any rule actually beat buy & hold OUT OF SAMPLE?

Design rules this module enforces, each because the naive version lies:
  * Causal. weights[i] is decided from data < i and executed at bar i's OPEN.
    No signal ever sees the bar it trades.
  * No vault. The PaperTrader vault folds passive BTC holding into "alpha";
    here equity is strictly the traded account.
  * Fixed periodicity. Sharpe/Sortino/Calmar come from DAILY equity points for
    the strategy and every baseline, on the same window.
  * Chronological split. Parameters are chosen on train, reported on test.
    Combos tried are counted and printed. A winner after 200 tries is noise.
  * Costs are paid. Half-spread crossing + 0.1% fee on every fill, for the
    strategy AND for buy & hold.

Stdlib only. Float math (research screening); the Decimal path in
paper_trader.py remains the money-moving code.

Run:  .venv/bin/python research.py search
      .venv/bin/python research.py pairs
      .venv/bin/python research.py walk
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from operator import attrgetter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from backtest import _cache_path
from client import Candle

CACHE_DIR = Path("candles")
FEE = 0.001          # Luno taker, charged on notional traded
HALF_SPREAD_BPS = 5.0
DAYS_PER_YEAR = 365.0

# Parameter grid. Deliberately small: fewer knobs, longer periods.
CROSS = [(10, 100), (20, 100), (30, 150), (50, 200), (20, 200), (30, 200),
         (50, 150), (100, 200), (10, 50), (30, 100)]
DONCH = [(20, 10), (40, 20), (55, 20), (100, 50)]
TARGET_VOLS = [0.3, 0.5]


# ---------------------------------------------------------------- data ----

def load_pair(pair: str, duration: int = 86400) -> list[Candle]:
    cache = _cache_path(pair, duration)
    if not cache.exists():
        raise SystemExit(f"no cache for {pair} {duration}s -- fetch it first")
    return [Candle(**{**c, "open": Decimal(c["open"]), "high": Decimal(c["high"]),
                      "low": Decimal(c["low"]), "close": Decimal(c["close"]),
                      "volume": Decimal(c["volume"])})
            for c in json.loads(cache.read_text())]


def closes_of(c: list[Candle]) -> list[float]:
    return [float(x.close) for x in c]


def opens_of(c: list[Candle]) -> list[float]:
    return [float(x.open) for x in c]


def highs_of(c: list[Candle]) -> list[float]:
    return [float(x.high) for x in c]


def lows_of(c: list[Candle]) -> list[float]:
    return [float(x.low) for x in c]


def bars_per_year(candles: list[Candle]) -> float:
    if len(candles) < 2:
        return DAYS_PER_YEAR
    step_days = (candles[1].timestamp - candles[0].timestamp) / 86_400_000
    return DAYS_PER_YEAR / step_days


def d(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def available_pairs() -> list[str]:
    return sorted(p.stem.rsplit("_", 1)[0] for p in CACHE_DIR.glob("*_86400.json"))


# ----------------------------------------------------------- indicators ----

def sma_series(closes: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    run = 0.0
    for i, c in enumerate(closes):
        run += c
        if i >= n:
            run -= closes[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def ema_series(closes: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if len(closes) < n:
        return out
    ema = sum(closes[:n]) / n
    out[n - 1] = ema
    k = 2.0 / (n + 1)
    for i in range(n, len(closes)):
        ema = (closes[i] - ema) * k + ema
        out[i] = ema
    return out


def realized_vol(closes: list[float], upto: int, n: int) -> float | None:
    """Stdev of log returns over the n bars ENDING at `upto`. None if short."""
    if upto - n < 0:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(upto - n + 1, upto + 1)
            if closes[i - 1] > 0 and closes[i] > 0]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    return math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))


# ------------------------------------------------------------ simulator ----

def simulate(candles: list[Candle], weights: list[float], start_i: int,
             fee: float = FEE, half_spread_bps: float = HALF_SPREAD_BPS) -> list[float]:
    """Equity normalised to 1.0 at start_i, one point per bar, mark-to-close.

    weights[i] executes at bar i's OPEN, paying spread on the side traded plus
    fee on notional. Bars before start_i sit in cash so no signal is ever fed
    a partially-formed window.
    """
    sp = half_spread_bps / 10_000
    opens, closes = opens_of(candles), closes_of(candles)
    cash, units, cur_w = 1.0, 0.0, 0.0
    equity = [1.0] * start_i
    for i in range(start_i, len(candles)):
        w = weights[i]
        if abs(w - cur_w) > 1e-12:
            opx = opens[i]
            pv = cash + units * opx
            delta = (w * pv / opx) - units
            px = opx * (1 + sp) if delta > 0 else opx * (1 - sp)
            cash -= delta * px + abs(delta) * px * fee
            units += delta
            cur_w = w
        equity.append(cash + units * closes[i])
    return equity


# --------------------------------------------------------------- metrics ----

@dataclass
class Metrics:
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    maxdd: float = 0.0
    calmar: float = 0.0
    time_in_market: float = 0.0
    total_ret: float = 0.0
    years: float = 0.0
    ci95: float = 0.0
    n_bars: int = 0

    def row(self) -> str:
        return (f"CAGR {self.cagr:+7.1f}%  Sharpe {self.sharpe:+.2f}±{self.ci95:.2f}  "
                f"Sortino {self.sortino:+.2f}  maxDD {self.maxdd:5.1f}%  "
                f"Calmar {self.calmar:+.2f}  TiM {self.time_in_market*100:4.0f}%")

    def short(self) -> str:
        return (f"Sharpe {self.sharpe:+.2f}±{self.ci95:.2f}  maxDD {self.maxdd:5.1f}%  "
                f"Calmar {self.calmar:+.2f}  CAGR {self.cagr:+7.1f}%")


def metrics(equity: list[float], bpy: float, weights: list[float] | None = None) -> Metrics:
    m = Metrics(n_bars=len(equity))
    if len(equity) < 10 or equity[0] <= 0 or bpy <= 0:
        return m
    years = len(equity) / bpy
    m.years, m.n_bars = years, len(equity)
    m.total_ret = equity[-1] / equity[0] - 1
    if years > 0 and equity[-1] > 0:
        m.cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100

    rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    mu = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
    if sd > 0:
        m.sharpe = mu / sd * math.sqrt(bpy)
        # Lo (2002): SE of the annualised Sharpe ~ sqrt((1 + S^2/2) / T_years).
        m.ci95 = 1.96 * math.sqrt((1 + (m.sharpe ** 2) / 2) / years)
    dn = [r for r in rets if r < 0]
    if dn:
        dsd = math.sqrt(sum(r * r for r in dn) / len(rets))
        if dsd > 0:
            m.sortino = mu / dsd * math.sqrt(bpy)

    peak, worst = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = min(worst, e / peak - 1)
    m.maxdd = -worst * 100
    m.calmar = m.cagr / m.maxdd if m.maxdd > 0 else 0.0
    if weights is not None:
        # A length mismatch means the caller mis-sliced, and silently dropping
        # time-in-market to 0% hid a real bug for a whole run. Fail loudly.
        if len(weights) != len(equity):
            raise ValueError(
                f"weights ({len(weights)}) must align 1:1 with equity "
                f"({len(equity)}) for time-in-market")
        m.time_in_market = sum(1 for w in weights if w > 1e-12) / len(weights)
    return m


# ------------------------------------------------------------- signals ----
# Every signal returns weights where weights[i] is decided from bars < i.

def w_buy_hold(c: list[Candle]) -> list[float]:
    return [1.0] * len(c)


def w_cash(c: list[Candle]) -> list[float]:
    return [0.0] * len(c)


def w_dca(c: list[Candle], every: int = 30) -> list[float]:
    """Equal notional every `every` bars; weight ramps toward full."""
    n = len(c)
    return [min(1.0, (i // every + 1) / (n / every)) for i in range(n)]


def w_sma(c: list[Candle], fast: int, slow: int) -> list[float]:
    cl = closes_of(c)
    f, s = sma_series(cl, fast), sma_series(cl, slow)
    return [1.0 if (f[i - 1] is not None and s[i - 1] is not None and f[i - 1] > s[i - 1])
            else 0.0 for i in range(len(cl))]


def w_ema(c: list[Candle], fast: int, slow: int) -> list[float]:
    cl = closes_of(c)
    f, s = ema_series(cl, fast), ema_series(cl, slow)
    return [1.0 if (f[i - 1] is not None and s[i - 1] is not None and f[i - 1] > s[i - 1])
            else 0.0 for i in range(len(cl))]


def w_donchian(c: list[Candle], entry: int, exit_: int) -> list[float]:
    hi_, lo_ = highs_of(c), lows_of(c)
    out, in_pos = [], False
    for i in range(len(c)):
        if i < max(entry, exit_) + 1:
            out.append(0.0)
            continue
        if in_pos:
            if lo_[i - 1] < min(lo_[i - exit_:i]):
                in_pos = False
        elif hi_[i - 1] >= max(hi_[i - entry:i]):
            in_pos = True
        out.append(1.0 if in_pos else 0.0)
    return out


def w_voltarget(c: list[Candle], fast: int, slow: int, target_vol: float,
                lookback: int = 30) -> list[float]:
    """Crossover picks direction, inverse-vol picks size.

    The one hypothesis where a crossover could plausibly beat holding is on
    RISK: de-risk when vol spikes, which buy & hold cannot do.
    """
    cl = closes_of(c)
    f, s = sma_series(cl, fast), sma_series(cl, slow)
    ann = math.sqrt(bars_per_year(c))
    out = []
    for i in range(len(cl)):
        if i == 0:
            out.append(0.0)
            continue
        if not (f[i - 1] is not None and s[i - 1] is not None and f[i - 1] > s[i - 1]):
            out.append(0.0)
            continue
        vol = realized_vol(cl, i - 1, lookback)
        if vol is None or vol <= 0:
            out.append(0.0)
        else:
            out.append(min(1.0, target_vol / (vol * ann)))
    return out


def weights_for(candles: list[Candle], spec: dict) -> list[float]:
    k = spec["kind"]
    if k == "sma":
        return w_sma(candles, spec["fast"], spec["slow"])
    if k == "ema":
        return w_ema(candles, spec["fast"], spec["slow"])
    if k == "donchian":
        return w_donchian(candles, spec["entry"], spec["exit"])
    if k == "voltarget":
        return w_voltarget(candles, spec["fast"], spec["slow"], spec["target_vol"])
    raise ValueError(k)


def build_grid() -> list[dict]:
    g: list[dict] = []
    for f, s in CROSS:
        g.append({"kind": "sma", "fast": f, "slow": s})
        g.append({"kind": "ema", "fast": f, "slow": s})
        for tv in TARGET_VOLS:
            g.append({"kind": "voltarget", "fast": f, "slow": s, "target_vol": tv})
    for e, x in DONCH:
        g.append({"kind": "donchian", "entry": e, "exit": x})
    return g


def warmup_for(grid: list[dict]) -> int:
    return max(max(s.get("slow", 0), s.get("entry", 0) + s.get("exit", 0))
               for s in grid) + 2


def label(spec: dict) -> str:
    k = spec["kind"]
    if k == "donchian":
        return f"donchian {spec['entry']}/{spec['exit']}"
    if k == "voltarget":
        return f"sma {spec['fast']}/{spec['slow']} voltgt {spec['target_vol']:.0%}"
    return f"{k} {spec['fast']}/{spec['slow']}"


# ------------------------------------------------------------- pipeline ----

def evaluate_pair(pair: str, grid: list[dict], train_frac: float) -> dict:
    """Every spec on one pair, sliced on ONE chronological boundary. Signals are
    computed on the full series (causal by construction) and only the equity
    curve is sliced, so the test segment inherits the live position."""
    candles = load_pair(pair)
    n = len(candles)
    warm = warmup_for(grid)
    if n < warm + 400:
        return {"pair": pair, "skipped": f"only {n} bars"}
    bpy = bars_per_year(candles)
    split = int(n * train_frac)
    out = {"pair": pair, "n_bars": n, "warmup": warm,
           "train": [d(candles[0].timestamp), d(candles[split].timestamp)],
           "test": [d(candles[split].timestamp), d(candles[-1].timestamp)],
           "specs": {}, "baselines": {}}

    for name, w in (("buy&hold", w_buy_hold(candles)), ("cash", w_cash(candles)),
                    ("dca30", w_dca(candles, 30))):
        eq = simulate(candles, w, warm)
        out["baselines"][name] = {
            "train": metrics(eq[warm:split], bpy, w[warm:split]),
            "test": metrics(eq[split:], bpy, w[split:]),
        }

    for spec in grid:
        w = weights_for(candles, spec)
        eq = simulate(candles, w, warm)
        out["specs"][label(spec)] = {
            "spec": spec,
            "train": metrics(eq[warm:split], bpy, w[warm:split]),
            "test": metrics(eq[split:], bpy, w[split:]),
        }
    return out


def cmd_search(args) -> None:
    grid = build_grid()
    r = evaluate_pair(args.pair, grid, args.train_frac)
    if "skipped" in r:
        print(f"SKIPPED: {r['skipped']}")
        return
    print(f"{r['pair']}: {r['n_bars']} bars  train {r['train'][0]} -> {r['train'][1]}  "
          f"|  test {r['test'][0]} -> {r['test'][1]}\n")
    for seg in ("train", "test"):
        print(f"--- {seg.upper()} baselines (n="
              f"{r['baselines']['buy&hold'][seg].n_bars} bars) ---")
        for name, v in r["baselines"].items():
            print(f"  {name:9s} {v[seg].row()}")
        print()

    key = attrgetter(args.objective)
    print(f"--- top 15 specs by {args.objective} on TRAIN ({len(grid)} combos tried) ---")
    for name, v in sorted(r["specs"].items(), key=lambda kv: -key(kv[1]["train"]))[:15]:
        print(f"  {name:28s} train {key(v['train']):+.3f}   "
              f"TEST  {v['test'].short()}")

    bh = r["baselines"]["buy&hold"]["test"].sharpe
    ts = [v["test"].sharpe for v in r["specs"].values()]
    beat = sum(1 for v in r["specs"].values() if v["test"].sharpe > bh)
    print(f"\n  {beat}/{len(grid)} specs beat buy&hold on TEST "
          f"(B&H test Sharpe {bh:+.2f})")
    print(f"  test Sharpe spread: min {min(ts):+.2f}  median {statistics.median(ts):+.2f}"
          f"  max {max(ts):+.2f}   <- the max is selection noise, not an edge")


def cmd_pairs(args) -> None:
    grid = build_grid()
    pairs = available_pairs()
    print(f"grid {len(grid)} specs x {len(pairs)} pairs = {len(grid)*len(pairs)} backtests")
    print(f"objective chosen on TRAIN ({args.train_frac:.0%}), reported on TEST\n")

    diffs: dict[str, list[float]] = {}
    live = []
    for p in pairs:
        r = evaluate_pair(p, grid, args.train_frac)
        if "skipped" in r:
            print(f"  {p:9s} SKIPPED ({r['skipped']})")
            continue
        live.append(p)
        bht = r["baselines"]["buy&hold"]["test"]
        best = max(r["specs"].items(), key=lambda kv: kv[1]["train"].sharpe)  # noqa: E501
        for name, v in r["specs"].items():
            diffs.setdefault(name, []).append(v["test"].sharpe - bht.sharpe)
        print(f"  {p:9s} n={r['n_bars']:5d}  B&H Sharpe {bht.sharpe:+.2f} "
              f"maxDD {bht.maxdd:4.1f}%  |  train-pick {best[0]:26s} "
              f"TEST {best[1]['test'].sharpe:+.2f} maxDD {best[1]['test'].maxdd:4.1f}%")

    print(f"\n--- every spec, TEST Sharpe minus buy&hold, over {len(live)} pairs ---")
    print("(a spec that is genuinely good should beat holding on MOST pairs)")
    for name, ds in sorted(diffs.items(), key=lambda kv: -statistics.mean(kv[1])):
        print(f"  {name:28s} mean {statistics.mean(ds):+.3f}  "
              f"median {statistics.median(ds):+.3f}  "
              f"beats B&H {sum(1 for x in ds if x > 0):2d}/{len(ds)}")


def cmd_walk(args) -> None:
    """Walk-forward: re-optimise on trailing bars, trade the NEXT block only.

    No bar is ever used both to choose parameters and to score them.
    """
    grid = build_grid()
    key = attrgetter(args.objective)
    print(f"walk-forward: train {args.train_bars} bars -> test {args.test_bars} bars, "
          f"objective={args.objective}, {len(grid)} combos\n")

    out = {}
    for p in available_pairs():
        candles = load_pair(p)
        n, warm = len(candles), warmup_for(grid)
        bpy = bars_per_year(candles)
        if n < warm + args.train_bars + args.test_bars + 10:
            continue
        eqs = {label(s): simulate(candles, weights_for(candles, s), warm) for s in grid}
        bh = simulate(candles, w_buy_hold(candles), warm)

        chain, bh_chain, chosen = [1.0], [1.0], []
        i = warm + args.train_bars
        while i + args.test_bars <= n:
            best = max(grid, key=lambda s: key(metrics(eqs[label(s)][warm:i], bpy)))
            chosen.append(label(best))
            seg = eqs[label(best)][i:i + args.test_bars]
            bseg = bh[i:i + args.test_bars]
            # Capture base BEFORE extending: extend() consumes a generator
            # lazily while `chain` is still growing, so reading chain[-1]
            # inside the generator re-reads the element just appended and
            # compounds. That produced CAGR of 1e27% and maxDD of 100% on a
            # curve that never fell.
            base, bbase = chain[-1], bh_chain[-1]
            chain.extend([base * (e / seg[0]) for e in seg[1:]])
            bh_chain.extend([bbase * (e / bseg[0]) for e in bseg[1:]])
            i += args.test_bars
        if len(chain) < 30:
            continue
        out[p] = (metrics(chain, bpy), metrics(bh_chain, bpy), chosen)
        print(f"  {p:9s} blocks {len(chosen):2d}  distinct specs chosen "
              f"{len(set(chosen)):2d}  WALK-FWD OOS {metrics(chain, bpy).short()}")
        print(f"  {'':9s} {'':28s} buy&hold  OOS {metrics(bh_chain, bpy).short()}")

    if not out:
        print("no pair had enough bars")
        return
    s_sh = [v[0].sharpe for v in out.values()]
    h_sh = [v[1].sharpe for v in out.values()]
    s_dd = [v[0].maxdd for v in out.values()]
    h_dd = [v[1].maxdd for v in out.values()]
    print(f"\n--- walk-forward pooled, {len(out)} pairs ---")
    print(f"  strategy Sharpe mean {statistics.mean(s_sh):+.3f}  "
          f"median {statistics.median(s_sh):+.3f}  "
          f"beats B&H {sum(1 for a, b in zip(s_sh, h_sh) if a > b)}/{len(s_sh)}")
    print(f"  buy&hold  Sharpe mean {statistics.mean(h_sh):+.3f}")
    print(f"  strategy maxDD   mean {statistics.mean(s_dd):.1f}%  "
          f"lower than B&H {sum(1 for a, b in zip(s_dd, h_dd) if a < b)}/{len(s_dd)} "
          f"(B&H mean {statistics.mean(h_dd):.1f}%)")
    print(f"  params unstable: "
          f"{sum(1 for v in out.values() if len(set(v[2])) > 1)}/{len(out)} pairs "
          f"picked a different spec in different blocks")


def main() -> None:
    ap = argparse.ArgumentParser(description="Honest out-of-sample edge research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--pair", default="XBTZAR")
    s.add_argument("--train-frac", type=float, default=0.6)
    s.add_argument("--objective", default="sharpe",
                   choices=["sharpe", "calmar", "sortino", "cagr"])
    s.set_defaults(func=cmd_search)
    s = sub.add_parser("pairs")
    s.add_argument("--train-frac", type=float, default=0.6)
    s.set_defaults(func=cmd_pairs)
    s = sub.add_parser("walk")
    s.add_argument("--train-bars", type=int, default=750)
    s.add_argument("--test-bars", type=int, default=250)
    s.add_argument("--objective", default="sharpe",
                   choices=["sharpe", "calmar", "sortino", "cagr"])
    s.set_defaults(func=cmd_walk)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()