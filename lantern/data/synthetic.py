"""A synthetic cross-asset market for tests, offline demos and CI.

Latent macro shocks (risk appetite, rates, dollar) drive ~20 market drivers with
stochastic volatility and two stress regimes; every symbol in the default factor
universe and the demo portfolio is a linear combination of those drivers plus noise.
One stock ("NVDA") and Bitcoin carry an explosive run-up over the last 18 months, so
the bubble-burst overlay has something to find. The output has exactly the shape of
live data (MarketData), so every pipeline runs unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import MarketData

DRIVERS = {  # name: (daily vol, loadings on [risk-on, rates-up, dollar-up])
    "EQ": (0.0100, [0.95, 0.15, -0.10]), "DM": (0.0040, [0.30, 0.0, -0.30]),
    "EM": (0.0060, [0.50, 0.0, -0.40]), "LVL": (0.0006, [0.25, 0.95, 0.10]),
    "SLOPE": (0.0003, [0.10, 0.40, 0.0]), "CREDIT": (0.00025, [-0.75, -0.10, 0.10]),
    "USD": (0.0045, [-0.25, 0.20, 0.90]), "OIL": (0.020, [0.40, 0.10, -0.20]),
    "GOLD": (0.0095, [0.05, -0.30, -0.50]), "METALS": (0.012, [0.45, 0.05, -0.30]),
    "BTC": (0.035, [0.35, -0.05, -0.15]), "SIZE": (0.005, [0.30, 0.0, 0.0]),
    "VALUE": (0.005, [0.10, 0.35, 0.0]), "MOM": (0.006, [-0.20, -0.10, 0.0]),
    "QUAL": (0.003, [-0.30, 0.0, 0.0]), "MINVOL": (0.004, [-0.55, -0.10, 0.0]),
    "TECH": (0.006, [0.25, -0.25, 0.0]), "FIN": (0.006, [0.20, 0.35, 0.0]),
    "ENERGY": (0.008, [0.10, 0.15, 0.0]), "INFL": (0.0015, [0.30, 0.20, -0.10]),
    "JPY": (0.005, [-0.40, -0.30, -0.50]), "GBP": (0.004, [0.20, 0.0, -0.70]),
    "CNY": (0.0015, [0.10, 0.0, -0.50]),
}

STOCKS = {  # symbol: (EQ beta, sector driver, sector beta, idio daily vol)
    "AAPL": (1.10, "TECH", 0.8, 0.012), "MSFT": (1.05, "TECH", 0.7, 0.011),
    "NVDA": (1.60, "TECH", 1.4, 0.022), "AMZN": (1.20, "TECH", 0.5, 0.015),
    "JPM": (1.10, "FIN", 1.0, 0.011), "XOM": (0.80, "ENERGY", 1.0, 0.012),
    "BRK-B": (0.85, "FIN", 0.3, 0.008),
}


def _drivers(n, rng):
    names = list(DRIVERS)
    vol = np.array([DRIVERS[k][0] for k in names])
    L = np.array([DRIVERS[k][1] for k in names])
    own = np.sqrt(np.clip(1 - (L ** 2).sum(axis=1), 0.05, None))
    # stochastic volatility (log-AR(1)) plus two stress regimes
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.985 * h[t - 1] + 0.11 * rng.standard_normal()
    stress = np.ones(n)
    stress[int(0.17 * n):int(0.22 * n)] = 3.0              # a 2008-like episode
    stress[int(0.70 * n):int(0.72 * n)] = 3.5              # a 2020-like crash
    m = np.exp(h - h.var() / 2) * stress
    macro = rng.standard_normal((n, 3))
    macro[:, 0] -= 0.10 * (stress > 1)                     # risk-off drift in stress
    X = (macro @ L.T + rng.standard_normal((n, len(names))) * own) * vol * m[:, None]
    return pd.DataFrame(X, columns=names), m


def synthetic_market(seed: int = 42, start: str = "2005-01-03", end: str = "2026-09-30") -> MarketData:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    D, m = _drivers(n, rng)
    D.index = idx
    D["EQ"] += 0.00035                                     # equity premium (~9%/yr)
    noise = lambda s: rng.standard_normal(n) * s * m      # noqa: E731
    r = {}
    r["SPY"] = D.EQ
    r["IWM"] = D.EQ + D.SIZE
    r["VLUE"], r["IWD"], r["IWF"] = D.EQ + D.VALUE, D.EQ + 0.5 * D.VALUE, D.EQ - 0.5 * D.VALUE
    r["MTUM"], r["QUAL"] = D.EQ + D.MOM, D.EQ + D.QUAL
    r["USMV"], r["SPLV"] = D.EQ + D.MINVOL, D.EQ + 1.1 * D.MINVOL + noise(0.001)
    r["XLK"], r["XLF"] = D.EQ + D.TECH, D.EQ + D.FIN
    r["XLE"] = D.EQ + D.ENERGY + 0.3 * D.OIL
    eur = -D.USD + noise(0.002)
    r["EFA"] = 0.9 * D.EQ + D.DM + 0.6 * eur
    r["EEM"] = 1.0 * D.EQ + D.EM + 0.3 * D.CNY
    r["ACWI"] = 0.62 * r["SPY"] + 0.28 * r["EFA"] + 0.10 * r["EEM"]
    r["VT"] = r["ACWI"] + noise(0.0005)
    # yields (percent) and bond ETFs
    tenors = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}
    start_lv = {"^IRX": 3.0, "^FVX": 3.6, "^TNX": 4.0, "^TYX": 4.4}
    dy = {}
    for s, t in tenors.items():
        slope = (np.log(t) - np.log(5.0)) / np.log(6.0)
        w = 0.6 if t < 1 else 1.0
        dy[s] = 100 * (w * D.LVL + slope * D.SLOPE) + noise(0.01)
    lv = {}
    for s in tenors:
        y = np.empty(n)
        y[0] = start_lv[s]
        for t in range(1, n):                               # mean-reverting, floored
            y[t] = max(y[t - 1] + dy[s].iloc[t] + 0.0005 * (start_lv[s] - y[t - 1]), 0.01)
        lv[s] = y
    dy = {s: np.diff(lv[s], prepend=lv[s][0]) / 100 for s in tenors}   # decimal
    ig = D.CREDIT
    hy = 3.5 * D.CREDIT + noise(0.0003)
    r["IEF"] = -7.2 * (0.5 * dy["^FVX"] + 0.5 * dy["^TNX"]) + 0.00012
    r["IEI"] = -4.3 * (0.8 * dy["^FVX"] + 0.2 * dy["^IRX"]) + 0.0001
    r["TLT"] = -17.0 * (0.4 * dy["^TNX"] + 0.6 * dy["^TYX"]) + 0.00015
    r["TIP"] = r["IEF"] + D.INFL
    r["LQD"] = -8.3 * (0.4 * dy["^FVX"] + 0.6 * dy["^TNX"]) - 8.3 * ig + 0.00016
    r["HYG"] = -3.3 * dy["^FVX"] - 3.3 * hy + 0.00022
    r["JNK"] = r["HYG"] + noise(0.0008)
    r["AGG"] = -6.0 * (0.5 * dy["^FVX"] + 0.5 * dy["^TNX"]) - 2.0 * ig + 0.00012
    # FX quotes: EURUSD/GBPUSD in USD per foreign unit; JPY=X/CNY=X in foreign per USD
    r["EURUSD=X"], r["GBPUSD=X"] = eur, D.GBP
    r["JPY=X"], r["CNY=X"] = -D.JPY, -D.CNY
    r["GLD"], r["USO"], r["DBB"] = D.GOLD, D.OIL - 0.0002, D.METALS
    r["BTC-USD"] = D.BTC + 0.0012
    for s, (b, sec, bs, iv) in STOCKS.items():
        r[s] = b * D.EQ + bs * D[sec] + noise(iv)
    R = pd.DataFrame(r, index=idx)

    # explosive run-ups over the last 18 months (convex, accelerating log path)
    k = 380
    tt = np.linspace(0, 1, k)
    for s, size in (("NVDA", 1.1), ("BTC-USD", 0.9)):
        path = size * (np.exp(3 * tt) - 1) / (np.exp(3) - 1)
        R.loc[R.index[-k:], s] += np.diff(path, prepend=0.0)

    prices = 100 * np.exp(R.cumsum())
    prices.iloc[0] = 100.0
    for s in tenors:
        prices[s] = lv[s]
    # VIX tracks the equity-vol regime plus noise
    vix = 100 * np.sqrt(252) * 0.0105 * m * np.exp(0.12 * rng.standard_normal(n))
    prices["^VIX"] = pd.Series(vix, index=idx).ewm(span=5).mean().clip(lower=9)
    # Fama-French daily factors (decimal simple returns), noisy versions of the styles
    ff = pd.DataFrame({"Mkt-RF": np.expm1(D.EQ - 0.00008), "SMB": np.expm1(0.8 * D.SIZE + noise(0.002)),
                       "HML": np.expm1(1.2 * D.VALUE + noise(0.002)),
                       "RMW": np.expm1(D.QUAL + noise(0.0015)),
                       "CMA": np.expm1(0.6 * D.VALUE + noise(0.0015)),
                       "Mom": np.expm1(1.5 * D.MOM + noise(0.003)), "RF": 0.0001}, index=idx)
    # short histories, as in real data
    for s, first in (("MTUM", "2013-04-18"), ("VLUE", "2013-04-18"), ("QUAL", "2013-07-18"),
                     ("USMV", "2011-10-20"), ("SPLV", "2011-05-05"), ("BTC-USD", "2014-09-17"),
                     ("HYG", "2007-04-11"), ("DBB", "2007-01-05")):
        prices.loc[prices.index < first, s] = np.nan
    return MarketData(prices=prices, french=ff, source="synthetic")
