"""Scenario libraries: historical replays and forward-looking hypothetical shocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .propagation import run_scenario


def pct(x):    # simple % move -> log return
    return float(np.log1p(x))


def bp(x):
    return x / 1e4


def pts(x):    # implied-vol points
    return x / 100


@dataclass(frozen=True)
class Episode:
    name: str
    start: str      # last close before the episode
    end: str
    description: str


HISTORICAL = [
    Episode("GFC: Lehman to the Nov-08 low", "2008-09-12", "2008-11-20",
            "Lehman default, money-market run, global deleveraging."),
    Episode("US downgrade / euro crisis 2011", "2011-07-22", "2011-10-03",
            "S&P downgrade of the US, euro-area sovereign stress."),
    Episode("Taper tantrum 2013", "2013-05-02", "2013-06-24",
            "Fed signals tapering of QE; global bond sell-off."),
    Episode("China devaluation 2015", "2015-08-10", "2015-08-25",
            "CNY devaluation, EM and commodity sell-off."),
    Episode("Volmageddon 2018", "2018-01-26", "2018-02-08",
            "Short-volatility unwind, VIX doubles."),
    Episode("Q4 2018 sell-off", "2018-10-03", "2018-12-24",
            "Fed tightening and trade war; equities -20%."),
    Episode("COVID-19 crash", "2020-02-19", "2020-03-23",
            "Pandemic shock: fastest bear market on record, dash for cash."),
    Episode("2022 inflation and rates shock", "2022-01-03", "2022-10-12",
            "Fastest Fed hiking cycle in 40 years; stocks and bonds fall together."),
    Episode("SVB / regional banks 2023", "2023-03-08", "2023-03-17",
            "Bank runs, front-end rally, financials sell off."),
    Episode("Yen carry-trade unwind 2024", "2024-07-10", "2024-08-05",
            "BoJ hike and weak US payrolls; JPY rallies, Nikkei -20%."),
    Episode("Tariff shock 2025", "2025-04-02", "2025-04-08",
            "US 'Liberation Day' tariffs; equities and the USD fall, term premium rises."),
]


HYPOTHETICAL: Dict[str, dict] = {
    "Equity bear market (S&P 500 -20%)": {"EQ.US": pct(-0.20)},
    "Hawkish Fed (+75 bp front end)": {"UST.3M": bp(75), "UST.5Y": bp(45)},
    "Fed cuts, bull steepener": {"UST.3M": bp(-100), "UST.5Y": bp(-60), "UST.30Y": bp(-10)},
    "Credit crunch (HY +250 bp)": {"CREDIT.HY": bp(250), "CREDIT.IG": bp(80)},
    "Oil supply shock (+40%)": {"CMDTY.OIL": pct(0.40)},
    "USD surge (+8%)": {"FX.EUR": pct(-0.08), "FX.JPY": pct(-0.08)},
    "Stagflation": {"UST.10Y": bp(75), "EQ.US": pct(-0.15), "CMDTY.OIL": pct(0.30),
                    "INFL.BE": pct(0.03)},
    "AI / tech de-rating": {"SECTOR.TECH": pct(-0.15), "EQ.US": pct(-0.12),
                            "STYLE.MOM": pct(-0.08)},
    "Crypto winter (BTC -60%)": {"CRYPTO.BTC": pct(-0.60)},
    "Yen carry unwind": {"FX.JPY": pct(0.10), "EQ.DMxUS": pct(-0.10), "VOL.VIX": pts(15)},
    "Volatility spike (VIX +20)": {"VOL.VIX": pts(20)},
    "China hard landing": {"EQ.EM": pct(-0.20), "FX.CNY": pct(-0.06),
                           "CMDTY.METALS": pct(-0.20)},
    "Gold rally (+20%)": {"CMDTY.GOLD": pct(0.20)},
}


def episode_moves(ctx, ep: Episode) -> pd.Series:
    """Cumulative factor moves over an episode; NaN where a factor did not exist."""
    X = ctx.moves.loc[pd.Timestamp(ep.start) + pd.Timedelta(days=1):ep.end]
    cnt = X.notna().sum()
    out = X.sum()
    out[cnt < 0.5 * len(X)] = np.nan
    return out


def run_historical(portfolio, ctx, cov: pd.DataFrame, ep: Episode, propagate_missing=True):
    """Replay an episode. Factors without history at the time (e.g. Bitcoin in 2008) are
    filled by their conditional expectation given the observed factors."""
    m = episode_moves(ctx, ep)
    obs = m.dropna()
    if obs.empty:
        return None
    h = max(len(ctx.moves.loc[pd.Timestamp(ep.start) + pd.Timedelta(days=1):ep.end]), 1)
    if not propagate_missing:
        m = m.fillna(0.0)
        obs = m
    res = run_scenario(portfolio, ctx, cov, obs.to_dict(), name=ep.name, horizon_days=h,
                       description=ep.description)
    return res


def run_library(portfolio, ctx, cov: pd.DataFrame, horizon_days: int = 21, severity: float = 1.0):
    out = []
    for name, shocks in HYPOTHETICAL.items():
        shocks = {k: v * severity for k, v in shocks.items() if k in ctx.factor_ids}
        if shocks:
            out.append(run_scenario(portfolio, ctx, cov, shocks, name=name,
                                    horizon_days=horizon_days))
    return out


def summary_table(results) -> pd.DataFrame:
    rows = [{"scenario": r.name, "P&L $": r.pnl, "P&L %": r.pnl_pct,
             "±95% $": r.pred_error_95, "shock size (σ)": r.mahalanobis,
             "1-in-N years": r.return_period_years, "horizon (d)": r.horizon_days}
            for r in results if r is not None]
    return pd.DataFrame(rows).set_index("scenario")
