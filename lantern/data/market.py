"""MarketData: one aligned panel of daily prices on the US trading calendar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass
class MarketData:
    prices: pd.DataFrame                      # adjusted closes / levels, US business days
    french: Optional[pd.DataFrame] = None     # Fama-French daily factors (decimal, simple)
    source: str = "yahoo"

    @property
    def asof(self) -> pd.Timestamp:
        return self.prices.index[-1]

    def price(self, symbol: str, asof=None) -> float:
        s = self.prices[symbol].loc[:asof or self.asof].dropna()
        if s.empty:
            raise KeyError(f"no price for {symbol} on or before {asof}")
        return float(s.iloc[-1])

    def monthly(self, symbols=None) -> pd.DataFrame:
        """Month-end prices. The last row is the latest close, so the current month
        is 'forming' (a real-time, month-to-date reading)."""
        p = self.prices if symbols is None else self.prices[list(symbols)]
        m = p.resample("ME").last()
        m.index = m.index[:-1].append(pd.DatetimeIndex([p.index[-1]]))
        return m


def align(raw: pd.DataFrame, calendar_symbol: str = "SPY", ffill_limit: int = 5) -> pd.DataFrame:
    """Put every series on the calendar of `calendar_symbol` (NYSE days). 24/7 and FX
    series take their last value on or before each date, so weekend moves are booked
    on Monday."""
    raw = raw.sort_index()
    cal = raw[calendar_symbol].dropna().index if calendar_symbol in raw else raw.index
    full = raw.reindex(raw.index.union(cal)).ffill(limit=ffill_limit)
    return full.reindex(cal)


def load_market(symbols: Iterable[str], start: str = "2004-01-01", refresh: bool = False,
                include_french: bool = True) -> MarketData:
    """Download (or read from cache) everything needed for a risk run."""
    from . import yahoo, french
    symbols = list(dict.fromkeys(["SPY", *symbols]))
    raw = yahoo.prices(symbols, refresh=refresh)
    prices = align(raw).loc[start:]
    ff = None
    if include_french:
        try:
            ff = french.us_factors(daily=True)
        except Exception:                     # the library is optional for risk runs
            ff = None
    return MarketData(prices=prices, french=ff, source="yahoo")
