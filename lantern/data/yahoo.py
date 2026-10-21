"""Daily prices from Yahoo Finance's public chart endpoint (no API key).

Adjusted closes are dividend- and split-adjusted, so equity/ETF series are
total-return indices. Yield indices (^IRX, ^FVX, ^TNX, ^TYX) and ^VIX are
levels in percent. Every download is cached as CSV; a refresh only fetches the
recent tail.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .cache import cache_dir

_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _fetch(sym: str, period1: int = 0) -> pd.Series:
    params = {"period1": period1, "period2": int(time.time()) + 86400,
              "interval": "1d", "events": "div,split", "includeAdjustedClose": "true"}
    for attempt in range(3):
        try:
            r = requests.get(_URL.format(sym=sym), params=params, headers=_HDR, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    ts = res.get("timestamp")
    if not ts:
        return pd.Series(dtype=float, name=sym)
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = (pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz)
           .tz_localize(None).normalize())
    s = pd.Series(adj, index=idx, dtype=float, name=sym)
    s = s[~s.index.duplicated(keep="last")].dropna()
    s.index.name = "date"
    return s


def _path(sym: str) -> Path:
    safe = sym.replace("^", "_").replace("=", "_").replace("/", "_")
    return cache_dir("yahoo") / f"{safe}.csv"


def history(sym: str, refresh: bool = False) -> pd.Series:
    """One symbol's adjusted-close history, extended from the cache when stale."""
    p = _path(sym)
    old = None
    if p.exists():
        old = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].rename(sym)
        fresh = (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")
                 < pd.Timedelta(hours=12))
        if not refresh and fresh:
            return old
    if old is not None and len(old) > 5:
        # re-fetch the last few weeks: adjusted closes are revised after dividends
        start = int((old.index[-1] - pd.Timedelta(days=40)).timestamp())
        new = _fetch(sym, start)
        overlap = old.reindex(new.index).dropna()
        if len(overlap) and abs(new[overlap.index[0]] / overlap.iloc[0] - 1) > 1e-6:
            out = _fetch(sym)   # the adjusted history was rescaled (dividend/split): refetch
        elif len(new):
            out = pd.concat([old[old.index < new.index[0]], new])
        else:
            out = old
    else:
        out = _fetch(sym)
    out.to_frame(sym).to_csv(p)
    return out


def prices(symbols: Iterable[str], refresh: bool = False, max_workers: int = 8) -> pd.DataFrame:
    """Adjusted closes for many symbols, outer-joined on date."""
    symbols = list(dict.fromkeys(symbols))
    with ThreadPoolExecutor(max_workers) as ex:
        out = list(ex.map(lambda s: history(s, refresh), symbols))
    return pd.concat(out, axis=1).sort_index()
