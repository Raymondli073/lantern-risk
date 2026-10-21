"""Factor returns from the Kenneth R. French Data Library.

Returns are converted from percent to decimal simple returns. Only the first
table of each file is read (the monthly or daily table; annual tables follow).
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from .cache import cache_dir

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"

# (file name, columns to keep, prefix for the column names)
US_MONTHLY = [
    ("F-F_Research_Data_5_Factors_2x3", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], "US"),
    ("F-F_Momentum_Factor", ["Mom"], "US"),
    ("F-F_ST_Reversal_Factor", ["ST_Rev"], "US"),
    ("F-F_LT_Reversal_Factor", ["LT_Rev"], "US"),
]
INTL_REGIONS = {"Developed_ex_US": "DMxUS", "Europe": "EU", "Japan": "JP",
                "Asia_Pacific_ex_Japan": "APxJ", "North_America": "NA", "Emerging": "EM"}


def _download(name: str) -> str:
    p = cache_dir("french") / f"{name}.csv"
    fresh = p.exists() and (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")
                            < pd.Timedelta(days=7))
    if not fresh:
        r = requests.get(_BASE.format(name=name), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        p.write_text(z.read(z.namelist()[0]).decode("latin-1"))
    return p.read_text()


def parse_first_table(text: str) -> pd.DataFrame:
    """Parse the first date-indexed table of a French-library CSV."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().startswith(",") and any(c.isalpha() for c in ln))
    rows = []
    for ln in lines[start + 1:]:
        cells = [c.strip() for c in ln.split(",")]
        if not cells[0].isdigit():
            break
        rows.append(cells)
    cols = ["date"] + [c.strip() for c in lines[start].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    d = df.pop("date")
    fmt = "%Y%m%d" if len(d.iloc[0]) == 8 else "%Y%m"
    idx = pd.to_datetime(d, format=fmt)
    if fmt == "%Y%m":
        idx = idx + pd.offsets.MonthEnd(0)
    df.index = idx
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df[df <= -0.99] = float("nan")          # -99.99 / -999 are missing-value codes
    return df


def load(name: str, daily: bool = False) -> pd.DataFrame:
    return parse_first_table(_download(name + ("_daily" if daily else "")))


def us_factors(daily: bool = False) -> pd.DataFrame:
    """US Fama-French 5 factors + momentum + short/long-term reversal (excess returns)."""
    frames = []
    for name, cols, _ in US_MONTHLY:
        df = load(name, daily)
        frames.append(df[[c for c in cols if c in df.columns]])
    return pd.concat(frames, axis=1, join="inner")


def factor_zoo(include_international: bool = True) -> pd.DataFrame:
    """Monthly long-short factor returns: US (8) and, optionally, six international
    regions (5 factors + momentum each). The market factor is in excess of RF."""
    us = us_factors(daily=False)
    zoo = us.drop(columns=["RF"]).add_prefix("US.")
    if include_international:
        for region, tag in INTL_REGIONS.items():
            f5 = load(f"{region}_5_Factors")
            mom = load(f"{region}_Mom_Factor")
            f = pd.concat([f5.drop(columns=["RF"], errors="ignore"), mom], axis=1)
            f.columns = [c.strip().replace("WML", "Mom") for c in f.columns]
            zoo = zoo.join(f.add_prefix(f"{tag}."), how="outer")
    return zoo.sort_index()
