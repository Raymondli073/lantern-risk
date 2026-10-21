"""Load a factor file from the Jensen-Kelly-Pedersen database (jkpfactors.com).

The download is in long format with (at least) columns `name`, `date`, `ret`; this
returns a wide monthly panel (factors in columns) of decimal returns.
"""
from __future__ import annotations

import pandas as pd


def load_jkp(path, start=None, end=None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in {"name", "date", "ret"})
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    wide = df.pivot_table(index="date", columns="name", values="ret").sort_index()
    return wide.loc[start:end]
