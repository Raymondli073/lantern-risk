"""Build daily factor moves (and levels) from MarketData using the factor recipes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import DEFAULT_UNIVERSE, FactorSpec

# moves larger than these are bad prints, not market moves
_MAX_ABS = {"return": 0.45, "rate": 0.01, "spread": 0.02, "vol": 0.6}


def _logret(p: pd.Series) -> pd.Series:
    p = p.where(p > 0)
    return np.log(p).diff()


def _recipe(recipe: str, market) -> pd.Series:
    how, arg = recipe.split(":", 1)
    px = market.prices
    if how == "ret":
        return _logret(px[arg])
    if how == "inv":
        return -_logret(px[arg])
    if how == "level":
        return px[arg].diff() / 100.0
    if how == "ls":
        a, b = arg.split("/")
        return _logret(px[a]) - _logret(px[b])
    if how == "credit":
        pair, dc, dg = arg.split(":")
        c, g = pair.split("/")
        rc, rg = px[c].pct_change(fill_method=None), px[g].pct_change(fill_method=None)
        return -(rc - float(dc) / float(dg) * rg) / float(dc)
    if how == "ff":
        if market.french is None or arg not in market.french:
            return pd.Series(np.nan, index=px.index)
        return np.log1p(market.french[arg]).reindex(px.index)
    raise ValueError(f"unknown recipe {recipe!r}")


def _backfill(x: pd.Series, proxy: pd.Series, min_overlap: int = 250) -> pd.Series:
    """Fill x before its first observation with beta * proxy (beta from the overlap)."""
    first = x.first_valid_index()
    both = pd.concat([x, proxy], axis=1).dropna()
    if first is None or len(both) < min_overlap:
        return x
    beta = float((both.iloc[:, 0] * both.iloc[:, 1]).sum() / (both.iloc[:, 1] ** 2).sum())
    fill = beta * proxy[proxy.index < first]
    return x.combine_first(fill)


def build_factors(market, universe=DEFAULT_UNIVERSE):
    """Return (moves, levels).

    moves  : DataFrame of daily factor moves in each factor's natural unit.
    levels : DataFrame of factor levels: yields/VIX in decimal for rate/vol factors,
             and a cumulative index exp(sum of log returns) for return factors.
    """
    moves, levels = {}, {}
    for f in universe:
        try:
            x = _recipe(f.recipe, market)
        except KeyError:
            x = pd.Series(np.nan, index=market.prices.index)
        for pr in f.proxies:
            try:
                x = _backfill(x, _recipe(pr, market))
            except KeyError:
                pass
        x = x.where(x.abs() <= _MAX_ABS[f.kind])
        moves[f.id] = x
        if f.kind in ("rate", "vol"):
            how, arg = f.recipe.split(":", 1)
            levels[f.id] = market.prices[arg] / 100.0
        elif f.kind == "return":
            levels[f.id] = np.exp(x.fillna(0).cumsum()).where(x.notna().cummax())
        else:
            levels[f.id] = x.fillna(0).cumsum().where(x.notna().cummax())
    moves = pd.DataFrame(moves).iloc[1:]
    levels = pd.DataFrame(levels).iloc[1:]
    return moves, levels


def spec_frame(universe=DEFAULT_UNIVERSE) -> pd.DataFrame:
    return pd.DataFrame([{"id": f.id, "name": f.name, "group": f.group, "kind": f.kind,
                          "recipe": f.recipe, "proxies": ", ".join(f.proxies)}
                         for f in universe]).set_index("id")


def unit_label(kind: str) -> str:
    return {"return": "%", "rate": "bp", "spread": "bp", "vol": "vol pts"}[kind]


def to_display(value: float, kind: str) -> float:
    """Natural units -> display units (% for returns, bp for rates/spreads, points for vol)."""
    if kind == "return":
        return 100 * (np.exp(value) - 1)
    if kind in ("rate", "spread"):
        return 1e4 * value
    return 100 * value


def from_display(value: float, kind: str) -> float:
    if kind == "return":
        return float(np.log1p(value / 100))
    if kind in ("rate", "spread"):
        return value / 1e4
    return value / 100
