"""Real-time explosiveness detection (Phillips, Shi & Yu, 2015) as used in
Li & Urga, "Timing the Factor Zoo: Explosiveness Beyond Momentum".

  Test series   : a total-return price index in levels (cumulative factor value for a
                  long-short factor).
  Regression    : dY_t = mu + beta Y_{t-1} + e_t, no lag augmentation.
  Statistic     : E_t = BSADF_t = sup over start points r1 of the DF t-stat on [r1, t].
                  It only uses data up to t, so it is available in real time.
  Critical value: Monte Carlo under Y_t = sum_{s<=t} (e_s + 1/T), e ~ N(0, 1):
                  'scv'   = quantile of SADF_t = sup_{s<=t} ADF_0^s (the main sequence),
                  'point' = pointwise quantile of BSADF_t.
  Indicator     : I_t = 1{E_t > cv_t and r_t > 0}, observed at the end of period t and
                  acted on in t + 1.

The O(T) prefix-sum recursion makes each BSADF_t cost O(t) instead of O(t^2) regressions,
the same idea as the recursive-least-squares speed-up of the thesis' IPCA-RLS algorithm.
numba is used when installed; otherwise a vectorized numpy path gives identical output.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                                   # pragma: no cover
    HAVE_NUMBA = False

RESOURCES = Path(__file__).resolve().parent / "resources"


def _prefix(y):
    x, d = y[:-1], np.diff(y)
    z = np.zeros(1)
    return (np.concatenate([z, np.cumsum(x)]), np.concatenate([z, np.cumsum(x * x)]),
            np.concatenate([z, np.cumsum(d)]), np.concatenate([z, np.cumsum(d * d)]),
            np.concatenate([z, np.cumsum(x * d)]))


def _bsadf_numpy(y, swin, wmax):
    T = y.size
    P1, P11, Pd, Pdd, P1d = _prefix(y)
    out = np.empty(T - swin)
    arg = np.empty(T - swin, dtype=np.int64)
    for r2 in range(swin, T):
        lo = max(0, r2 - wmax) if wmax > 0 else 0
        r1 = np.arange(lo, r2 - swin + 1)
        n = (r2 - r1).astype(float)
        Sx, Sxx = P1[r2] - P1[r1], P11[r2] - P11[r1]
        Sd, Sdd, Sxd = Pd[r2] - Pd[r1], Pdd[r2] - Pdd[r1], P1d[r2] - P1d[r1]
        sxx = Sxx - Sx * Sx / n
        sxd = Sxd - Sx * Sd / n
        sdd = Sdd - Sd * Sd / n
        with np.errstate(divide="ignore", invalid="ignore"):
            b = sxd / sxx
            rss = sdd - b * sxd
            t = b / np.sqrt(rss / (n - 2) / sxx)
        t = np.where((sxx > 0) & (rss > 0), t, -1e300)
        j = int(np.argmax(t))
        out[r2 - swin], arg[r2 - swin] = t[j], r1[j]
    return out, arg


if HAVE_NUMBA:
    @njit(cache=True)
    def _bsadf_numba(y, swin, wmax):
        T = y.size
        P1 = np.zeros(T); P11 = np.zeros(T); Pd = np.zeros(T); Pdd = np.zeros(T); P1d = np.zeros(T)
        for k in range(T - 1):
            x = y[k]
            d = y[k + 1] - y[k]
            P1[k + 1] = P1[k] + x
            P11[k + 1] = P11[k] + x * x
            Pd[k + 1] = Pd[k] + d
            Pdd[k + 1] = Pdd[k] + d * d
            P1d[k + 1] = P1d[k] + x * d
        out = np.empty(T - swin)
        arg = np.empty(T - swin, dtype=np.int64)
        for r2 in range(swin, T):
            lo = 0
            if wmax > 0 and r2 - wmax > 0:
                lo = r2 - wmax
            best = -1e300
            bi = lo
            for r1 in range(lo, r2 - swin + 1):
                n = r2 - r1
                Sx = P1[r2] - P1[r1]
                Sxx = P11[r2] - P11[r1]
                Sd = Pd[r2] - Pd[r1]
                Sdd = Pdd[r2] - Pdd[r1]
                Sxd = P1d[r2] - P1d[r1]
                sxx = Sxx - Sx * Sx / n
                if sxx <= 0.0:
                    continue
                sxd = Sxd - Sx * Sd / n
                sdd = Sdd - Sd * Sd / n
                b = sxd / sxx
                rss = sdd - b * sxd
                if rss <= 0.0:
                    continue
                t = b / np.sqrt(rss / (n - 2) / sxx)
                if t > best:
                    best = t
                    bi = r1
            out[r2 - swin] = best
            arg[r2 - swin] = bi
        return out, arg

    @njit(parallel=True, cache=True)
    def _sim_bsadf(T, swin, eps):
        m = eps.shape[1]
        a = 1.0 / T
        bs = np.empty((m, T - swin))
        for j in prange(m):
            y = np.empty(T)
            c = 0.0
            for t in range(T):
                c += eps[t, j] + a
                y[t] = c
            bs[j, :] = _bsadf_numba(y, swin, 0)[0]
        return bs


def bsadf(y, swin: int, wmax: int = 0, engine: str = "auto"):
    """BSADF_t for t = swin..T-1 (0-based observation index) and the start point r1 that
    attains the sup (the beginning of the explosive window). wmax <= 0: expanding."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    if engine == "numba" or (engine == "auto" and HAVE_NUMBA):
        return _bsadf_numba(y, swin, wmax)
    return _bsadf_numpy(y, swin, wmax)


def _sadf_null(T, swin, eps):
    """Forward-expanding sup-ADF sequence under the null for many paths (vectorized)."""
    y = np.cumsum(eps + 1.0 / T, axis=0)                   # (T, m)
    x, d = y[:-1], np.diff(y, axis=0)
    n = np.arange(1, T, dtype=float)[:, None]
    Sx, Sxx = np.cumsum(x, 0), np.cumsum(x * x, 0)
    Sd, Sdd, Sxd = np.cumsum(d, 0), np.cumsum(d * d, 0), np.cumsum(x * d, 0)
    sxx = Sxx - Sx * Sx / n
    sxd = Sxd - Sx * Sd / n
    sdd = Sdd - Sd * Sd / n
    with np.errstate(divide="ignore", invalid="ignore"):
        b = sxd / sxx
        t = b / np.sqrt((sdd - b * sxd) / (n - 2) / sxx)
    t = t[swin - 1:]                                        # n pairs = swin .. T-1
    return np.maximum.accumulate(np.nan_to_num(t, nan=-1e300), axis=0).T   # (m, T - swin)


def simulate_cv(T: int = 1000, swin: int = 36, n_sim: int = 5000, seed: int = 20251231,
                levels=(0.90, 0.95, 0.99), point: bool = False) -> dict:
    """Critical values as a function of the number of observations n = swin+1..T."""
    rng = np.random.default_rng(seed)
    sa, bs, done = [], [], 0
    while done < n_sim:
        m = min(500, n_sim - done)
        eps = rng.standard_normal((T, m))
        sa.append(_sadf_null(T, swin, eps))
        if point:
            if HAVE_NUMBA:
                bs.append(_sim_bsadf(T, swin, eps))
            else:
                y = np.cumsum(eps + 1.0 / T, axis=0)
                bs.append(np.array([_bsadf_numpy(y[:, j], swin, 0)[0] for j in range(m)]))
        done += m
    out = {"n": np.arange(swin + 1, T + 1), "levels": np.array(levels),
           "scv": np.quantile(np.vstack(sa), levels, axis=0).T}
    if point:
        out["point"] = np.quantile(np.vstack(bs), levels, axis=0).T
    return out


@lru_cache(maxsize=8)
def cv_table(T: int = 1000, swin: int = 36, n_sim: int = 20000) -> dict:
    """Packaged critical values when available, otherwise simulated (scv only)."""
    f = RESOURCES / f"cv_scv_T{T}_s{swin}_n{n_sim}.csv"
    if f.exists():
        df = pd.read_csv(f)
        lv = np.array([float(c[2:]) / 100 for c in df.columns if c.startswith("cv")])
        return {"n": df["n"].values, "levels": lv,
                "scv": df[[c for c in df.columns if c.startswith("cv")]].values}
    return simulate_cv(T, swin, n_sim)


def cv_for_length(tab: dict, n, kind: str = "scv", level: float = 0.95) -> np.ndarray:
    j = int(np.argmin(np.abs(tab["levels"] - level)))
    col = tab[kind][:, j]
    ix = np.clip(np.asarray(n) - tab["n"][0], 0, len(col) - 1)
    return col[ix]


def statistic(price: pd.Series, swin: int = 36, wmax: int = 0, logs: bool = False):
    """(E_t, window start dates) on a price series; real-time at every date."""
    p = price.dropna()
    if len(p) <= swin + 1:
        return pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
    y = np.log(p.values) if logs else p.values / p.values[0]
    e, arg = bsadf(y, swin, wmax)
    idx = p.index[swin:]
    return pd.Series(e, index=idx), pd.Series(p.index[arg], index=idx)


def panel(prices: pd.DataFrame, swin: int = 36, wmax: int = 0, logs: bool = False,
          kind: str = "scv", level: float = 0.95, T: int = 1000):
    """(E, CV, I) frames on the index of `prices` (one column per series)."""
    tab = cv_table(T, swin) if kind == "scv" else simulate_cv(T, swin, 2000, point=True)
    E = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    CV = E.copy()
    for c in prices.columns:
        e, _ = statistic(prices[c], swin, wmax, logs)
        if e.empty:
            continue
        E.loc[e.index, c] = e.values
        first = prices[c].first_valid_index()
        n = prices.index.get_indexer(e.index) - prices.index.get_loc(first) + 1
        if wmax > 0:
            n = np.minimum(n, wmax + 1)
        CV.loc[e.index, c] = cv_for_length(tab, n, kind, level)
    r = prices.pct_change(fill_method=None)
    I = ((E > CV) & (r > 0)).astype(float).where(E.notna())
    return E, CV, I


def episodes(E: pd.Series, CV: pd.Series, min_length: int = 1) -> pd.DataFrame:
    """PSY date-stamping: origination = first date E > CV, termination = first date
    back below. Returns one row per episode."""
    above = (E > CV).fillna(False)
    rows, start = [], None
    for t, a in above.items():
        if a and start is None:
            start = t
        elif not a and start is not None:
            rows.append((start, t))
            start = None
    if start is not None:
        rows.append((start, None))
    df = pd.DataFrame(rows, columns=["origination", "termination"])
    if len(df) and min_length > 1:
        lens = [above.loc[s:e].sum() if e is not None else above.loc[s:].sum()
                for s, e in rows]
        df = df[np.array(lens) >= min_length]
    return df.reset_index(drop=True)


def current_state(price: pd.Series, swin: int = 36, level: float = 0.95, T: int = 1000) -> dict:
    """Latest reading for one monthly price series: statistic, critical value, whether
    the indicator is on, and the origination date and run-up of the current episode."""
    p = price.dropna()
    e, start = statistic(p, swin)
    if e.empty:
        return {"E": np.nan, "cv": np.nan, "explosive": False, "signal": False}
    tab = cv_table(T, swin)
    n = np.arange(swin + 1, len(p) + 1)
    cv = pd.Series(cv_for_length(tab, n, "scv", level), index=e.index)
    eps = episodes(e, cv)
    live = len(eps) and pd.isna(eps.iloc[-1]["termination"])
    origin = eps.iloc[-1]["origination"] if live else None
    r_last = p.iloc[-1] / p.iloc[-2] - 1
    runup = float(np.log(p.iloc[-1] / p.loc[:origin].iloc[-2])) if live and \
        len(p.loc[:origin]) > 1 else 0.0
    return {"E": float(e.iloc[-1]), "cv": float(cv.iloc[-1]),
            "explosive": bool(e.iloc[-1] > cv.iloc[-1]),
            "signal": bool(e.iloc[-1] > cv.iloc[-1] and r_last > 0),
            "origination": origin, "window_start": start.iloc[-1], "runup_log": runup,
            "E_series": e, "cv_series": cv}
