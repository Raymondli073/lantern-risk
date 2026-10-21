"""Factor covariance estimation.

The default follows the practice of commercial risk models: exponentially weighted
volatilities with a short half-life (fast reaction to regime changes) combined with
exponentially weighted correlations with a longer half-life (stable structure),
estimated on zero-mean daily moves with pairwise-complete data, then repaired to be
positive semi-definite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _weights(n: int, halflife: float) -> np.ndarray:
    if halflife is None or halflife <= 0:
        return np.ones(n)
    return 0.5 ** (np.arange(n)[::-1] / halflife)


def nearest_psd(S: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Clip negative eigenvalues, then restore the original diagonal."""
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    if vals.min() >= floor * max(vals.max(), 1e-30):
        return S
    vals = np.clip(vals, floor * vals.max(), None)
    P = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(S), 1e-30, None) / np.clip(np.diag(P), 1e-30, None))
    return P * np.outer(d, d)


def ewma_covariance(moves: pd.DataFrame, vol_halflife: float = 63, corr_halflife: float = 252,
                    lookback: int = 2520, end=None, shrink: float = 0.0,
                    min_obs: int = 60) -> pd.DataFrame:
    """Daily covariance of factor moves.

    vol_halflife / corr_halflife : days (<= 0 means equal weights)
    lookback                     : number of most recent days used
    shrink                       : weight on the identity in the correlation matrix
    Factors with fewer than `min_obs` observations in the window get NaN rows/columns.
    """
    X = moves.loc[:end].iloc[-lookback:] if lookback else moves.loc[:end]
    cols = X.columns
    M = X.notna().values.astype(float)
    X0 = np.nan_to_num(X.values)
    n = len(X)

    wv = _weights(n, vol_halflife)[:, None]
    var = (wv * X0 ** 2).sum(0) / np.clip((wv * M).sum(0), 1e-30, None)

    wc = _weights(n, corr_halflife)[:, None]
    Xw = X0 * np.sqrt(wc)
    S = Xw.T @ Xw                                   # sum w x_i x_j over joint obs
    A = (wc * X0 ** 2).T @ M                        # A_ij = sum w x_i^2 m_j
    denom = np.sqrt(np.clip(A * A.T, 1e-300, None))
    R = S / denom
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -0.999, 0.999)
    np.fill_diagonal(R, 1.0)
    if shrink:
        R = (1 - shrink) * R + shrink * np.eye(len(cols))
    ok = M.sum(0) >= min_obs
    R = nearest_psd(np.where(np.outer(ok, ok), R, np.eye(len(cols))))
    sd = np.sqrt(var)
    C = R * np.outer(sd, sd)
    C[~ok, :] = np.nan
    C[:, ~ok] = np.nan
    return pd.DataFrame(C, index=cols, columns=cols)


def sample_covariance(moves: pd.DataFrame, start=None, end=None, min_obs: int = 20) -> pd.DataFrame:
    """Equal-weighted covariance over a fixed window (e.g. a stress period)."""
    return ewma_covariance(moves.loc[start:end], vol_halflife=0, corr_halflife=0,
                           lookback=0, min_obs=min_obs)


def fill_missing(cov: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Fill factors that are NaN in `cov` (absent from a stress window, e.g. Bitcoin in
    2008) from `fallback`: their correlations come from `fallback`, and their vols are
    the fallback vols scaled by the median stress/fallback vol ratio of the other factors."""
    bad = cov.isna().all(axis=1).values
    if not bad.any():
        return cov
    c, f = cov.values, fallback.loc[cov.index, cov.columns].values
    vc, vf = np.sqrt(np.diag(c)), np.sqrt(np.diag(f))
    ratio = np.nanmedian(vc[~bad] / vf[~bad])
    vol = np.where(bad, vf * ratio, vc)
    R = np.where(np.outer(~bad, ~bad), c / np.outer(vc, vc), f / np.outer(vf, vf))
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.0)
    R = nearest_psd(R)
    return pd.DataFrame(R * np.outer(vol, vol), index=cov.index, columns=cov.columns)


def correlation(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov.values))
    return pd.DataFrame(cov.values / np.outer(d, d), index=cov.index, columns=cov.columns)


def annualized_vol(cov: pd.DataFrame, days: int = 252) -> pd.Series:
    return pd.Series(np.sqrt(np.diag(cov.values) * days), index=cov.index)
