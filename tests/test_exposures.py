import numpy as np
import pandas as pd

from lantern.factors.exposures import estimate_exposures


def test_adaptive_lasso_recovers_sparse_betas():
    rng = np.random.default_rng(7)
    n, k = 1000, 12
    C = 0.6 * np.ones((k, k)) + 0.4 * np.eye(k)              # correlated factors
    X = pd.DataFrame(rng.multivariate_normal(np.zeros(k), C, n) * 0.01,
                     columns=[f"f{i}" for i in range(k)])
    beta = np.zeros(k)
    beta[[0, 3]] = [1.2, -0.5]
    y = pd.Series(X.values @ beta + rng.standard_normal(n) * 0.004, name="y")
    fit = estimate_exposures(y, X, "adaptive_lasso", halflife=0, lookback=n)
    sel = fit.beta[fit.beta.abs() > 1e-10].index.tolist()
    assert set(["f0", "f3"]).issubset(sel) and len(sel) <= 4
    assert np.isclose(fit.beta["f0"], 1.2, atol=0.05)
    assert 0.003 < fit.specific_vol < 0.005


def test_exposures_on_synthetic_market(lantern_demo):
    f = lantern_demo.ctx.fit("AGG")
    rates = f.beta[["UST.3M", "UST.5Y", "UST.10Y", "UST.30Y"]].sum()
    assert -6.5 < rates < -5.5                            # ~6-year duration
