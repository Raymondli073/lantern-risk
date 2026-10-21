import numpy as np
import pandas as pd

from lantern.data.french import parse_first_table
from lantern.systematic import explosive as ex
from lantern.systematic import optimizer as opt
from lantern.systematic import signals
from lantern.systematic.common import pc1_expost, pc1_realtime


def test_numpy_and_numba_bsadf_agree():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(300)) + 30
    a, ia = ex.bsadf(y, 36, 0, engine="numpy")
    if ex.HAVE_NUMBA:
        b, ib = ex.bsadf(y, 36, 0, engine="numba")
        assert np.allclose(a, b) and (ia == ib).all()
    assert len(a) == 300 - 36


def test_bsadf_detects_an_explosive_segment():
    rng = np.random.default_rng(4)
    y = np.empty(240)
    y[0] = 100
    for t in range(1, 240):
        rho = 1.03 if t > 200 else 1.0
        y[t] = rho * y[t - 1] + rng.standard_normal()
    p = pd.Series(y, index=pd.date_range("2000-01-31", periods=240, freq="ME"))
    st = ex.current_state(p)
    assert st["explosive"] and st["origination"] > pd.Timestamp("2015-06-30")


def test_critical_values_increase_with_level_and_size():
    tab = ex.cv_table()
    assert (np.diff(tab["scv"], axis=1) > 0).all()
    small = ex.simulate_cv(200, 36, 2000)
    assert small["scv"][-1, 1] > 0.8                 # 95% scv is roughly 1.2-1.5 at n=200


def test_tsfe_tsfm_strategies_shapes():
    rng = np.random.default_rng(5)
    idx = pd.date_range("1990-01-31", periods=300, freq="ME")
    R = pd.DataFrame(rng.normal(0.004, 0.03, (300, 6)), index=idx, columns=list("abcdef"))
    out = signals.run_all(R, swin=36, cost_bps=10)
    assert set(out.columns) == {"TSFE (L)", "TSFE (LS)", "TSFM (L)", "TSFM (LS)", "Buy & hold (EW)"}
    assert out.notna().all().all()
    w = signals.weights(signals.tsfm_signal(R), "LS")
    sig = signals.tsfm_signal(R)
    both = (sig == 1).any(axis=1) & (sig == 0).any(axis=1)
    assert np.allclose(w[both].sum(axis=1), 0, atol=1e-12)


def test_pca_sign_and_shape():
    rng = np.random.default_rng(6)
    f = np.cumsum(rng.standard_normal(200))
    X = pd.DataFrame(np.column_stack([f + rng.standard_normal(200) * 0.3 for _ in range(5)]))
    a, b = pc1_expost(X), pc1_realtime(X)
    assert np.corrcoef(a, f)[0, 1] > 0.9 and np.corrcoef(b, f)[0, 1] > 0.8


def test_french_parser():
    text = ("This file was created ...\n\n,Mkt-RF,SMB,HML,RF\n196307,   -0.39,   -0.41,   -0.97,    0.27\n"
            "196308,    5.07,   -0.80,    1.80,    0.25\n\n Annual Factors: January-December \n,Mkt-RF\n1964,1\n")
    df = parse_first_table(text)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and len(df) == 2
    assert np.isclose(df.iloc[1, 0], 0.0507) and df.index[0] == pd.Timestamp("1963-07-31")


def test_factor_targeting_hits_targets(lantern_demo):
    ctx = lantern_demo.ctx
    syms = ["SPY", "EFA", "EEM", "MTUM", "IEF", "GLD"]
    out = opt.factor_targeting(ctx, syms, lantern_demo.cov, {"EQ.US": 0.4, "UST.10Y": -2.0}, hi=0.6)
    w = out["weights"]
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all() and (w <= 0.6 + 1e-9).all()
    assert abs(out["achieved"]["EQ.US"] - 0.4) < 0.03


def test_risk_parity_equalizes_contributions(lantern_demo):
    ctx = lantern_demo.ctx
    syms = ["SPY", "IEF", "GLD", "EEM"]
    Sigma, _ = opt.asset_covariance(ctx, syms, lantern_demo.cov)
    w = opt.risk_parity(Sigma)
    rc = opt.risk_contributions(w, Sigma)
    assert np.allclose(rc, 0.25, atol=1e-3)
