import numpy as np
import pandas as pd

from lantern.factors.covariance import ewma_covariance, nearest_psd
from lantern.risk.var import christoffersen, kupiec, var_es
from lantern.scenarios.propagation import conditional


def test_covariance_is_psd_and_symmetric(lantern_demo):
    C = lantern_demo.cov.values
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-14


def test_nearest_psd_repairs_and_keeps_diagonal():
    A = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    P = nearest_psd(A)
    assert np.linalg.eigvalsh(P).min() >= -1e-12
    assert np.allclose(np.diag(P), 1.0)


def test_ewma_equal_weight_matches_sample():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((500, 3)) @ np.array([[1, .5, 0], [0, 1, .3], [0, 0, 1]]))
    C = ewma_covariance(X, 0, 0, 0).values
    S = (X.values.T @ X.values) / len(X)
    assert np.allclose(C, S, rtol=1e-6)


def test_euler_contributions_add_up(lantern_demo):
    r = lantern_demo.risk()
    assert np.isclose(r.by_factor["contribution_%"].sum() + r.by_group.loc["Specific", "contribution_%"],
                      100 * r.total_vol, rtol=1e-8)
    assert np.isclose(r.by_position["contribution_%"].sum(), 100 * r.total_vol, rtol=1e-8)


def test_gaussian_mc_var_matches_parametric(lantern_demo):
    p = lantern_demo.var("parametric", 0.99)
    g = lantern_demo.var("gaussian", 0.99, n=200_000)
    assert abs(g["VaR"] / p["VaR"] - 1) < 0.05          # options/convexity create small gaps


def test_component_es_sums_to_es(lantern_demo):
    r = lantern_demo.var("student_t", 0.975, n=50_000)
    assert np.isclose(r["contributions"]["ES_contribution"].sum(), r["ES"], rtol=1e-6)


def test_var_es_definition():
    pnl = -np.arange(1000.0)
    v, es = var_es(pnl, 0.99)
    assert v <= es and np.isclose(v, np.quantile(np.arange(1000.0), 0.99))


def test_kupiec_and_christoffersen():
    LR, p = kupiec(1000, 10, 0.99)
    assert LR < 1e-8 and p > 0.99
    assert kupiec(1000, 30, 0.99)[1] < 0.001
    hits = np.zeros(500, bool)
    hits[100:110] = True                                   # clustered exceptions
    assert christoffersen(hits)[1] < 0.01


def test_conditional_expectation_formula():
    S = np.array([[4.0, 1.2, 0.0], [1.2, 1.0, 0.3], [0.0, 0.3, 2.0]])
    mu, C, d2 = conditional(S, [0], np.array([2.0]))
    assert np.allclose(mu, [2.0, 1.2 * 2 / 4, 0.0])
    assert np.isclose(C[1, 1], 1.0 - 1.2 ** 2 / 4)
    assert np.isclose(d2, 1.0)
