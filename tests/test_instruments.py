import numpy as np
import pandas as pd

from lantern.instruments import Bond, EquityOption, bs_greeks, bs_price
from lantern.instruments.curve import Curve


def test_curve_weights_partition_of_unity():
    c = Curve([0.25, 5, 10, 30], [0.04, 0.042, 0.045, 0.048], ["a", "b", "c", "d"])
    W = c.weights([0.1, 1, 7, 10, 20, 40])
    assert np.allclose(W.sum(axis=1), 1)
    assert np.allclose(c.zero([5.0]), 0.042)
    assert np.allclose(c.zero([40.0]), 0.048)          # flat extrapolation


def test_bond_key_rate_exposures_match_finite_differences(lantern_demo):
    ctx = lantern_demo.ctx
    b = Bond(id="b", face=1e6, coupon=0.05, maturity="2034-05-15", spread=0.01,
             spread_factor="CREDIT.IG")
    e = b.exposures(ctx)
    h = 1e-6
    for fid in ["UST.5Y", "UST.10Y", "CREDIT.IG"]:
        F = np.zeros((2, len(ctx.factor_ids)))
        F[0, ctx.index(fid)], F[1, ctx.index(fid)] = h, -h
        p = b.revalue(ctx, F, {})
        fd = (p[0] - p[1]) / (2 * h)
        assert abs(fd - e[ctx.index(fid)]) / abs(fd) < 1e-4
    # analytic sensitivities sum to the full curve duration
    rate_ids = [f for f in ctx.factor_ids if ctx.specs[f].kind == "rate"]
    tot = sum(e[ctx.index(f)] for f in rate_ids)
    a = b.analytics(ctx)
    assert np.isclose(-tot / b.value(ctx), a["duration"], rtol=1e-6)


def test_bond_convexity_is_positive(lantern_demo):
    ctx = lantern_demo.ctx
    b = Bond(id="b", face=1e6, coupon=0.04, maturity="2050-01-01")
    F = np.zeros((2, len(ctx.factor_ids)))
    for f in ["UST.3M", "UST.5Y", "UST.10Y", "UST.30Y"]:
        F[0, ctx.index(f)], F[1, ctx.index(f)] = 0.01, -0.01
    up, dn = b.revalue(ctx, F, {})
    assert up + dn > 0                                   # P(+100bp) + P(-100bp) > 2 P(0)


def test_black_scholes_put_call_parity_and_delta():
    S, K, T, r, q, v = 100.0, 95.0, 0.75, 0.04, 0.01, 0.2
    c, p = bs_price(S, K, T, r, q, v, "call"), bs_price(S, K, T, r, q, v, "put")
    assert np.isclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T))
    d = bs_greeks(S, K, T, r, q, v, "put")["delta"]
    fd = (bs_price(S + 1e-4, K, T, r, q, v, "put") - bs_price(S - 1e-4, K, T, r, q, v, "put")) / 2e-4
    assert np.isclose(d, fd, atol=1e-6)


def test_put_hedges_equity_down_move(lantern_demo):
    ctx = lantern_demo.ctx
    o = EquityOption(id="p", underlying="SPY", right="put", strike_pct=0.95,
                     expiry="2027-06-18", quantity=10)
    F = np.zeros((1, len(ctx.factor_ids)))
    F[0, ctx.index("EQ.US")] = np.log(0.8)
    assert o.revalue(ctx, F, {})[0] > 0
