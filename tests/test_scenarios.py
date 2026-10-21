import numpy as np

from lantern.scenarios.library import pct


def test_explicit_shocks_are_respected(lantern_demo):
    r = lantern_demo.scenario({"EQ.US": pct(-0.2)}, horizon_days=21)
    assert np.isclose(r.factor_moves["EQ.US"], pct(-0.2))
    # with the synthetic market, the implied move of developed ex-US equity is negative
    assert r.factor_moves["EQ.DMxUS"] < 0
    assert r.pnl < 0 and r.pred_error_95 > 0


def test_security_shock_moves_the_market_with_it(lantern_demo):
    r = lantern_demo.scenario({}, security_shocks={"NVDA": np.log(0.7)})
    assert r.factor_moves["SECTOR.TECH"] < 0
    nvda = r.pnl_by_position["NVDA"]
    assert np.isclose(nvda, 350_000 * (0.7 - 1), rtol=1e-6)


def test_historical_and_library_run(lantern_demo):
    h = lantern_demo.historical_scenarios()
    y = lantern_demo.hypothetical_scenarios()
    assert len(h) >= 8 and len(y) >= 10


def test_bubble_scan_flags_the_synthetic_bubble(lantern_demo):
    flags = lantern_demo.bubble_scan()
    assert flags.loc["NVDA", "explosive"]
    assert flags.loc["CRYPTO.BTC", "explosive"]
    assert not flags.loc["EQ.US", "explosive"]
    b = lantern_demo.bubble_scenarios(flags)
    assert all(s.pnl < 0 for s in b)
