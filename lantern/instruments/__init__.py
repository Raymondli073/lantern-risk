from .base import Instrument
from .bond import Bond
from .curve import Curve
from .equity import Cash, Equity, FactorProxy, FXForward
from .option import EquityOption, bs_greeks, bs_price

TYPES = {
    "equity": Equity,
    "etf": Equity,
    "crypto": Equity,
    "bond": Bond,
    "option": EquityOption,
    "fx_forward": FXForward,
    "cash": Cash,
    "proxy": FactorProxy,
}


def make(spec: dict) -> Instrument:
    spec = dict(spec)
    kind = spec.pop("type").lower()
    cls = TYPES[kind]
    if kind == "crypto":
        spec.setdefault("asset_class", "Crypto")
    if kind == "equity" and "asset_class" not in spec:
        spec["asset_class"] = "Equity"
    return cls(**spec)


__all__ = ["Instrument", "Bond", "Curve", "Cash", "Equity", "FactorProxy", "FXForward",
           "EquityOption", "bs_price", "bs_greeks", "make", "TYPES"]
