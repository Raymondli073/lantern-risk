"""The cross-asset risk-factor universe.

Each factor has a *kind* that fixes its units:

    return   daily log return                       (equity, style, sector, FX, commodity, crypto)
    rate     daily change in a yield, decimal       (key-rate factors; 0.0001 = 1 bp)
    spread   daily change in a credit spread, decimal
    vol      daily change in implied volatility, decimal (VIX/100)

and a *recipe* that builds it from public price series:

    ret:SYM            log return of SYM
    inv:SYM            minus the log return of SYM (e.g. JPY=X is quoted JPY per USD)
    ls:A/B             log return of A minus log return of B (a long-short style/sector factor)
    level:SYM          change in SYM's level / 100 (yield indices and VIX are quoted in percent)
    credit:C/G:Dc:Dg   spread change implied by a corporate-bond ETF C hedged with a Treasury
                       ETF G:  ds = -(r_C - (Dc/Dg) r_G) / Dc
    ff:COL             a Fama-French daily factor (log of 1 + simple return); used as a proxy

Optional `proxies` backfill a factor before its primary series starts, scaled by
the regression beta on the overlap (the same way risk vendors proxy short histories).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class FactorSpec:
    id: str
    name: str
    group: str
    kind: str
    recipe: str
    proxies: Tuple[str, ...] = field(default_factory=tuple)
    tenor: Optional[float] = None       # years, for key-rate factors

    @property
    def symbols(self) -> list:
        out = []
        for r in (self.recipe,) + tuple(self.proxies):
            out += recipe_symbols(r)
        return out


def recipe_symbols(recipe: str) -> list:
    how, arg = recipe.split(":", 1)
    if how in ("ret", "inv", "level"):
        return [arg]
    if how == "ls":
        return arg.split("/")
    if how == "credit":
        return arg.split(":")[0].split("/")
    return []                            # ff: comes from the French library


GROUPS = ["Rates", "Credit", "Inflation", "Equity", "Style", "Sector", "FX",
          "Commodity", "Crypto", "Volatility"]

DEFAULT_UNIVERSE = [
    # --- rates: US Treasury key rates (Yahoo yield indices) ------------------------
    FactorSpec("UST.3M", "US Treasury 3M yield", "Rates", "rate", "level:^IRX", tenor=0.25),
    FactorSpec("UST.5Y", "US Treasury 5Y yield", "Rates", "rate", "level:^FVX", tenor=5.0),
    FactorSpec("UST.10Y", "US Treasury 10Y yield", "Rates", "rate", "level:^TNX", tenor=10.0),
    FactorSpec("UST.30Y", "US Treasury 30Y yield", "Rates", "rate", "level:^TYX", tenor=30.0),
    # --- credit spreads implied by duration-hedged ETF returns ------------------------
    FactorSpec("CREDIT.IG", "USD investment-grade spread", "Credit", "spread",
               "credit:LQD/IEF:8.3:7.2"),
    FactorSpec("CREDIT.HY", "USD high-yield spread", "Credit", "spread",
               "credit:HYG/IEI:3.3:4.3", proxies=("credit:JNK/IEI:3.4:4.3",)),
    # --- inflation: long TIPS / short nominals ----------------------------------------
    FactorSpec("INFL.BE", "US breakeven inflation (TIPS - nominal)", "Inflation", "return",
               "ls:TIP/IEF"),
    # --- equity markets -----------------------------------------------------------------
    FactorSpec("EQ.US", "US equity market", "Equity", "return", "ret:SPY"),
    FactorSpec("EQ.DMxUS", "Developed ex-US equity", "Equity", "return", "ret:EFA"),
    FactorSpec("EQ.EM", "Emerging-market equity", "Equity", "return", "ret:EEM"),
    # --- US equity styles (iShares factor ETFs vs the market; French factors backfill) ---
    FactorSpec("STYLE.SIZE", "Size (small minus large)", "Style", "return", "ls:IWM/SPY",
               proxies=("ff:SMB",)),
    FactorSpec("STYLE.VALUE", "Value", "Style", "return", "ls:VLUE/SPY",
               proxies=("ls:IWD/IWF", "ff:HML")),
    FactorSpec("STYLE.MOM", "Momentum", "Style", "return", "ls:MTUM/SPY", proxies=("ff:Mom",)),
    FactorSpec("STYLE.QUALITY", "Quality", "Style", "return", "ls:QUAL/SPY", proxies=("ff:RMW",)),
    FactorSpec("STYLE.MINVOL", "Minimum volatility", "Style", "return", "ls:USMV/SPY",
               proxies=("ls:SPLV/SPY",)),
    # --- sectors relative to the market ---------------------------------------------------
    FactorSpec("SECTOR.TECH", "Technology vs market", "Sector", "return", "ls:XLK/SPY"),
    FactorSpec("SECTOR.FIN", "Financials vs market", "Sector", "return", "ls:XLF/SPY"),
    FactorSpec("SECTOR.ENERGY", "Energy vs market", "Sector", "return", "ls:XLE/SPY"),
    # --- FX (foreign currency vs USD; positive = foreign currency appreciates) -------------
    FactorSpec("FX.EUR", "EUR/USD", "FX", "return", "ret:EURUSD=X"),
    FactorSpec("FX.JPY", "JPY/USD", "FX", "return", "inv:JPY=X"),
    FactorSpec("FX.GBP", "GBP/USD", "FX", "return", "ret:GBPUSD=X"),
    FactorSpec("FX.CNY", "CNY/USD", "FX", "return", "inv:CNY=X"),
    # --- commodities ---------------------------------------------------------------------
    FactorSpec("CMDTY.GOLD", "Gold", "Commodity", "return", "ret:GLD"),
    FactorSpec("CMDTY.OIL", "Crude oil", "Commodity", "return", "ret:USO"),
    FactorSpec("CMDTY.METALS", "Industrial metals", "Commodity", "return", "ret:DBB"),
    # --- crypto and volatility -----------------------------------------------------------
    FactorSpec("CRYPTO.BTC", "Bitcoin", "Crypto", "return", "ret:BTC-USD"),
    FactorSpec("VOL.VIX", "S&P 500 implied volatility (VIX)", "Volatility", "vol", "level:^VIX"),
]


def universe_symbols(universe=DEFAULT_UNIVERSE) -> list:
    out = []
    for f in universe:
        out += f.symbols
    return list(dict.fromkeys(out))


def by_id(universe=DEFAULT_UNIVERSE) -> dict:
    return {f.id: f for f in universe}
