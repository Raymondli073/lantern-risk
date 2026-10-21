"""Lantern: open-source cross-asset factor risk, scenario simulation and systematic
factor investing, modelled on the publicly described design of institutional
platforms such as BlackRock's Aladdin. Not affiliated with or endorsed by BlackRock."""
from .context import MarketContext
from .platform import Lantern
from .portfolio import Portfolio

__version__ = "0.1.0"
__all__ = ["Lantern", "MarketContext", "Portfolio", "__version__"]
