import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def lantern_demo(tmp_path_factory):
    os.environ.setdefault("LANTERN_CACHE", str(tmp_path_factory.mktemp("cache")))
    from lantern import Lantern, Portfolio
    pf = Portfolio.from_yaml(ROOT / "examples" / "multi_asset.yaml")
    return Lantern(pf, synthetic=True)
