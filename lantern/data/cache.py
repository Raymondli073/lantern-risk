"""Location of the local download cache (override with LANTERN_CACHE)."""
from __future__ import annotations

import os
from pathlib import Path


def cache_dir(sub: str = "") -> Path:
    root = Path(os.environ.get("LANTERN_CACHE", Path.home() / ".cache" / "lantern"))
    p = root / sub if sub else root
    p.mkdir(parents=True, exist_ok=True)
    return p
