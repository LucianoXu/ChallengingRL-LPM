"""Compatibility entry point for the publication MiniGrid figures.

The plotting implementation now lives in ``reports/make_publication_figures.py``
so MiniGrid and MiniWorld share one visual style.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reports"))

from make_publication_figures import main  # noqa: E402


if __name__ == "__main__":
    main(["--domain", "minigrid", *sys.argv[1:]])
