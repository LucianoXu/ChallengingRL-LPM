"""Path helpers: make `miniworld_play` importable from the experiments pkg."""
import os
import sys

# experiments/ -> Miniworld/ -> LPM_exploration/ -> <repo root>
_THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
PKG_DIR = _THIS


def ensure_repo_on_path() -> str:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT
