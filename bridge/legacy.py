"""
Bridge to the linear CustomNerd / CloudNerd backend.

The agent backend reuses retrieval, relevance, and synthesis modules from
customnerd-backend without modifying that codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from config import LEGACY_BACKEND_ROOT

_initialized = False


def get_legacy_root() -> Path:
    return LEGACY_BACKEND_ROOT


def ensure_legacy_backend() -> Path:
    """Add customnerd-backend to sys.path once so imports work."""
    global _initialized
    root = LEGACY_BACKEND_ROOT
    if not root.is_dir():
        raise RuntimeError(
            f"Legacy backend not found at {root}. "
            "Expected sibling folder customnerd-backend/."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _initialized = True
    return root


def import_legacy(module_name: str):
    ensure_legacy_backend()
    return __import__(module_name)
