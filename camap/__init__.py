"""CaMAP — Calcium imaging mapping and analysis pipeline."""

from pathlib import Path as _Path

from camap.dataset import (
    ArenaDataset,
    BasePlaceCellDataset,
    MazeDataset,
    StabilitySplitResult,
    UnitResult,
    unique_bundle_path,
)


def _resolve_version() -> str:
    """Live via setuptools_scm in a git checkout, else the hatch-vcs-written ``_version.py``."""
    repo_root = _Path(__file__).resolve().parent.parent
    if (repo_root / ".git").is_dir():
        try:
            from setuptools_scm import get_version

            return get_version(root=str(repo_root))
        except Exception:
            pass
    try:
        from camap._version import __version__ as _v

        return _v
    except ImportError:
        return "0.0.0.dev0"


__version__ = _resolve_version()


__all__ = [
    "ArenaDataset",
    "BasePlaceCellDataset",
    "MazeDataset",
    "StabilitySplitResult",
    "UnitResult",
    "unique_bundle_path",
    "__version__",
]
