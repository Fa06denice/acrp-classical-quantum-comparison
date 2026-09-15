"""Minimal ``.env`` loader (pure stdlib, no python-dotenv dependency).

IBM Quantum credentials and other settings can be placed in a ``.env`` file at
the repository root (or any ancestor of the working directory) instead of being
exported manually each session. The file is git-ignored and never read into any
result/output file.

Supported syntax (a small, common subset):

    # comment
    KEY=value
    KEY="quoted value"
    export KEY=value        # leading 'export' is tolerated

Only keys not already present in ``os.environ`` are set, so a real environment
variable always wins over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_NAME = ".env"


def find_dotenv(start: Path | None = None, name: str = _DEFAULT_NAME) -> Path | None:
    """Search upward from ``start`` (or cwd) for a ``.env`` file."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a mapping (no side effects)."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # strip a single layer of matching quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_dotenv(
    path: str | Path | None = None, *, override: bool = False
) -> dict[str, str]:
    """Load a ``.env`` file into ``os.environ`` and return what was applied.

    Parameters
    ----------
    path:
        Explicit ``.env`` path; if ``None``, search upward from the cwd.
    override:
        If ``False`` (default), existing environment variables are kept; if
        ``True``, the file overrides them.
    """
    dotenv_path = Path(path) if path else find_dotenv()
    if dotenv_path is None or not Path(dotenv_path).is_file():
        return {}
    values = parse_dotenv(Path(dotenv_path).read_text(encoding="utf-8"))
    applied: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
