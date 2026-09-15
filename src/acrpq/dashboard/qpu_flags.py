"""Fail-closed feature flags for the QPU pipeline (Phase 7).

A real IBM submission is only ever allowed when ALL of these align:

    ACRPQ_QPU_ENABLED=true          # the QPU workflow is switched on at all
    ACRPQ_IBM_SUBMISSION_ENABLED=true   # real (payable) submission is allowed
    ACRPQ_QPU_DRY_RUN=false         # not forced into dry-run-only mode

Every flag is fail-closed: an absent or unrecognised value means the SAFE
setting. ``ACRPQ_QPU_DRY_RUN`` defaults to ``true`` (force dry-run) precisely so
that a missing/typo'd value can never enable a real submission. Even a fully
UI-confirmed run cannot submit while ``submission_allowed()`` is False.
"""

from __future__ import annotations

import os
from typing import Callable

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _read_bool(name: str, *, default: bool, env: Callable[[str], str | None]) -> bool:
    raw = env(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v == "":
        return default  # explicitly-empty (unresolved interpolation/empty secret) == unset
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return default  # unrecognised -> the safe default (fail closed)


def qpu_enabled(*, env: Callable[[str], str | None] = os.environ.get) -> bool:
    """Whether the QPU workflow is switched on at all (default False)."""
    return _read_bool("ACRPQ_QPU_ENABLED", default=False, env=env)


def dry_run_only(*, env: Callable[[str], str | None] = os.environ.get) -> bool:
    """Whether the pipeline is forced into dry-run-only mode (default True)."""
    return _read_bool("ACRPQ_QPU_DRY_RUN", default=True, env=env)


def ibm_submission_enabled(*, env: Callable[[str], str | None] = os.environ.get) -> bool:
    """Whether real (payable) IBM submission is allowed (default False)."""
    return _read_bool("ACRPQ_IBM_SUBMISSION_ENABLED", default=False, env=env)


def submission_allowed(*, env: Callable[[str], str | None] = os.environ.get) -> bool:
    """A real submission is allowed ONLY if all three flags align. Default False."""
    return (qpu_enabled(env=env)
            and ibm_submission_enabled(env=env)
            and not dry_run_only(env=env))


def flags_snapshot(*, env: Callable[[str], str | None] = os.environ.get) -> dict[str, bool]:
    """A JSON-native snapshot of the effective flags for the UI/API."""
    return {
        "qpu_enabled": qpu_enabled(env=env),
        "dry_run_only": dry_run_only(env=env),
        "ibm_submission_enabled": ibm_submission_enabled(env=env),
        "submission_allowed": submission_allowed(env=env),
    }
