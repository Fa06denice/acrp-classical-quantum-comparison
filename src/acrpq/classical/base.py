"""Common protocol for classical ACRP solvers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..model import Instance, Result


@runtime_checkable
class ClassicalSolver(Protocol):
    """A classical solver maps an :class:`Instance` to a :class:`Result`.

    Implementations must always return a :class:`Result` (with an appropriate
    :class:`~acrpq.model.Status`) rather than raising on an unsolved instance,
    so the benchmark runner can record outcomes uniformly.
    """

    def solve(self, inst: Instance, *, time_limit_s: float | None = None) -> Result: ...
