"""Exception hierarchy for the acrpq package.

All exceptions derive from :class:`ACRPError` so callers can catch the whole
family with a single ``except``. Errors are explicit and carry actionable
messages (e.g. the exact ``pip install`` line for a missing optional
dependency); the codebase never relies on broad ``except Exception`` blocks.
"""

from __future__ import annotations


class ACRPError(Exception):
    """Base class for every error raised by acrpq."""


class ParseError(ACRPError):
    """Raised when an AMPL ``.dat`` instance file cannot be parsed."""


class InstanceError(ACRPError):
    """Raised when a parsed instance is internally inconsistent."""


class QubitBudgetError(ACRPError):
    """Raised when a QUBO would need more qubits than the configured budget.

    The quantum layer refuses to silently build an intractable circuit; the
    caller must explicitly raise ``max_qubits`` or reduce the scenario.
    """


class SearchBudgetError(ACRPError):
    """Raised when an exact discrete search would exceed its space budget."""


class OptionalDependencyError(ACRPError):
    """Raised when an optional dependency (Pyomo, Qiskit, ...) is missing.

    The message always includes the exact extra to install, e.g.::

        pip install "acrp-quantum[quantum]"
    """
