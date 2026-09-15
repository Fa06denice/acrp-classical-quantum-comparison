"""acrpq — Classical vs quantum/hybrid benchmark for the 2D ACRP.

This package extends the existing AMPL ACRP benchmark library (``Code/`` and
``Data/`` at the repository root) with a Python benchmark framework that runs
both a **classical baseline** and a **quantum/hybrid (QAOA)** approach on the
*same* problem instances through *one common interface*, and compares them with
shared metrics.

Design contract
---------------
``import acrpq`` and the entire pure-Python core import with **zero**
third-party dependencies. Heavy libraries (Pyomo/AMPL, Qiskit, numpy,
matplotlib) are optional extras imported lazily only inside the features that
need them. To keep this guarantee, this module does not import the submodules
eagerly; the most common public symbols are re-exported lazily via PEP 562
``__getattr__``.

Example
-------
>>> from acrpq import InstanceLoader, DiscreteReferenceSolver
>>> inst = InstanceLoader().load("CP_3")          # doctest: +SKIP
>>> result = DiscreteReferenceSolver().solve(inst)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Map public name -> (submodule, attribute). Kept lazy so that pulling, say,
# ``InstanceLoader`` does not drag in the quantum stack.
_LAZY: dict[str, tuple[str, str]] = {
    # model
    "Instance": ("acrpq.model", "Instance"),
    "Maneuver": ("acrpq.model", "Maneuver"),
    "ManeuverOption": ("acrpq.model", "ManeuverOption"),
    "ManeuverGrid": ("acrpq.model", "ManeuverGrid"),
    "Solution": ("acrpq.model", "Solution"),
    "Result": ("acrpq.model", "Result"),
    "BenchmarkRecord": ("acrpq.model", "BenchmarkRecord"),
    "Family": ("acrpq.model", "Family"),
    "SolverKind": ("acrpq.model", "SolverKind"),
    "Status": ("acrpq.model", "Status"),
    # io
    "InstanceLoader": ("acrpq.io.loader", "InstanceLoader"),
    # discretize
    "DiscreteACRP": ("acrpq.discretize", "DiscreteACRP"),
    # classical
    "DiscreteReferenceSolver": ("acrpq.classical.reference", "DiscreteReferenceSolver"),
    # quantum (pure-python parts only; QAOA solver requires the [quantum] extra)
    "build_qubo": ("acrpq.quantum.qubo", "build_qubo"),
    "ManeuverQUBO": ("acrpq.quantum.qubo", "ManeuverQUBO"),
    # exceptions
    "ACRPError": ("acrpq.exceptions", "ACRPError"),
    "OptionalDependencyError": ("acrpq.exceptions", "OptionalDependencyError"),
}

if TYPE_CHECKING:  # help type checkers without runtime cost; re-exported lazily below
    from .discretize import DiscreteACRP as DiscreteACRP
    from .exceptions import ACRPError as ACRPError
    from .exceptions import OptionalDependencyError as OptionalDependencyError
    from .io.loader import InstanceLoader as InstanceLoader
    from .model import BenchmarkRecord as BenchmarkRecord
    from .model import Family as Family
    from .model import Instance as Instance
    from .model import Maneuver as Maneuver
    from .model import ManeuverGrid as ManeuverGrid
    from .model import ManeuverOption as ManeuverOption
    from .model import Result as Result
    from .model import Solution as Solution
    from .model import SolverKind as SolverKind
    from .model import Status as Status
    from .quantum.qubo import ManeuverQUBO as ManeuverQUBO
    from .quantum.qubo import build_qubo as build_qubo


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'acrpq' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY.keys()])


__all__ = ["__version__", *_LAZY.keys()]
