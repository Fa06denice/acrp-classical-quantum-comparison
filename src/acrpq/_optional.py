"""Optional-dependency guard.

The acrpq core is pure-stdlib. Heavier libraries (Pyomo/AMPL for the faithful
MINLP baseline, Qiskit for the quantum layer, numpy/matplotlib) are optional
*extras*. This module centralises the lazy-import policy so that:

* the core never imports a heavy library at module import time;
* a missing dependency yields a clear :class:`OptionalDependencyError` naming
  the exact ``pip install`` line, rather than a bare ``ModuleNotFoundError``;
* reproducibility records can list the resolved version of every relevant
  library, or ``"absent"`` when it is not installed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from types import ModuleType

from .exceptions import OptionalDependencyError

# module import name -> pyproject extra that provides it
_EXTRAS: dict[str, str] = {
    "pyomo": "classical",
    "amplpy": "classical",
    "qiskit": "quantum",
    "qiskit_aer": "quantum",
    "qiskit_optimization": "quantum",
    "numpy": "quantum",
    "qiskit_ibm_runtime": "ibm",
    "matplotlib": "viz",
    "dimod": "dwave",
    "dwave": "dwave",  # covers dwave.system + dwave.samplers via root split
    "minorminer": "dwave",
}

# Import name -> installed distribution name. Metadata lookup avoids importing
# packages that initialise caches, providers, or solver runtimes as a side effect.
_DISTRIBUTIONS: dict[str, str] = {
    "pyomo": "pyomo",
    "amplpy": "amplpy",
    "qiskit": "qiskit",
    "qiskit_aer": "qiskit-aer",
    "qiskit_optimization": "qiskit-optimization",
    "numpy": "numpy",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
    "matplotlib": "matplotlib",
    "dimod": "dimod",
    "dwave": "dwave-system",
    "minorminer": "minorminer",
}


def require(module: str) -> ModuleType:
    """Import and return ``module`` or raise a helpful error.

    Parameters
    ----------
    module:
        Importable module name, e.g. ``"qiskit_optimization"``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # narrow: only import failures
        # _EXTRAS is keyed on the top-level distribution (e.g. "pyomo"), so map
        # a submodule like "pyomo.environ" back to its root before lookup.
        root = module.split(".", 1)[0]
        extra = _EXTRAS.get(root)
        hint = (
            f'pip install "acrp-quantum[{extra}]"'
            if extra
            else f"pip install {root}"
        )
        raise OptionalDependencyError(
            f"Optional dependency '{module}' is required for this feature but is "
            f"not installed. Install it with:\n    {hint}"
        ) from exc


def have(module: str) -> bool:
    """Return ``True`` if ``module`` can be imported, without raising."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def versions() -> dict[str, str]:
    """Map every known optional dependency to its version or ``"absent"``.

    Used by the reproducibility record so each result file documents exactly
    which solver/quantum stack produced it.
    """
    out: dict[str, str] = {}
    for module, distribution in _DISTRIBUTIONS.items():
        try:
            out[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            out[module] = "absent"
    return out
