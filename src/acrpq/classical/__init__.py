"""Classical ACRP solvers.

* :class:`~acrpq.classical.reference.DiscreteReferenceSolver` — exact
  brute-force over the maneuver grid, pure stdlib, **always runnable**. It is
  the ground-truth anchor for the benchmark.
* :class:`~acrpq.classical.pyomo_minlp.PyomoMINLPSolver` — a faithful Pyomo
  rebuild of ``Code/NC_ACRP.mod`` driven by a MINLP solver (Couenne by
  default). Requires the ``[classical]`` extra and an installed solver, so it
  is imported lazily.
"""

from .base import ClassicalSolver
from .reference import DiscreteReferenceSolver

__all__ = ["ClassicalSolver", "DiscreteReferenceSolver"]
