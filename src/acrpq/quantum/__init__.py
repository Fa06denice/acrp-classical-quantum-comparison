"""Quantum/hybrid ACRP layer.

Pipeline::

    DiscreteACRP --build_qubo--> ManeuverQUBO --(QAOA)--> bitstring --decode--> Result

The QUBO/Ising construction and decoding are pure Python (no Qiskit); only the
QAOA solver and backends require the ``[quantum]`` / ``[ibm]`` extras and are
imported lazily.
"""

from .decode import decode, decode_assignment
from .ising import IsingModel, qubo_to_ising
from .qubo import ManeuverQUBO, build_qubo
from .dwave_solver import DWaveAnnealingSolver, DWaveConfig, DWaveMode
from .qubo_solvers import QuboAnnealingSolver, QuboExactSolver

__all__ = [
    "ManeuverQUBO",
    "build_qubo",
    "IsingModel",
    "qubo_to_ising",
    "decode",
    "decode_assignment",
    "QuboExactSolver",
    "QuboAnnealingSolver",
    "DWaveAnnealingSolver",
    "DWaveConfig",
    "DWaveMode",
]
