"""QUBO -> Ising conversion (pure Python).

Substituting ``x = (1 - z) / 2`` with ``z in {+1, -1}`` maps the QUBO
``H(x) = c + sum_b a_b x_b + sum_{b<c} q_{bc} x_b x_c`` to an Ising Hamiltonian
``H(z) = offset + sum_b h_b z_b + sum_{b<c} J_{bc} z_b z_c``.

Used for metrics (term counts / structure) and as an explicit, dependency-free
representation of the Hamiltonian. The QAOA solver builds its operator directly
from the QUBO via ``qiskit-optimization``; this module is the auditable
reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from .qubo import ManeuverQUBO


@dataclass(slots=True)
class IsingModel:
    """Ising form: ``offset + sum h_b z_b + sum_{b<c} J_{bc} z_b z_c``."""

    h: dict[int, float]
    J: dict[tuple[int, int], float]
    offset: float
    n_spins: int


def qubo_to_ising(qubo: ManeuverQUBO) -> IsingModel:
    """Convert a :class:`ManeuverQUBO` to its Ising form."""
    n = qubo.n_qubits
    h: dict[int, float] = {b: 0.0 for b in range(n)}
    J: dict[tuple[int, int], float] = {}
    offset = qubo.constant

    # linear: a_b x_b = a_b (1 - z_b)/2 = a_b/2 - a_b/2 z_b
    for b, a in qubo.linear.items():
        offset += a / 2.0
        h[b] -= a / 2.0

    # quadratic: q (1-z_a)/2 (1-z_b)/2 = q/4 (1 - z_a - z_b + z_a z_b)
    for (a, b), q in qubo.quadratic.items():
        offset += q / 4.0
        h[a] -= q / 4.0
        h[b] -= q / 4.0
        key = (a, b) if a < b else (b, a)
        J[key] = J.get(key, 0.0) + q / 4.0

    # drop ~zero linear terms for compactness
    h = {b: v for b, v in h.items() if abs(v) > 1e-12}
    return IsingModel(h=h, J=J, offset=offset, n_spins=n)
