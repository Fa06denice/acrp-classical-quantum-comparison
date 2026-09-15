"""Statevector-simulation memory predictor and admission control (pure stdlib).

Classical simulation of an ``N``-qubit statevector stores ``2**N`` complex128
amplitudes (16 bytes each), so memory **doubles per qubit** — the exponential
wall that bounds how large an instance a laptop can simulate and, in turn,
motivates real quantum hardware. This module makes that wall explicit and
refuses statevector runs that would exceed a safe fraction of installed RAM
(rather than silently swapping to death).

The target-RAM ceiling is a documented constant (24 GiB on the target M4 Pro),
further capped by physical RAM detected on the executing host without a
``psutil`` dependency. The Aer sampling/QAOA path was measured to use ~1.05x
the theoretical statevector size (no 2x scratch copy), so a 0.58 safe fraction
on the target (~14 GiB) admits up to 29 qubits (8 GiB) while refusing 30
(16 GiB). Smaller hosts receive a lower limit.

Matrix-product-state (MPS) memory is bond-dimension dependent, not ``2**N``, so
this guard never refuses MPS — it only advises monitoring RSS.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

TOTAL_RAM_GIB: float = 24.0  # documented constant for the target machine
SAFE_FRACTION: float = 0.58  # ~14 GiB budget; admits 29 qubits, refuses 30
_BYTES_PER_AMPLITUDE = 16  # complex128
_GIB = 1024**3

# simulation methods that scale as 2**N (statevector family)
_EXACT_METHODS = {"statevector", "automatic"}
_SUPPORTED_METHODS = _EXACT_METHODS | {"matrix_product_state"}


def statevector_bytes(n_qubits: int) -> int:
    """Bytes for an ``n_qubits`` complex128 statevector (``2**N * 16``)."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 0:
        raise ValueError("n_qubits must be an integer >= 0")
    return (2**n_qubits) * _BYTES_PER_AMPLITUDE


def detected_ram_bytes() -> int | None:
    """Best-effort physical RAM detection using only the standard library."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    try:  # macOS does not expose SC_PHYS_PAGES on every Python build
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0 and int(result.stdout) > 0:
            return int(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def safe_budget_bytes() -> int:
    """Usable RAM budget, capped by both the target and detected machine RAM."""
    documented = int(TOTAL_RAM_GIB * _GIB)
    detected = detected_ram_bytes()
    physical = min(documented, detected) if detected is not None else documented
    return int(physical * SAFE_FRACTION)


def max_safe_qubits() -> int:
    """Largest ``N`` whose statevector fits the safe budget."""
    budget = safe_budget_bytes()
    n = 0
    while statevector_bytes(n + 1) <= budget:
        n += 1
    return n


@dataclass(frozen=True)
class MemoryVerdict:
    ok: bool  # False only when a statevector run would exceed the budget
    warn: bool  # True when close to the budget, or for MPS (bond-dim unknown)
    predicted_bytes: int | None  # None for MPS (not 2**N)
    message: str


def check_memory(n_qubits: int, method: str = "statevector") -> MemoryVerdict:
    """Admission check for simulating ``n_qubits`` with ``method``.

    Statevector/automatic: refuse past the safe budget. MPS: always allowed,
    with an advisory warning (memory depends on entanglement, not ``2**N``).
    """
    method = method.lower()
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"unsupported Aer simulation method {method!r}")
    if method not in _EXACT_METHODS:  # matrix_product_state and friends
        return MemoryVerdict(
            ok=True,
            warn=True,
            predicted_bytes=None,
            message=(
                f"MPS ('{method}') memory is bond-dimension dependent, not 2**N; "
                f"not bounded by the statevector wall — monitor RSS."
            ),
        )
    predicted = statevector_bytes(n_qubits)
    budget = safe_budget_bytes()
    ok = predicted <= budget
    warn = predicted > 0.75 * budget
    gib = predicted / _GIB
    bgib = budget / _GIB
    if ok:
        msg = f"statevector {n_qubits}q ~= {gib:.2f} GiB (budget {bgib:.1f} GiB)"
    else:
        msg = (
            f"statevector {n_qubits}q ~= {gib:.1f} GiB exceeds safe budget "
            f"{bgib:.1f} GiB (max {max_safe_qubits()} qubits on {TOTAL_RAM_GIB:.0f} "
            f"GiB RAM). Use --sim-method matrix_product_state, reduce n_theta/K, "
            f"or sub-sample aircraft."
        )
    return MemoryVerdict(ok=ok, warn=warn, predicted_bytes=predicted, message=msg)


def ram_table(max_n: int = 32) -> list[tuple[int, float, bool]]:
    """(n_qubits, statevector_GiB, within_budget) rows for documentation."""
    budget = safe_budget_bytes()
    return [
        (n, statevector_bytes(n) / _GIB, statevector_bytes(n) <= budget)
        for n in range(18, max_n + 1)
    ]
