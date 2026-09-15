"""Tests for the statevector RAM guard (pure Python, no deps)."""

from __future__ import annotations

from acrpq.quantum.memguard import (
    check_memory,
    max_safe_qubits,
    ram_table,
    statevector_bytes,
)


def test_statevector_bytes():
    assert statevector_bytes(28) == 4 * 2**30  # 4 GiB
    assert statevector_bytes(30) == 16 * 2**30


def test_max_safe_qubits_is_29():
    # The documented 24 GiB target caps this at 29; smaller hosts are safer.
    assert max_safe_qubits() <= 29


def test_statevector_refused_past_budget():
    # Robust to host RAM: the guard admits up to max_safe_qubits and refuses N+1.
    n = max_safe_qubits()
    assert check_memory(n, "statevector").ok is True
    over = check_memory(n + 1, "statevector")
    assert over.ok is False
    assert "exceeds" in over.message


def test_mps_never_refused():
    v = check_memory(40, "matrix_product_state")
    assert v.ok is True
    assert v.predicted_bytes is None


def test_boundary_29_30():
    assert check_memory(max_safe_qubits(), "statevector").ok
    assert not check_memory(30, "statevector").ok


def test_ram_table_marks_wall():
    rows = {n: within for n, _gib, within in ram_table()}
    assert rows[max_safe_qubits()] is True
    assert rows[30] is False


def test_memory_guard_rejects_invalid_inputs():
    import pytest

    with pytest.raises(ValueError, match="n_qubits"):
        statevector_bytes(-1)
    with pytest.raises(ValueError, match="unsupported"):
        check_memory(10, "typo")
