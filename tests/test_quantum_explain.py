"""Scientific explainability must remain an identity over the real QUBO."""

from __future__ import annotations

import itertools
import math

import pytest

from acrpq.quantum.explain import (
    bitstring_explanation,
    coefficient_explanation,
    energy_breakdown,
    qubo_components,
    resource_analysis,
    sample_explanations,
    split_qubo_components,
)
from acrpq.quantum.qubo import build_qubo


def test_energy_breakdown_matches_encoded_energy_exhaustively(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    for bits in itertools.product((0, 1), repeat=qubo.n_qubits):
        explained = energy_breakdown(qubo, bits)
        assert explained["identity_verified"]
        assert math.isclose(explained["total"], qubo.energy(bits), abs_tol=1e-7)


def test_bitstring_explanation_discloses_onehot_repair(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    empty = "0" * qubo.n_qubits

    explained = bitstring_explanation(qubo, empty)

    assert not explained["raw_onehot_valid"]
    assert explained["repair_applied"]
    assert all(row["repair"] == "NOOP fallback" for row in explained["aircraft"])
    assert explained["decoded_choice"] == [qubo.discrete.grid.noop_index()] * cp3.n


def test_components_are_a_partition_and_resource_peak_matches(cp4):
    qubo = build_qubo(cp4, n_theta=3)
    components = qubo_components(qubo)
    flattened = sorted(bit for component in components for bit in component)

    assert flattened == list(range(qubo.n_qubits))
    analysis = resource_analysis(qubo)
    assert analysis["largest_component_qubits"] == max(map(len, components))
    assert analysis["decomposed_peak_bytes"] == 2 ** max(map(len, components)) * 16
    assert analysis["raw_statevector_bytes"] == 2**qubo.n_qubits * 16


def test_coefficient_explanations_reconstruct_coefficients(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    terms = coefficient_explanation(qubo)

    assert {row["bit"]: row["coefficient"] for row in terms["linear"]} == qubo.linear
    assert {
        (row["a"], row["b"]): row["coefficient"] for row in terms["quadratic"]
    } == qubo.quadratic
    assert {row["kind"] for row in terms["quadratic"]} <= {"onehot", "conflict"}


def test_sample_explanations_are_bounded_sorted_and_validated(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    noop = qubo.discrete.grid.noop_index()
    valid = ["0"] * qubo.n_qubits
    for i in qubo.discrete.instance.aircraft():
        valid[qubo.discrete.var_index(i, noop)] = "1"
    valid_string = "".join(valid)
    counts = {valid_string: 7, "0" * qubo.n_qubits: 3}

    rows = sample_explanations(qubo, counts, limit=1)

    assert len(rows) == 1
    assert rows[0]["bitstring"] == valid_string
    assert rows[0]["probability"] == pytest.approx(0.7)
    assert rows[0]["raw_onehot_valid"]
    with pytest.raises(ValueError, match="non-negative"):
        sample_explanations(qubo, {valid_string: -1})


def test_resource_integers_are_bounded_beyond_simulable_range():
    # Inside the simulable range the exact 2**N figures are reported verbatim;
    # past it they collapse to None so no ~300-digit integer ever hits the JSON
    # payload (and the GiB float never overflows).
    from acrpq.quantum.explain import (
        _MAX_EXACT_QUBITS,
        _exact_amplitudes,
        _exact_bytes,
        _gib,
    )

    assert _exact_amplitudes(12) == 2**12
    assert _exact_bytes(12) == 2**12 * 16
    assert _gib(30) == pytest.approx(2**30 * 16 / 1024**3)

    assert _exact_amplitudes(_MAX_EXACT_QUBITS + 1) is None
    assert _exact_bytes(200) is None
    assert _gib(300) is not None and math.isfinite(_gib(300))  # verdict stays usable
    assert _gib(2000) is None  # only None once the float would overflow


def test_split_components_preserve_energy_identity(loader):
    # FP_4 contains independent conflict groups on the real benchmark data.
    qubo = build_qubo(loader.load("FP_4"), n_theta=3)
    parts = split_qubo_components(qubo)
    assert len(parts) > 1

    bits = "".join("1" if bit % 3 == 0 else "0" for bit in range(qubo.n_qubits))
    local_energy = qubo.constant
    for component, local in parts:
        local_bits = "".join(bits[original] for original in component)
        local_energy += local.energy(local_bits)
    assert local_energy == pytest.approx(qubo.energy(bits))


def test_qaoa_decomposition_recombines_original_bit_order(loader, monkeypatch):
    from acrpq.quantum.qaoa import QAOAResult, QAOASolver

    qubo = build_qubo(loader.load("FP_4"), n_theta=3)

    def fake_solve(self, local):
        del self
        bits = "1" + "0" * (local.n_qubits - 1)
        return QAOAResult(
            best_bitstring=bits,
            best_energy=local.energy(bits),
            n_qubits=local.n_qubits,
            circuit_depth=4,
            circuit_size=10,
            two_qubit_gates=3,
            backend_meta={"name": "fake"},
            n_jobs=2,
        )

    monkeypatch.setattr(QAOASolver, "solve", fake_solve)
    solver = object.__new__(QAOASolver)  # bypass optional-dependency validation
    result = QAOASolver.solve_decomposed(solver, qubo)

    assert len(result.best_bitstring) == qubo.n_qubits
    assert result.best_energy == pytest.approx(qubo.energy(result.best_bitstring))
    assert result.backend_meta["exact_component_decomposition"] is True
    assert result.n_jobs == 2 * len(qubo_components(qubo))
    assert result.counts == {}  # no fabricated joint distribution
