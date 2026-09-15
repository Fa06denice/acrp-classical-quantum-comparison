"""Phase 5.2.B — versioned, immutable, fully-hashed batch protocol."""
from __future__ import annotations

import dataclasses

import pytest

from acrpq.dashboard.qpu_protocol import BatchProtocol, JobSpec, ProtocolError


def _job(**over):
    base = dict(instance="CP_3", objective_id="maneuver_count_v1", n_theta=3, n_q=1,
                n_qubits=9, qubo_sha256="sha256:" + "a" * 64, instance_sha256="sha256:" + "b" * 64)
    base.update(over)
    return JobSpec(**base)


def _proto(**over):
    base = dict(qaoa_protocol_id="aer_default_p1_v1", qaoa_reps=1,
                angle_strategy="seeded_random_uniform", optimizer="COBYLA", shots=1024,
                transpiler_seed=1, backend_requested="ibm_x", optimization_level=1, max_jobs=1,
                max_quantum_seconds=60.0, total_shot_budget=100000, read_retry_policy="bounded-3",
                reconciliation_policy="poll-terminal", max_wait_seconds=600.0, endianness="qubo",
                qiskit_version="2.4.1", jobs=(_job(),))
    base.update(over)
    return BatchProtocol(**base)


def test_hash_is_canonical_and_no_secret_or_path():
    p = _proto()
    h = p.protocol_hash()
    assert h.startswith("sha256:") and len(h) == 71
    assert h == _proto().protocol_hash()                 # deterministic
    blob = p.to_canonical()
    assert "token" not in str(blob).lower() and "/Users/" not in str(blob)


@pytest.mark.parametrize("field,val", [
    ("qaoa_protocol_id", "other"),
    ("qaoa_reps", 2),
    ("angle_strategy", "zeros"),
    ("optimizer", "NELDER_MEAD"),
    ("shots", 2048),
    ("transpiler_seed", 2),
    ("backend_requested", "ibm_y"),
    ("optimization_level", 3),
    ("max_jobs", 2),
    ("max_quantum_seconds", 120.0),
    ("total_shot_budget", 4096),
    ("read_retry_policy", "bounded-5"),
    ("reconciliation_policy", "poll-slow"),
    ("max_wait_seconds", 300.0),
    ("endianness", "qiskit"),
    ("qiskit_version", "2.5.0"),
])
def test_every_batch_field_changes_the_hash(field, val):
    assert _proto().protocol_hash() != _proto(**{field: val}).protocol_hash()


@pytest.mark.parametrize("field,val", [
    ("instance", "CP_4"), ("objective_id", "quadratic_control_cost_v1"),
    ("n_theta", 5), ("n_q", 2), ("n_qubits", 12),
    ("qubo_sha256", "sha256:" + "c" * 64), ("instance_sha256", "sha256:" + "d" * 64),
    ("logical_circuit_sha256", "sha256:" + "e" * 64), ("isa_sha256", "sha256:" + "f" * 64),
])
def test_every_job_field_changes_the_hash(field, val):
    assert _proto().protocol_hash() != _proto(jobs=(_job(**{field: val}),)).protocol_hash()


def test_optimizer_options_and_deps_change_the_hash():
    assert _proto().protocol_hash() != _proto(optimizer_options={"rhobeg": 0.5}).protocol_hash()
    assert _proto().protocol_hash() != _proto(dependency_versions={"numpy": "2.4.6"}).protocol_hash()


def test_effective_backend_is_part_of_identity():
    p = _proto()
    assert p.backend_effective is None
    p2 = p.with_effective_backend("ibm_real_123")
    assert p2.backend_effective == "ibm_real_123"
    assert p.protocol_hash() != p2.protocol_hash()       # a distinct identity, on purpose


def test_frozen_and_hash_order_independent():
    p = _proto()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.shots = 1  # type: ignore[misc]
    # dict field ordering must not affect the hash
    a = _proto(optimizer_options={"rhobeg": 0.5, "tol": 1e-6})
    b = _proto(optimizer_options={"tol": 1e-6, "rhobeg": 0.5})
    assert a.protocol_hash() == b.protocol_hash()


@pytest.mark.parametrize("bad", [
    dict(qaoa_reps=0), dict(shots=0), dict(max_jobs=0), dict(transpiler_seed=-1),
    dict(optimization_level=4), dict(max_quantum_seconds=0.0),
    dict(max_quantum_seconds=float("inf")), dict(max_wait_seconds=-1.0),
    dict(angle_strategy="nope"), dict(optimizer="ADAM"), dict(endianness="big"),
    dict(schema_version="wrong"), dict(jobs=()), dict(total_shot_budget=1),
    dict(qaoa_reps=True), dict(shots=1.5),
])
def test_invalid_protocols_are_refused(bad):
    with pytest.raises(ProtocolError):
        _proto(**bad)


def test_non_finite_optimizer_option_refused():
    with pytest.raises(ProtocolError):
        _proto(optimizer_options={"rhobeg": float("nan")})
