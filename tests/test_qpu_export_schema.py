"""Phase 5.6 — versioned export schema, JSON/CSV parity, integrity validator (no IBM)."""
from __future__ import annotations

import pytest

from acrpq.dashboard import qpu_export_schema as X


def _row(**over):
    r = {c: None for c in X.EXPORT_COLUMNS}
    r.update(dict(
        batch_id="b1", batch_config_hash="sha256:" + "a" * 64, job_config_hash="sha256:" + "b" * 64,
        local_run_id="run-xyz", provider_job_id_hash="sha256:" + "c" * 64, protocol_id="p1",
        instance="CP_3", family="CP", objective_id="maneuver_count_v1", n_theta=3, n_q=1,
        n_qubits=9, qubo_sha256="sha256:" + "d" * 64, logical_circuit_sha256="sha256:" + "e" * 64,
        isa_sha256="sha256:" + "f" * 64, backend_requested="ibm_x", backend_effective="ibm_x",
        qaoa_reps=1, angle_strategy="seeded_random_uniform", optimizer="COBYLA", shots=1024,
        transpiler_seed=1, optimization_level=1, final_state="completed",
        prepared_utc="2026-01-01T00:00:00Z", submitted_utc="2026-01-01T00:00:01Z",
        provider_start_utc="", provider_end_utc="", queue_seconds=0.0, provider_seconds=1.0,
        qpu_seconds=0.5, wall_seconds=2.0, best_raw_bitstring="100", modal_bitstring="100",
        best_onehot_valid_bitstring="100", best_feasible_bitstring="100", onehot_valid_mass=1.0,
        feasibility_rate=1.0, primary_cost=2.0, qubo_energy=2.0, residual_conflicts=0,
        delta_vs_reference=0.0, comparability="comparable", warnings="", data_source="synthetic_offline",
        official=False))
    r.update(over)
    return r


def _doc(rows=None, **prov_over):
    prov = dict(source_commit="abc123", scientific_code_dirty=False, repository_dirty=False,
                versions={"qiskit": "2.4.1"}, data_source="synthetic_offline",
                status=X.OFFLINE_STATUS, official=False)
    prov.update(prov_over)
    doc = {"schema": X.EXPORT_SCHEMA, "rows": rows if rows is not None else [_row()],
           "provenance": prov}
    doc["export_sha256"] = X.canonical_export_sha256(doc)
    return doc


# --- parity + hash --------------------------------------------------------
def test_json_csv_parity_roundtrip():
    rows = [_row(), _row(instance="CP_4", n_qubits=12)]
    csv_text = X.rows_to_csv(rows)
    X.assert_json_csv_parity(rows, csv_text)             # no raise


def test_parity_fails_on_divergence():
    rows = [_row()]
    csv_text = X.rows_to_csv(rows).replace("completed", "failed")
    with pytest.raises(X.ExportError, match="final_state"):
        X.assert_json_csv_parity(rows, csv_text)


def test_hash_excludes_wall_times_and_ids():
    a = _doc()
    # change only volatile columns -> hash unchanged
    b = _doc(rows=[_row(wall_seconds=999.0, local_run_id="other", queue_seconds=5.0)])
    assert X.canonical_export_sha256(a) == X.canonical_export_sha256(b)
    # change a scientific column -> hash changes
    c = _doc(rows=[_row(primary_cost=3.0)])
    assert X.canonical_export_sha256(a) != X.canonical_export_sha256(c)


# --- validator ------------------------------------------------------------
def test_valid_export_passes():
    X.validate_export(_doc())                            # no raise


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="wrong"),
    lambda d: d["rows"][0].pop("primary_cost"),
    lambda d: d["rows"][0].update(unknown_col=1),
    lambda d: d["rows"][0].update(primary_cost=float("inf")),
    lambda d: d["provenance"].pop("source_commit"),
    lambda d: d["provenance"].update(official=True),                 # synthetic marked official
    lambda d: d["provenance"].update(status="quantum result"),      # dishonest label
    lambda d: d["rows"][0].update(instance="/Users/fabio/secret"),  # local path/PII
    lambda d: d.update(export_sha256="sha256:" + "0" * 64),          # tampered hash
])
def test_validator_fails_closed(mutate):
    doc = _doc()
    mutate(doc)
    with pytest.raises(X.ExportError):
        X.validate_export(doc)


def test_row_schema_rejects_non_scalar():
    with pytest.raises(X.ExportError, match="JSON scalar"):
        X.assert_row_schema(_row(warnings=["a", "b"]))


# --- red-team #2: broadened leak scan -------------------------------------
@pytest.mark.parametrize("leak", [
    "/home/alice/secret", "/var/run/thing", "/private/tmp/secret/run",
    "C:\\Users\\bob\\key", "ghp_ABCDEFGHIJKLMNOP0000", "Bearer abc.def",
    "-----BEGIN RSA PRIVATE KEY-----", "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY",
])
def test_validator_catches_paths_and_secrets(leak):
    doc = _doc(rows=[_row(warnings=leak)])
    doc["export_sha256"] = X.canonical_export_sha256(doc)
    with pytest.raises(X.ExportError, match="local absolute path|secret-like"):
        X.validate_export(doc)


def test_legit_content_with_slash_not_flagged():
    # "n/a" and hashes contain no absolute path -> must pass
    doc = _doc(rows=[_row(warnings="n/a", comparability="comparable")])
    doc["export_sha256"] = X.canonical_export_sha256(doc)
    X.validate_export(doc)                                # no raise


# --- Codex item 3: strict/scientific validation ---------------------------
def _official(**prov_over):
    prov = dict(source_commit="abc", scientific_code_dirty=False, repository_dirty=False,
                versions={"qiskit": "2.4.1"}, data_source="ibm_quantum_hardware", official=True)
    prov.update(prov_over)
    doc = {"schema": X.EXPORT_SCHEMA, "rows": [_row(data_source="ibm_quantum_hardware")],
           "provenance": prov}
    doc["export_sha256"] = X.canonical_export_sha256(doc)
    return doc


def test_strict_requires_wellformed_hash():
    doc = _doc()
    doc.pop("export_sha256")
    with pytest.raises(X.ExportError, match="strict: export_sha256 is missing"):
        X.validate_export(doc, strict=True)
    doc["export_sha256"] = "not-a-hash"
    with pytest.raises(X.ExportError, match="malformed"):
        X.validate_export(doc, strict=True)
    good = _doc()                                        # has a valid hash
    X.validate_export(good, strict=True)                 # no raise


def _rehash(d):
    d["export_sha256"] = X.canonical_export_sha256(d)
    return d


def test_official_export_sanity():
    empty = _official()
    empty["rows"] = []
    with pytest.raises(X.ExportError, match="must not be empty"):
        X.validate_export(_rehash(empty))
    with pytest.raises(X.ExportError, match="dirty provenance"):
        X.validate_export(_rehash(_official(scientific_code_dirty=True)))
    with pytest.raises(X.ExportError, match="dirty provenance"):
        X.validate_export(_rehash(_official(repository_dirty=True)))
    X.validate_export(_official())                       # a clean, non-empty official export ok


def test_hash_binds_repository_dirty_blocking_launder():
    """Red-team #1: repository_dirty is hashed, so a dirty official export cannot be laundered
    to clean by flipping the flag while keeping the stored hash."""
    clean = _official()                                  # repository_dirty=False, valid hash
    dirty = _official(repository_dirty=True)
    dirty["export_sha256"] = X.canonical_export_sha256(dirty)
    assert clean["export_sha256"] != dirty["export_sha256"]   # the flag is bound into the hash
    with pytest.raises(X.ExportError, match="dirty provenance"):
        X.validate_export(dirty)                         # honest dirty-official is refused
    dirty["provenance"]["repository_dirty"] = False       # LAUNDER: flip clean, keep dirty hash
    with pytest.raises(X.ExportError, match="does not match"):
        X.validate_export(dirty)                         # hash now mismatches -> refused


def test_unknown_top_level_field_refused():
    d = _doc()
    d["surprise"] = 1
    d["export_sha256"] = X.canonical_export_sha256(d)
    with pytest.raises(X.ExportError, match="unknown top-level"):
        X.validate_export(d)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(scientific_code_dirty="no"),
    lambda p: p.update(repository_dirty=1),
    lambda p: p.update(versions=[]),
    lambda p: p.update(source_commit=""),
])
def test_invalid_provenance_types_refused(mutate):
    d = _doc()
    mutate(d["provenance"])
    d["export_sha256"] = X.canonical_export_sha256(d)
    with pytest.raises(X.ExportError):
        X.validate_export(d)


# --- red-team #3: honest-offline flag -------------------------------------
def test_require_synthetic_offline_forces_honest_label():
    doc = _doc()
    doc["provenance"]["data_source"] = "ibm_quantum_hardware"    # a mislabel
    doc["provenance"]["official"] = True
    doc["export_sha256"] = X.canonical_export_sha256(doc)
    # default validate still passes structurally (provenance is unprovable)...
    X.validate_export(doc)
    # ...but an OFFLINE export forced to declare synthetic_offline is refused
    with pytest.raises(X.ExportError, match="synthetic_offline"):
        X.validate_export(doc, require_synthetic_offline=True)
