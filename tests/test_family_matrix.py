"""Phase 3A — applicability-matrix tests (metadata only, no solves)."""

from __future__ import annotations

import pytest

from acrpq.benchmark.family_matrix import (
    assert_json_csv_parity,
    build_matrix,
    canonical_matrix_sha256,
    cell,
    rows_to_csv,
)
from acrpq.io.loader import InstanceLoader

_METHODS = {"reference", "qubo_exact", "qubo_annealing", "dwave_sa", "qaoa_aer", "gurobi_milp"}


@pytest.fixture(scope="module")
def ld():
    return InstanceLoader()


def test_cell_metadata_is_correct_for_cp4(ld):
    c = cell(ld, "CP_4", n_theta=3)
    assert c.family == "CP" and c.n_aircraft == 4 and c.grid_k == 3
    assert c.n_vars == 12                       # n * K
    assert c.n_onehot_terms == 4 * (3 * 2 // 2)  # n * C(K,2) = 12
    assert c.initial_conflicts == 6 and c.n_pairs == 6
    assert c.conflict_density == 1.0
    assert c.search_space == "81"               # 3**4
    assert set(c.methods) == _METHODS
    # small instance: exact enumeration + QAOA both apply
    assert c.methods["qubo_exact"]["applicable"] is True
    assert c.methods["reference"]["resolution"] == "exact"


def test_large_instance_drops_exact_and_qaoa_but_keeps_scalable_methods(ld):
    name = sorted(ld.list_names("RCP"))
    big = next(n for n in name if ld.load(n).n >= 40)
    c = cell(ld, big, n_theta=3)
    assert c.methods["qubo_exact"]["applicable"] is False   # K^n beyond the cap
    assert c.methods["qaoa_aer"]["applicable"] is False      # exceeds RAM ceiling
    # methods that scale past the enumeration/RAM caps still apply
    assert c.methods["reference"]["applicable"] is True
    assert c.methods["qubo_annealing"]["applicable"] is True
    assert c.methods["gurobi_milp"]["applicable"] is True


def test_cell_is_deterministic(ld):
    from dataclasses import asdict
    assert asdict(cell(ld, "CP_5", n_theta=3)) == asdict(cell(ld, "CP_5", n_theta=3))


def test_build_matrix_covers_and_logs_caps(ld):
    rows, coverage = build_matrix(
        ld, families=["CP"], per_family_limit={"CP": 4},
        k_sensitivity=(5,), sensitivity_max_aircraft=8)
    cov = coverage[0]
    assert cov["family"] == "CP" and cov["covered"] == 4 and cov["dropped"] == 14
    # 4 instances at K3 + K5 for those with n<=8 (CP_3..CP_6 all qualify) = 4 + 4
    assert len(rows) == 8
    assert {r["grid_k"] for r in rows} == {3, 5}
    assert all(set(r["methods"]) == _METHODS for r in rows)


def test_json_csv_parity_holds(ld):
    rows, _ = build_matrix(ld, families=["CP"], per_family_limit={"CP": 4},
                           k_sensitivity=(5,), sensitivity_max_aircraft=8)
    csv_text = rows_to_csv(rows)
    # one header + one line per row
    assert len(csv_text.strip().splitlines()) == len(rows) + 1
    assert_json_csv_parity(rows, csv_text)  # does not raise


def test_json_csv_parity_detects_tampering(ld):
    rows, _ = build_matrix(ld, families=["CP"], per_family_limit={"CP": 2})
    csv_text = rows_to_csv(rows)
    rows[0]["methods"]["qubo_exact"]["applicable"] = not rows[0]["methods"]["qubo_exact"]["applicable"]
    with pytest.raises(ValueError, match="parity mismatch"):
        assert_json_csv_parity(rows, csv_text)


def test_canonical_hash_ignores_sha_field_and_is_stable(ld):
    rows, coverage = build_matrix(ld, families=["CP"], per_family_limit={"CP": 3})
    m = {"schema": "acrpq-family-matrix/1", "rows": rows, "coverage": coverage}
    h1 = canonical_matrix_sha256(m)
    m["matrix_sha256"] = h1                       # embedding the hash must not change it
    m["matrix_sha256_note"] = "note"
    assert canonical_matrix_sha256(m) == h1
    # any content change flips the hash
    m["rows"][0]["initial_conflicts"] += 1
    assert canonical_matrix_sha256(m) != h1
