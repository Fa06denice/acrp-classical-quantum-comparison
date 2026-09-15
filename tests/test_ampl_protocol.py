"""Frozen AMPL/Gurobi protocol v3 + FAIL-CLOSED anchoring, measurement proof, guard."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from acrpq.classical.ampl_protocol import (
    AMPL_GUROBI_SUPERVISED_V1 as V1,
)
from acrpq.classical.ampl_protocol import (
    AMPL_GUROBI_SUPERVISED_V2 as V2,
)
from acrpq.classical.ampl_protocol import (
    AMPL_GUROBI_SUPERVISED_V3 as V3,
)
from acrpq.classical.ampl_protocol import (
    AMPL_GUROBI_V3_TIMEOUT_RECOVERY as RECOVERY,
)
from acrpq.classical.ampl_protocol import (
    AMPL_GUROBI_ORIGINAL_DEFAULTS_V1 as ORIGINAL_DEFAULTS,
)
from acrpq.classical.ampl_protocol import (
    ANCHOR_QUALIFICATION,
    OFFICIAL_PROTOCOL,
    AnchorRejected,
    GridMismatchError,
    accept_official_anchor,
    assert_same_grid,
    discrete_optimum_proven,
    evaluate_anchor,
    grid_key,
    verify_invalidation_sidecar,
)

_H = {"model_sha256": "m", "preprocessing_sha256": "p", "data_sha256": "d"}
_ECHO = {"pre:funcnonlinear": "1", "alg:feastol": "1e-09", "mip:gap": "1e-06",
         "sol:chk:feastol": "1e-08"}
_PROV = {
    **_H, "ampl_version": "AMPL 20260809", "solver_version": "13.0.2",
    "solver_requested": "gurobi", "solver_effective": "gurobi",
    "solver_options": V3.solver_options, "solver_options_echo": _ECHO,
    "protocol_id": V3.protocol_id,
}
_MEAS = {
    "performed": True, "status": "optimal", "solution_matches_primary": True,
    "max_abs_delta_q": 0.0, "max_abs_delta_theta": 0.0,
    "solution_match_tolerance": 1e-12,
    "solver_options_requested": V3.mp_measure_options,
    "measured_max_abs_mp_residual": 9e-9,
    "mp_violation_table": {"failed": False, "max_abs_residual": 9e-9},
    **_H,
}
_EXTRA = {"mip_rel_gap": 9.7e-7, "provider_objective": 6.25e-4,
          "rescored_objective": 6.25e-4, "integrity_ok": 1.0}


def _anchor(*, extra=None, prov=None, meas=None, **over):
    base = dict(
        status="optimal", extra={**_EXTRA, **(extra or {})},
        provenance={**_PROV, **(prov or {})},
        feasible_common_numeric=True, n=4,
        q=(1.0, 1.0, 1.0, 1.0), theta=(0.0, 0.0, 0.0, 0.0),
        scientific_code_dirty=False, measurement={**_MEAS, **(meas or {})}, protocol=V3,
    )
    base.update(over)
    return evaluate_anchor(**base)


# --- config ---------------------------------------------------------------- #
def test_v3_official_two_thresholds_and_measure_options() -> None:
    assert OFFICIAL_PROTOCOL is V3
    assert V3.max_rel_optimality_gap == 1e-6 and V3.max_abs_mp_residual == 1e-8
    assert V3.mp_measure_options and "sol:chk:fail" not in V3.mp_measure_options
    assert V3.measurement_match_tol == 1e-12


def test_original_defaults_changes_only_solver_and_has_no_wrapper_limit() -> None:
    assert ORIGINAL_DEFAULTS.solver == "gurobi"
    assert ORIGINAL_DEFAULTS.solver_options == ""
    assert ORIGINAL_DEFAULTS.timeout_s is None
    assert ORIGINAL_DEFAULTS.mp_check_feastol is None
    assert ORIGINAL_DEFAULTS.mp_measure_options is None
    solver = ORIGINAL_DEFAULTS.build_solver()
    assert solver.solver == "gurobi"
    assert solver.solver_options == ""
    assert solver.timeout_s is None


def test_qualification_states_both_thresholds_never_exact() -> None:
    low = ANCHOR_QUALIFICATION.lower()
    assert "exact" not in low and "positive" not in low and "margin" not in low
    assert "1e-6" in low and "1e-8" in low


# --- happy path ------------------------------------------------------------ #
def test_valid_v3_leg_with_measurement_proof_anchors() -> None:
    a = _anchor()
    assert a["is_anchor"] is True
    assert all(a["criteria"].values())


# --- FAIL-CLOSED measurement proof (A) ------------------------------------- #
def test_absent_measurement_never_anchors() -> None:
    a = evaluate_anchor(
        status="optimal", extra=_EXTRA, provenance=_PROV, feasible_common_numeric=True,
        n=4, q=(1.0,) * 4, theta=(0.0,) * 4, scientific_code_dirty=False,
        measurement=None, protocol=V3)
    assert a["criteria"]["measurement_performed"] is False
    assert a["criteria"]["mp_residual_present_and_ok"] is False
    assert a["is_anchor"] is False


def test_measurement_not_reproducing_primary_never_anchors() -> None:
    assert _anchor(meas={"solution_matches_primary": False})[
        "criteria"]["measurement_reproduces_primary"] is False
    assert _anchor(meas={"max_abs_delta_q": 1e-9})[  # > 1e-12 tol
        "criteria"]["measurement_reproduces_primary"] is False
    assert _anchor(meas={"max_abs_delta_theta": None})[
        "criteria"]["measurement_reproduces_primary"] is False


def test_measurement_options_must_conform() -> None:
    assert _anchor(meas={"solver_options_requested": "pre:funcnonlinear=1"})[
        "criteria"]["measurement_options_conform"] is False


def test_measurement_hashes_must_match_primary() -> None:
    assert _anchor(meas={"model_sha256": "different"})[
        "criteria"]["measurement_hashes_match_primary"] is False
    assert _anchor(meas={"data_sha256": None})[
        "criteria"]["measurement_hashes_match_primary"] is False


def test_residual_must_be_present_finite_and_le_1e_8() -> None:
    assert _anchor(meas={"measured_max_abs_mp_residual": None})[
        "criteria"]["mp_residual_present_and_ok"] is False
    assert _anchor(meas={"measured_max_abs_mp_residual": 2e-8})[
        "criteria"]["mp_residual_present_and_ok"] is False
    assert _anchor(meas={"measured_max_abs_mp_residual": math.inf})[
        "criteria"]["mp_residual_present_and_ok"] is False
    assert _anchor(meas={"measured_max_abs_mp_residual": 1e-8})[
        "criteria"]["mp_residual_present_and_ok"] is True
    assert _anchor(meas={"mp_violation_table": {"unparsed": True},
                         "measured_max_abs_mp_residual": None})[
        "criteria"]["mp_residual_present_and_ok"] is False


# --- other fail-closed criteria -------------------------------------------- #
def test_gap_uses_its_own_threshold() -> None:
    assert _anchor(extra={"mip_rel_gap": 2.4e-5})["criteria"]["rel_gap_le_threshold"] is False


def test_nan_primal_options_provenance_git_all_fail_closed() -> None:
    assert _anchor(q=(1.0, math.inf, 1.0, 1.0))["criteria"]["primal_complete_finite"] is False
    assert _anchor(prov={"solver_options": "x"})[
        "criteria"]["effective_options_match_protocol"] is False
    assert _anchor(prov={"preprocessing_sha256": None})[
        "criteria"]["provenance_complete"] is False
    assert _anchor(prov={"solver_effective": "couenne"})[
        "criteria"]["solver_identity_gurobi"] is False
    assert _anchor(prov={"protocol_id": None})["criteria"]["protocol_id_present"] is False
    assert _anchor(scientific_code_dirty=True)["criteria"]["scientific_code_clean"] is False
    assert _anchor(scientific_code_dirty=None)["criteria"]["scientific_code_clean"] is False


def test_v1_v2_never_anchor() -> None:
    a1 = evaluate_anchor(status="optimal", extra=_EXTRA, provenance=_PROV,
                         feasible_common_numeric=True, n=4, q=(1.0,) * 4, theta=(0.0,) * 4,
                         scientific_code_dirty=False, measurement=_MEAS, protocol=V1)
    assert a1["is_anchor"] is False   # v1 has no MP residual gate
    a2 = evaluate_anchor(status="optimal", extra=_EXTRA,
                         provenance={**_PROV, "solver_options": V2.solver_options},
                         feasible_common_numeric=True, n=4, q=(1.0,) * 4, theta=(0.0,) * 4,
                         scientific_code_dirty=False,
                         measurement={**_MEAS, "solver_options_requested": V2.mp_measure_options},
                         protocol=V2)
    # v2's residual threshold is 1e-9; the measured 9e-9 exceeds it
    assert a2["criteria"]["mp_residual_present_and_ok"] is False


# --- machine-readable invalidation guard (B) ------------------------------- #
def _write(dirpath, name, obj):
    p = dirpath / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _v3_artifact():
    return {"protocol_id": V3.protocol_id, "is_official_anchor": True,
            "anchor_criteria": {"a": True, "b": True}}


def test_guard_accepts_official_v3(tmp_path):
    d = tmp_path / V3.protocol_id
    d.mkdir()
    assert accept_official_anchor(_v3_artifact(), d) is True


def test_guard_refuses_v1_even_if_stored_flag_true(tmp_path):
    d = tmp_path / V1.protocol_id
    d.mkdir()
    art = {"protocol_id": V1.protocol_id, "is_official_anchor": True,
           "anchor_criteria": {"x": True}}
    with pytest.raises(AnchorRejected):
        accept_official_anchor(art, d)


def test_guard_refuses_v2_and_unknown_protocol(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    with pytest.raises(AnchorRejected):
        accept_official_anchor({"protocol_id": V2.protocol_id, "anchor_criteria": {"x": True}}, d)
    with pytest.raises(AnchorRejected):
        accept_official_anchor({"protocol_id": "unknown", "anchor_criteria": {"x": True}}, d)


def test_guard_refuses_when_criteria_not_all_true(tmp_path):
    d = tmp_path / V3.protocol_id
    d.mkdir()
    bad = {"protocol_id": V3.protocol_id, "is_official_anchor": True,
           "anchor_criteria": {"a": True, "b": False}}
    with pytest.raises(AnchorRejected):
        accept_official_anchor(bad, d)


def test_invalidation_sidecar_refuses_invalidated_protocol(tmp_path):
    d = tmp_path / V3.protocol_id
    d.mkdir()
    art = _v3_artifact()
    art_path = _write(d, "CP_4.json", art)
    h = "sha256:" + hashlib.sha256(art_path.read_bytes()).hexdigest()
    _write(d, "INVALIDATED.json", {"protocol_id": V3.protocol_id, "valid_for_anchoring": False,
                                   "invalidation_reason": "test",
                                   "affected_artifact_hashes": {"CP_4.json": h}})
    with pytest.raises(AnchorRejected):
        accept_official_anchor(art, d)


def test_tampered_sidecar_hash_is_refused(tmp_path):
    d = tmp_path / V3.protocol_id
    d.mkdir()
    _write(d, "CP_4.json", _v3_artifact())
    _write(d, "INVALIDATED.json", {"protocol_id": V3.protocol_id, "valid_for_anchoring": False,
                                   "affected_artifact_hashes": {"CP_4.json": "sha256:deadbeef"}})
    with pytest.raises(AnchorRejected):
        verify_invalidation_sidecar(d)


def test_no_sidecar_returns_none(tmp_path):
    d = tmp_path / V3.protocol_id
    d.mkdir()
    assert verify_invalidation_sidecar(d) is None


# --- timeout-recovery protocol is EXPLICITLY non-official ------------------- #
def test_recovery_protocol_can_never_anchor(tmp_path):
    assert RECOVERY.protocol_id == "ampl_gurobi_v3_timeout_recovery"
    assert RECOVERY.max_abs_mp_residual is None          # measurement criteria always fail
    assert "sol:chk:fail" not in RECOVERY.solver_options  # a limit incumbent is not rejected
    assert "lim:time=110" in RECOVERY.solver_options
    # even with an otherwise-perfect record, this protocol never yields an anchor
    a = evaluate_anchor(
        status="optimal", extra=_EXTRA, provenance={**_PROV, "solver_options": RECOVERY.solver_options},
        feasible_common_numeric=True, n=4, q=(1.0,) * 4, theta=(0.0,) * 4,
        scientific_code_dirty=False, measurement=_MEAS, protocol=RECOVERY)
    assert a["is_anchor"] is False
    assert a["criteria"]["mp_residual_present_and_ok"] is False


def test_recovery_artifact_is_refused_by_official_guard(tmp_path):
    d = tmp_path / RECOVERY.protocol_id
    d.mkdir()
    art = {"protocol_id": RECOVERY.protocol_id, "is_official_anchor": False,
           "anchor_criteria": {"x": True}}
    with pytest.raises(AnchorRejected):
        accept_official_anchor(art, d)


# --- grid + discrete proof ------------------------------------------------- #
def test_grid_mismatch_is_refused() -> None:
    assert grid_key(3, 1) == "K3_q1"
    assert_same_grid("K3_q1", "K3_q1")
    with pytest.raises(GridMismatchError):
        assert_same_grid("K5_q1", "K3_q1")


def test_discrete_optimum_proven_only_when_exhaustive_and_optimal() -> None:
    cap = 10_000_000
    assert discrete_optimum_proven(status="optimal", search_space_size=3125, search_cap=cap)
    assert not discrete_optimum_proven(status="optimal", search_space_size=cap + 1, search_cap=cap)
    for s in ("timeout", "feasible", "infeasible", "unknown", "error"):
        assert not discrete_optimum_proven(status=s, search_space_size=100, search_cap=cap)
