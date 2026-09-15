"""The supervised AMPL/Gurobi protocol — a single, frozen, versioned config.

The thesis director's benchmark solver is Gurobi. After the CP_4 counter-review,
the validated official configuration adds an ENABLED strict MP solution check so a
leg only anchors when the solver's own constraint residuals are actually verified
(``ampl_gurobi_supervised_v2``). ``v1`` is kept for provenance but can never anchor
(it ran no MP check). Config lives in ONE place, not as a free string scattered
across scripts. Experiments may pass an explicit override; the runner NEVER falls
back to a different solver or option set silently.

Anchor decisions are **fail-closed**: an ABSENT measurement (no residual, no MP
check, missing provenance/options, dirty tree) is a failure, never a pass. A leg is
described only as a *numerical optimum certified to a relative gap <= 1e-6 and
accepted by the common geometry under its documented tolerance* — never "exact
optimum", "exact feasibility" or "positive operational margin". The historical model
optimises to the separation boundary and imposes no positive margin; a ``d + epsilon``
robust study would be a different problem.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, TypeGuard

ANCHOR_QUALIFICATION = (
    "numerical optimum certified to a relative gap <= 1e-6, with maximum absolute MP "
    "residual <= 1e-8, and accepted by the common geometry under its documented tolerance"
)


@dataclass(frozen=True)
class AmplProtocol:
    """A frozen, named AMPL/Gurobi configuration + its anchor tolerances."""

    protocol_id: str
    solver: str
    solver_options: str
    timeout_s: float | None
    pi_literal: float               # historical NC_ACRP.mod constant (documented, not math.pi)
    scorer_tol_msq: float           # common-geometry conflict tolerance (on squared sep)
    objective_abs_tol: float        # concordance: absolute term
    objective_rel_tol: float        # concordance: relative term
    max_rel_optimality_gap: float   # official relative optimality-gap threshold (distinct field)
    max_abs_mp_residual: float | None  # official absolute MP-residual threshold; None => no gate
    mp_check_feastol: float | None  # sol:chk:feastol value in the options (fatal check tol)
    mp_measure_options: str | None  # warn-mode options to MEASURE the residual (no sol:chk:fail)
    anchor_kind: str
    measurement_match_tol: float = 1e-12  # max |Δq|,|Δθ| between primary and measurement

    def build_solver(self, *, timeout_s: float | None = None,
                     solver_options: str | None = None):
        """Construct the :class:`OriginalAMPLSolver` for this protocol.

        ``solver_options``/``timeout_s`` may be overridden EXPLICITLY (experiments);
        omitting them uses the frozen protocol values. There is no silent fallback.
        The concordance abs tolerance seeds the runner's integrity check.
        """
        from .original_ampl import OriginalAMPLSolver

        return OriginalAMPLSolver(
            solver=self.solver,
            solver_options=self.solver_options if solver_options is None else solver_options,
            timeout_s=self.timeout_s if timeout_s is None else timeout_s,
            objective_tolerance=self.objective_abs_tol,
        )

    def objective_concordant(self, provider: float | None, rescored: float | None) -> bool:
        """Documented concordance: abs_delta <= abs_tol + rel_tol*max(|prov|,|resc|)."""
        if provider is None or rescored is None:
            return False
        if not (math.isfinite(provider) and math.isfinite(rescored)):
            return False
        abs_delta = abs(provider - rescored)
        tol = self.objective_abs_tol + self.objective_rel_tol * max(abs(provider), abs(rescored))
        return abs_delta <= tol

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "anchor_qualification": ANCHOR_QUALIFICATION}


# v1 — NO MP solution check ran (max_abs_mp_residual=None => can never anchor).
AMPL_GUROBI_SUPERVISED_V1 = AmplProtocol(
    protocol_id="ampl_gurobi_supervised_v1",
    solver="gurobi",
    solver_options="pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6",
    timeout_s=120.0, pi_literal=3.141592, scorer_tol_msq=1e-9,
    objective_abs_tol=1e-6, objective_rel_tol=1e-6,
    max_rel_optimality_gap=1e-6, max_abs_mp_residual=None,
    mp_check_feastol=None, mp_measure_options=None,
    anchor_kind="numerical_optimum_gap_1e-6",
)

# v2 — strict MP check at 1e-9 (FATAL); the gap-1e-6 incumbent residual ~7-9e-9
# exceeds 1e-9, so the check fails and v2 never anchors (documented).
AMPL_GUROBI_SUPERVISED_V2 = AmplProtocol(
    protocol_id="ampl_gurobi_supervised_v2",
    solver="gurobi",
    solver_options=("pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 "
                    "sol:chk:feastol=1e-9 sol:chk:fail"),
    timeout_s=120.0, pi_literal=3.141592, scorer_tol_msq=1e-9,
    objective_abs_tol=1e-6, objective_rel_tol=1e-6,
    max_rel_optimality_gap=1e-6, max_abs_mp_residual=1e-9,
    mp_check_feastol=1e-9,
    mp_measure_options="pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 sol:chk:feastol=1e-30",
    anchor_kind="numerical_optimum_gap_1e-6",
)

# v3 — OFFICIAL (reviewer-reconciled): keep gap <= 1e-6 and alg:feastol=1e-9, adopt a
# fatal MP check at 1e-8 (the measured residual scale is 7-9e-9). Two distinct
# thresholds: max_rel_optimality_gap=1e-6, max_abs_mp_residual=1e-8. The primary solve
# CERTIFIES residual <= 1e-8 (fatal sol:chk:fail); a warn-mode measurement run
# (mp_measure_options) reports the actual residual on the identical solution.
AMPL_GUROBI_SUPERVISED_V3 = AmplProtocol(
    protocol_id="ampl_gurobi_supervised_v3",
    solver="gurobi",
    solver_options=("pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 "
                    "sol:chk:feastol=1e-8 sol:chk:fail"),
    timeout_s=120.0, pi_literal=3.141592, scorer_tol_msq=1e-9,
    objective_abs_tol=1e-6, objective_rel_tol=1e-6,
    max_rel_optimality_gap=1e-6, max_abs_mp_residual=1e-8,
    mp_check_feastol=1e-8,
    mp_measure_options="pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 sol:chk:feastol=1e-30",
    anchor_kind="numerical_optimum_gap_1e-6_mp_1e-8",
)

# Operationally faithful counterpart of the historical ``NC_ACRP.run``: only
# the solver changes from Couenne to the supervisor-requested Gurobi. AMPL sends
# no solver options and the Python wrapper imposes no wall-clock deadline, so
# Gurobi uses its own defaults. This protocol makes no custom 1e-6-gap or MP-
# residual certification claim.
AMPL_GUROBI_ORIGINAL_DEFAULTS_V1 = AmplProtocol(
    protocol_id="ampl_gurobi_original_defaults_v1",
    solver="gurobi",
    solver_options="",
    timeout_s=None,
    pi_literal=3.141592,
    scorer_tol_msq=1e-9,
    objective_abs_tol=1e-6,
    objective_rel_tol=1e-6,
    max_rel_optimality_gap=math.inf,
    max_abs_mp_residual=None,
    mp_check_feastol=None,
    mp_measure_options=None,
    anchor_kind="solver_declared_optimum_gurobi_defaults",
)

OFFICIAL_PROTOCOL = AMPL_GUROBI_SUPERVISED_V3

# EXPLICITLY NON-OFFICIAL experimental protocol: recover the CP_8..CP_20 incumbents
# that the external timeout otherwise kills before the markers are written. It adds an
# INTERNAL solver time limit (lim:time=110) BELOW the external timeout (125 s) so Gurobi
# terminates gracefully and prints its incumbent, and drops the fatal ``sol:chk:fail``
# (a limit-time incumbent need not pass the 1e-8 check) — a warn-mode ``sol:chk`` still
# REPORTS the residual without rejecting. It can NEVER produce an official anchor:
# ``max_abs_mp_residual=None`` (measurement criteria all fail) AND its ``protocol_id`` is
# not the official one, so ``accept_official_anchor`` refuses it.
AMPL_GUROBI_V3_TIMEOUT_RECOVERY = AmplProtocol(
    protocol_id="ampl_gurobi_v3_timeout_recovery",
    solver="gurobi",
    solver_options=("pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 "
                    "lim:time=110 sol:chk:feastol=1e-30"),
    timeout_s=125.0, pi_literal=3.141592, scorer_tol_msq=1e-9,
    objective_abs_tol=1e-6, objective_rel_tol=1e-6,
    max_rel_optimality_gap=1e-6, max_abs_mp_residual=None,  # NEVER anchors
    mp_check_feastol=None, mp_measure_options=None,
    anchor_kind="NON_OFFICIAL_timeout_recovery",
)


def _finite_number(v: Any) -> TypeGuard[float]:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _measurement_criteria(measurement: dict[str, Any] | None,
                          provenance: dict[str, Any],
                          protocol: AmplProtocol) -> dict[str, bool]:
    """Fail-closed criteria on the residual-MEASUREMENT solve (A).

    The residual must come from a warn-mode measurement run that provably reproduced
    the primary solution; an absent/incomplete measurement fails every check. No
    ``Δ=0`` is asserted unless the deltas are present and finite.
    """
    threshold = protocol.max_abs_mp_residual
    if threshold is None or not isinstance(measurement, dict):
        return {
            "measurement_performed": False,
            "measurement_reproduces_primary": False,
            "measurement_options_conform": False,
            "measurement_hashes_match_primary": False,
            "mp_residual_present_and_ok": False,
        }
    dq = measurement.get("max_abs_delta_q")
    dt = measurement.get("max_abs_delta_theta")
    tol = protocol.measurement_match_tol
    residual = measurement.get("measured_max_abs_mp_residual")
    table = measurement.get("mp_violation_table")
    unparsed = isinstance(table, dict) and table.get("unparsed") is True
    return {
        "measurement_performed": measurement.get("performed") is True
            and measurement.get("status") == "optimal",
        "measurement_reproduces_primary": (
            measurement.get("solution_matches_primary") is True
            and _finite_number(dq) and _finite_number(dt)
            and dq <= tol and dt <= tol
        ),
        "measurement_options_conform":
            measurement.get("solver_options_requested") == protocol.mp_measure_options,
        "measurement_hashes_match_primary": all(
            measurement.get(k) is not None and measurement.get(k) == provenance.get(k)
            for k in ("model_sha256", "preprocessing_sha256", "data_sha256")
        ),
        "mp_residual_present_and_ok": (
            not unparsed and _finite_number(residual) and residual <= threshold
        ),
    }


def evaluate_anchor(
    *,
    status: str,
    extra: dict[str, Any],
    provenance: dict[str, Any],
    feasible_common_numeric: bool | None,
    n: int,
    q: tuple[float, ...] | None,
    theta: tuple[float, ...] | None,
    scientific_code_dirty: bool | None,
    measurement: dict[str, Any] | None = None,
    protocol: AmplProtocol = OFFICIAL_PROTOCOL,
) -> dict[str, Any]:
    """Fail-closed decision on whether a leg may anchor OFFICIAL deltas.

    Every criterion must be POSITIVELY satisfied by a present, finite value. Absent
    evidence is a failure. The positive result is a *numerical* optimum at gap <=
    threshold with a measured MP residual <= threshold, accepted by the common
    geometry's documented tolerance — NOT exact feasibility, no positive margin.
    """
    rel_gap = extra.get("mip_rel_gap")
    provider = extra.get("provider_objective")
    rescored = extra.get("rescored_objective")
    primal_ok = (
        q is not None and theta is not None
        and len(q) == n and len(theta) == n
        and all(_finite_number(v) for v in (*q, *theta))
    )
    echo = provenance.get("solver_options_echo") or {}
    options_match = (
        provenance.get("solver_options") == protocol.solver_options
        and bool(echo)
        and (protocol.mp_check_feastol is None or "sol:chk:feastol" in echo)
    )
    criteria = {
        "solver_status_optimal": status == "optimal",
        "rel_gap_le_threshold": (
            _finite_number(rel_gap) and rel_gap <= protocol.max_rel_optimality_gap
        ),
        "primal_complete_finite": primal_ok,
        "objective_concordance": protocol.objective_concordant(provider, rescored),
        "common_numeric_acceptance": feasible_common_numeric is True,
        **_measurement_criteria(measurement, provenance, protocol),
        "provenance_complete": all(
            provenance.get(k) for k in
            ("model_sha256", "preprocessing_sha256", "data_sha256",
             "ampl_version", "solver_version")
        ),
        "solver_identity_gurobi": (
            provenance.get("solver_requested") == protocol.solver
            and provenance.get("solver_effective") == protocol.solver
        ),
        "effective_options_match_protocol": options_match,
        "protocol_id_present": bool(provenance.get("protocol_id") == protocol.protocol_id),
        "scientific_code_clean": scientific_code_dirty is False,
    }
    is_anchor = all(criteria.values())
    return {
        "is_anchor": is_anchor,
        "anchor_kind": protocol.anchor_kind if is_anchor else None,
        "anchor_qualification": ANCHOR_QUALIFICATION if is_anchor else None,
        "criteria": criteria,
        "protocol_id": protocol.protocol_id,
    }


# --------------------------------------------------------------------------- #
# Machine-readable invalidation + official-anchor gate (B). Loaders/comparators/
# exporters MUST call accept_official_anchor(); they must never trust a stored
# ``is_official_anchor`` bool, a non-official ``protocol_id``, or an invalidated
# protocol.
# --------------------------------------------------------------------------- #
class AnchorRejected(Exception):
    """Raised when an artifact is refused as an official anchor."""


def _sha256_file(path: Any) -> str:
    import hashlib
    from pathlib import Path
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_invalidation_sidecar(protocol_dir: Any) -> dict[str, Any] | None:
    """Load + hash-verify ``INVALIDATED.json`` in ``protocol_dir``.

    Returns the sidecar dict if a hash-consistent invalidation is present, ``None``
    if there is no sidecar. Raises :class:`AnchorRejected` if the sidecar is present
    but tampered (an affected-artifact hash does not match the file on disk).
    """
    import json
    from pathlib import Path
    sidecar = Path(protocol_dir) / "INVALIDATED.json"
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnchorRejected(f"unreadable invalidation sidecar: {exc}") from exc
    for name, expected in (data.get("affected_artifact_hashes") or {}).items():
        target = Path(protocol_dir) / name
        if not target.is_file() or _sha256_file(target) != expected:
            raise AnchorRejected(
                f"invalidation sidecar hash mismatch for {name!r} (tampered artifact/record)")
    return data


def accept_official_anchor(
    artifact: dict[str, Any],
    protocol_dir: Any,
    *,
    official_protocol_id: str = OFFICIAL_PROTOCOL.protocol_id,
) -> bool:
    """Fail-closed gate for using an artifact as an OFFICIAL anchor.

    Refuses: a ``protocol_id`` != the official one (so v1/v2/unknown are rejected);
    an invalidated protocol (verified sidecar); and any artifact whose recorded
    ``anchor_criteria`` are absent or not all true. The stored ``is_official_anchor``
    flag alone is NEVER trusted. Returns ``True`` or raises :class:`AnchorRejected`.
    """
    pid = artifact.get("protocol_id")
    if pid != official_protocol_id:
        raise AnchorRejected(
            f"protocol_id {pid!r} is not the official {official_protocol_id!r}")
    reason = verify_invalidation_sidecar(protocol_dir)
    if reason is not None and reason.get("valid_for_anchoring") is not True:
        raise AnchorRejected(
            f"protocol {pid!r} invalidated: {reason.get('invalidation_reason')}")
    criteria = artifact.get("anchor_criteria")
    if not isinstance(criteria, dict) or not criteria or not all(criteria.values()):
        raise AnchorRejected("anchor_criteria absent or not all satisfied (fail-closed)")
    return True


# --------------------------------------------------------------------------- #
# Grid alignment guard: a QUBO/approximate result may ONLY be compared to the
# discrete reference computed on the IDENTICAL (n_theta, n_q) grid.
# --------------------------------------------------------------------------- #
class GridMismatchError(ValueError):
    """Raised when an approximate result is compared against a different grid."""


def grid_key(n_theta: int, n_q: int) -> str:
    return f"K{n_theta}_q{n_q}"


def assert_same_grid(reference_key: str, approximate_key: str) -> None:
    """Refuse to compare an approximate result to a differently-gridded reference."""
    if reference_key != approximate_key:
        raise GridMismatchError(
            f"grid mismatch: reference {reference_key!r} vs approximate {approximate_key!r} "
            "— an approximate/QUBO result must use the SAME K/n_q as its discrete reference")


def discrete_optimum_proven(*, status: str, search_space_size: int, search_cap: int) -> bool:
    """A discrete reference is a PROVEN optimum only when the exhaustive search was
    within the cap AND returned an ``optimal`` status.

    A heuristic fallback (search space over the cap) or a timeout / any non-optimal
    status is NEVER a proven optimum, so it can never anchor a discretisation delta.
    """
    return status == "optimal" and search_space_size <= search_cap
