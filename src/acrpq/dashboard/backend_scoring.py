"""Deterministic, explainable IBM-backend scoring for a QAOA workload.

This engine RANKS candidate backends for a given workload; it never submits a
job, never holds a token, never touches the network, and imports no Qiskit at
module load. It is a *recommendation* only — the chosen backend must still be
displayed and explicitly confirmed later (Phase 7).

Pipeline: hard applicability filters (a rejected backend can never be selected,
whatever its partial score) -> bounded, weighted multi-criteria scoring separated
into a *scientific* (physical quality) and an *operational* (availability) score
-> a structured, human-readable explanation per backend -> a deterministic
ranking with a stable name tie-break, a canonical JSON export and a SHA-256
decision hash covering the ENTIRE decision (including provenance).

Scores are heuristic and explicitly NOT a probability of quantum success. Missing
metrics are represented explicitly, raise the decision's uncertainty, and never
score as if the metric were good. A hard error cap requires the corresponding
metric to be present (a missing metric can never bypass a user constraint).
Topology quality is a coarse pre-transpilation estimate; a real post-transpilation
refinement is an optional injectable hook.

Identity & normalisation policy (documented, part of the contract):
- A backend's identity within a decision is its ``name`` (case-sensitive) after
  surrounding whitespace is stripped. Empty-after-strip names are refused.
  ``rank_backends`` refuses two snapshots with the same normalised name rather
  than silently merging them.
- ``region`` and ``family`` are stripped and must be non-empty.
- Error metrics live in ``[0, 1]``; counts/depths are non-negative non-bool ints;
  timestamps and budgets are finite and non-negative; ``max_quantum_seconds`` (if
  given) is strictly positive.
- Every stored/exported value is JSON-native and finite: no ``set``/``bytes``/
  ``datetime``/arbitrary object is ever silently coerced (no ``default=str``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

SCORING_SCHEMA_VERSION = "acrpq-backend-scoring/1"

# Reference scales for normalisation (documented, not IBM ground truth). They set
# what counts as "good vs bad" for each metric; tune per study, don't treat as
# durable hardware truth.
REF_TWO_QUBIT_ERROR = 0.02      # ~2% 2Q error maps to score 0
REF_READOUT_ERROR = 0.05        # ~5% readout error maps to score 0
REF_PENDING_JOBS = 200.0        # queue pressure scale
REF_CLOPS = 5000.0              # throughput scale
REF_FRESH_SECONDS = 24 * 3600.0  # data older than this scores 0 on freshness

# Rounding used across the module; contribution sums are consistent to this.
_ROUND = 6


# --------------------------------------------------------------------------- #
# Numeric / string / collection validators
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _req_finite(v: Any, name: str, *, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite, non-bool number; got {v!r}")
    fv = float(v)
    if lo is not None and fv < lo:
        raise ValueError(f"{name} must be >= {lo}; got {fv}")
    if hi is not None and fv > hi:
        raise ValueError(f"{name} must be <= {hi}; got {fv}")
    return fv


def _opt_finite(v: Any, name: str, *, lo: float | None = None, hi: float | None = None) -> float | None:
    return None if v is None else _req_finite(v, name, lo=lo, hi=hi)


def _req_int(v: Any, name: str, *, lo: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name} must be a non-bool int; got {v!r}")
    if lo is not None and v < lo:
        raise ValueError(f"{name} must be >= {lo}; got {v}")
    return v


def _req_bool(v: Any, name: str) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{name} must be a real bool; got {v!r}")
    return v


def _clean_str(v: Any, name: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{name} must be a string; got {type(v).__name__}")
    s = v.strip()
    if not s:
        raise ValueError(f"{name} must be a non-empty string")
    return s


def _str_tuple(v: Any, name: str, *, allow_none: bool) -> tuple[str, ...] | None:
    if v is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None")
    if not isinstance(v, tuple):
        raise ValueError(f"{name} must be a tuple of strings")
    out: list[str] = []
    for i, item in enumerate(v):
        s = _clean_str(item, f"{name}[{i}]")
        if s in out:
            raise ValueError(f"{name} contains a duplicate entry {s!r}")
        out.append(s)
    return tuple(out)


_OPTIONAL_METRIC_FIELDS = (
    "pending_jobs", "two_qubit_error", "two_qubit_error_layered",
    "readout_error", "clops", "avg_coupling_degree",
)


def _validate_provenance(v: Any) -> dict[str, Any]:
    """Provenance must be a JSON-native mapping (str keys, finite scalar values)."""
    if not isinstance(v, dict):
        raise ValueError("provenance must be a dict")
    out: dict[str, Any] = {}
    for k, val in v.items():
        if not isinstance(k, str) or not k:
            raise ValueError("provenance keys must be non-empty strings")
        if isinstance(val, bool) or val is None or isinstance(val, str):
            out[k] = val
        elif isinstance(val, (int, float)):
            if not math.isfinite(val):
                raise ValueError(f"provenance[{k!r}] must be finite")
            out[k] = val
        else:
            raise ValueError(f"provenance[{k!r}] must be a JSON-native scalar; got {type(val).__name__}")
    return out


# --------------------------------------------------------------------------- #
# Typed models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkloadRequirements:
    logical_qubits: int
    est_two_qubit_gates: int
    logical_depth: int
    interaction_density: float           # 0..1: fraction of qubit pairs interacting
    needs_dynamic_circuits: bool = False
    needs_fractional_gates: bool = False
    shots: int = 1024
    max_quantum_seconds: float | None = None
    preferred_region: str | None = None
    allowed_regions: tuple[str, ...] | None = None   # None -> any
    allowed_backends: tuple[str, ...] | None = None   # None -> any
    blocked_backends: tuple[str, ...] = ()
    max_two_qubit_error: float | None = None          # hard reject above this
    max_readout_error: float | None = None
    require_all_metrics: bool = False                 # data-completeness policy
    decision_ts: float = 0.0                          # injected clock (epoch seconds)

    def __post_init__(self) -> None:
        _req_int(self.logical_qubits, "logical_qubits", lo=1)
        _req_int(self.est_two_qubit_gates, "est_two_qubit_gates", lo=0)
        _req_int(self.logical_depth, "logical_depth", lo=0)
        _req_int(self.shots, "shots", lo=1)
        _req_finite(self.interaction_density, "interaction_density", lo=0.0, hi=1.0)
        for flag in ("needs_dynamic_circuits", "needs_fractional_gates", "require_all_metrics"):
            _req_bool(getattr(self, flag), flag)
        if self.max_quantum_seconds is not None:
            v = _req_finite(self.max_quantum_seconds, "max_quantum_seconds")
            if v <= 0:
                raise ValueError("max_quantum_seconds must be strictly positive")
        _opt_finite(self.max_two_qubit_error, "max_two_qubit_error", lo=0.0, hi=1.0)
        _opt_finite(self.max_readout_error, "max_readout_error", lo=0.0, hi=1.0)
        _req_finite(self.decision_ts, "decision_ts", lo=0.0)
        if self.preferred_region is not None:
            _clean_str(self.preferred_region, "preferred_region")
        # normalise collections (strict, deduped) and store canonical tuples
        object.__setattr__(self, "allowed_regions",
                           _str_tuple(self.allowed_regions, "allowed_regions", allow_none=True))
        object.__setattr__(self, "allowed_backends",
                           _str_tuple(self.allowed_backends, "allowed_backends", allow_none=True))
        blocked = _str_tuple(self.blocked_backends, "blocked_backends", allow_none=False)
        object.__setattr__(self, "blocked_backends", blocked or ())


@dataclass(frozen=True)
class BackendSnapshot:
    name: str
    region: str
    family: str
    physical_qubits: int
    programmable_qubits: int
    operational: bool
    maintenance: bool
    pending_jobs: int | None = None
    two_qubit_error: float | None = None       # best/representative 2Q error
    two_qubit_error_layered: float | None = None
    readout_error: float | None = None
    clops: float | None = None
    supports_dynamic: bool = False
    supports_fractional: bool = False
    avg_coupling_degree: float | None = None   # coarse topology summary
    data_ts: float = 0.0                        # epoch seconds of this snapshot
    provenance: dict[str, Any] = field(default_factory=dict)   # metric -> source
    unknown_fields: tuple[str, ...] = ()        # explicitly-unknown metric names
    synthetic: bool = False                     # True for test/fixture snapshots

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_str(self.name, "name"))
        object.__setattr__(self, "region", _clean_str(self.region, "region"))
        object.__setattr__(self, "family", _clean_str(self.family, "family"))
        phys = _req_int(self.physical_qubits, "physical_qubits", lo=0)
        prog = _req_int(self.programmable_qubits, "programmable_qubits", lo=0)
        if prog > phys:
            raise ValueError(f"programmable_qubits ({prog}) > physical_qubits ({phys})")
        for flag in ("operational", "maintenance", "supports_dynamic",
                     "supports_fractional", "synthetic"):
            _req_bool(getattr(self, flag), flag)
        if self.pending_jobs is not None:
            _req_int(self.pending_jobs, "pending_jobs", lo=0)
        _opt_finite(self.two_qubit_error, "two_qubit_error", lo=0.0, hi=1.0)
        _opt_finite(self.two_qubit_error_layered, "two_qubit_error_layered", lo=0.0, hi=1.0)
        _opt_finite(self.readout_error, "readout_error", lo=0.0, hi=1.0)
        _opt_finite(self.clops, "clops", lo=0.0)
        _opt_finite(self.avg_coupling_degree, "avg_coupling_degree", lo=0.0)
        _req_finite(self.data_ts, "data_ts", lo=0.0)
        # provenance: JSON-native + defensive copy (external mutation must not leak in)
        object.__setattr__(self, "provenance", _validate_provenance(self.provenance))
        # unknown_fields: strict, deduped, coherent with actually-missing metrics
        uf = _str_tuple(self.unknown_fields, "unknown_fields", allow_none=False) or ()
        for name in uf:
            if name not in _OPTIONAL_METRIC_FIELDS:
                raise ValueError(f"unknown_fields entry {name!r} is not an optional metric")
            if getattr(self, name) is not None:
                raise ValueError(f"unknown_fields lists {name!r} but it has a value")
        object.__setattr__(self, "unknown_fields", uf)


@dataclass(frozen=True)
class BackendAssessment:
    name: str
    applicable: bool
    rejection_reasons: tuple[dict[str, str], ...]
    scientific_score: float | None
    operational_score: float | None
    total_score: float | None
    subscores: dict[str, float]
    penalties: dict[str, float]
    contributions: dict[str, float]
    metrics_used: tuple[str, ...]
    metrics_missing: tuple[str, ...]
    uncertainty: float
    rank: int | None
    tie_break_key: str
    explanation: dict[str, Any]


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoringWeights:
    # scientific (physical quality)
    two_qubit_quality: float = 0.30
    readout_quality: float = 0.15
    topological_fit: float = 0.15
    qubit_margin: float = 0.05
    # operational (availability)
    queue_pressure: float = 0.15
    throughput: float = 0.05
    data_freshness: float = 0.05
    metric_completeness: float = 0.05
    region_preference: float = 0.05

    _SCIENTIFIC = ("two_qubit_quality", "readout_quality", "topological_fit", "qubit_margin")
    _OPERATIONAL = ("queue_pressure", "throughput", "data_freshness",
                    "metric_completeness", "region_preference")

    def __post_init__(self) -> None:
        for name in self._SCIENTIFIC + self._OPERATIONAL:
            _req_finite(getattr(self, name), f"weight {name}", lo=0.0)
        total = sum(getattr(self, n) for n in self._SCIENTIFIC + self._OPERATIONAL)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"weights must sum to 1.0; got {total}")

    def as_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in self._SCIENTIFIC + self._OPERATIONAL}


DEFAULT_WEIGHTS = ScoringWeights()

# Uncertainty above this makes a backend non-selectable (still scored/explained).
DEFAULT_UNCERTAINTY_BLOCK = 0.6


# --------------------------------------------------------------------------- #
# Hard applicability filters
# --------------------------------------------------------------------------- #
def _hard_rejections(req: WorkloadRequirements, snap: BackendSnapshot) -> list[dict[str, str]]:
    r: list[dict[str, str]] = []
    if not snap.operational:
        r.append({"code": "offline", "message": "backend is not operational"})
    if snap.maintenance:
        r.append({"code": "maintenance", "message": "backend is in maintenance"})
    if snap.programmable_qubits < req.logical_qubits:
        r.append({"code": "insufficient_qubits",
                  "message": f"{snap.programmable_qubits} programmable < {req.logical_qubits} required"})
    if req.needs_dynamic_circuits and not snap.supports_dynamic:
        r.append({"code": "no_dynamic_circuits", "message": "dynamic circuits not supported"})
    if req.needs_fractional_gates and not snap.supports_fractional:
        r.append({"code": "no_fractional_gates", "message": "fractional gates not supported"})
    if req.allowed_regions is not None and snap.region not in req.allowed_regions:
        r.append({"code": "region_not_allowed", "message": f"region {snap.region!r} not allowed"})
    if req.allowed_backends is not None and snap.name not in req.allowed_backends:
        r.append({"code": "not_in_allowlist", "message": "backend not in the allow-list"})
    if snap.name in req.blocked_backends:
        r.append({"code": "blocked", "message": "backend explicitly blocked"})
    # A hard error cap REQUIRES the metric — a missing metric never bypasses it.
    if req.max_two_qubit_error is not None:
        if snap.two_qubit_error is None:
            r.append({"code": "two_qubit_error_required_for_threshold",
                      "message": "a 2Q-error cap is set but the backend reports no 2Q error"})
        elif snap.two_qubit_error > req.max_two_qubit_error:
            r.append({"code": "two_qubit_error_too_high",
                      "message": f"2Q error {snap.two_qubit_error} > cap {req.max_two_qubit_error}"})
    if req.max_readout_error is not None:
        if snap.readout_error is None:
            r.append({"code": "readout_error_required_for_threshold",
                      "message": "a readout-error cap is set but the backend reports no readout error"})
        elif snap.readout_error > req.max_readout_error:
            r.append({"code": "readout_error_too_high",
                      "message": f"readout {snap.readout_error} > cap {req.max_readout_error}"})
    if req.require_all_metrics:
        missing = [m for m in ("two_qubit_error", "readout_error", "pending_jobs")
                   if getattr(snap, m) is None]
        if missing:
            r.append({"code": "missing_required_metrics",
                      "message": f"required metrics missing: {', '.join(missing)}"})
    return r


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_ALL_DIMS = ScoringWeights._SCIENTIFIC + ScoringWeights._OPERATIONAL


def _subscores(
    req: WorkloadRequirements, snap: BackendSnapshot
) -> tuple[dict[str, float], list[str], list[str], list[str]]:
    """Per-dimension score in [0,1]; a missing metric scores 0 and is flagged.

    Returns (subscores, used, missing, warnings).
    """
    used: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []
    s: dict[str, float] = {}

    def metric(dim: str, value: float | None, score_if_present: float) -> None:
        if value is None:
            s[dim] = 0.0          # missing never beats a known-good metric
            missing.append(dim)
        else:
            s[dim] = _clamp01(score_if_present)
            used.append(dim)

    metric("two_qubit_quality", snap.two_qubit_error,
           1.0 - (snap.two_qubit_error or 0.0) / REF_TWO_QUBIT_ERROR)
    metric("readout_quality", snap.readout_error,
           1.0 - (snap.readout_error or 0.0) / REF_READOUT_ERROR)
    metric("topological_fit", snap.avg_coupling_degree,
           _topo_fit(req, snap.avg_coupling_degree))
    # qubit margin is always known (qubit counts are mandatory)
    s["qubit_margin"] = _clamp01((snap.programmable_qubits - req.logical_qubits)
                                 / max(1, req.logical_qubits))
    used.append("qubit_margin")
    metric("queue_pressure", (None if snap.pending_jobs is None else float(snap.pending_jobs)),
           1.0 - (snap.pending_jobs or 0) / REF_PENDING_JOBS)
    metric("throughput", snap.clops, (snap.clops or 0.0) / REF_CLOPS)
    # freshness from the injected clock + data_ts. A future snapshot is NOT
    # silently treated as perfectly fresh: it scores 0 and raises a warning.
    age = req.decision_ts - snap.data_ts
    if age < 0:
        s["data_freshness"] = 0.0
        warnings.append("future_data_timestamp")
    else:
        s["data_freshness"] = _clamp01(1.0 - age / REF_FRESH_SECONDS)
    used.append("data_freshness")
    # completeness across the optional metrics
    optional = ("two_qubit_error", "readout_error", "avg_coupling_degree", "pending_jobs", "clops")
    present = sum(1 for m in optional if getattr(snap, m) is not None)
    s["metric_completeness"] = present / len(optional)
    used.append("metric_completeness")
    # region preference (soft): preferred -> 1, otherwise 0.5
    if req.preferred_region is None:
        s["region_preference"] = 1.0
    else:
        s["region_preference"] = 1.0 if snap.region == req.preferred_region else 0.5
    used.append("region_preference")
    if not snap.synthetic and not snap.provenance:
        warnings.append("no_provenance")
    return s, used, missing, warnings


def _topo_fit(req: WorkloadRequirements, avg_degree: float | None) -> float:
    """Coarse pre-transpilation topology fit; higher coupling degree vs the
    workload's interaction density scores better. This is an approximation — the
    true routing cost needs transpilation (see :func:`refine_with_transpilation`)."""
    if avg_degree is None:
        return 0.0
    # a device degree of ~ (density * (n-1)) would embed the interactions locally
    required = max(1.0, req.interaction_density * (req.logical_qubits - 1))
    return _clamp01(avg_degree / required)


def _uncertainty(missing: list[str], freshness: float) -> float:
    """0 (certain) .. 1 (very uncertain), from missing metrics + data staleness."""
    optional_dims = ("two_qubit_quality", "readout_quality", "topological_fit",
                     "queue_pressure", "throughput")
    miss_frac = sum(1 for d in optional_dims if d in missing) / len(optional_dims)
    stale = 1.0 - freshness
    return _clamp01(0.6 * miss_frac + 0.4 * stale)


def assess_backend(
    req: WorkloadRequirements, snap: BackendSnapshot, *, weights: ScoringWeights = DEFAULT_WEIGHTS
) -> BackendAssessment:
    """Filter, score and explain a single backend (no ranking)."""
    rejections = _hard_rejections(req, snap)
    if rejections:
        return BackendAssessment(
            name=snap.name, applicable=False, rejection_reasons=tuple(rejections),
            scientific_score=None, operational_score=None, total_score=None,
            subscores={}, penalties={}, contributions={}, metrics_used=(), metrics_missing=(),
            uncertainty=1.0, rank=None, tie_break_key=snap.name,
            explanation={"compatible": False, "rejections": rejections,
                         "heuristic_warning": _HEURISTIC_WARNING},
        )

    subs, used, missing, warnings = _subscores(req, snap)
    contributions = {d: round(getattr(weights, d) * subs[d], _ROUND) for d in _ALL_DIMS}
    sci = sum(getattr(weights, d) * subs[d] for d in ScoringWeights._SCIENTIFIC)
    op = sum(getattr(weights, d) * subs[d] for d in ScoringWeights._OPERATIONAL)
    total = _clamp01(sci + op)
    penalties = {d: round(getattr(weights, d) * (1.0 - subs[d]), _ROUND)
                 for d in _ALL_DIMS if getattr(weights, d) > 0}
    unc = _uncertainty(missing, subs["data_freshness"])
    expl = _build_explanation(snap, subs, contributions, penalties, missing,
                              warnings, req, weights, unc, round(total, _ROUND))
    return BackendAssessment(
        name=snap.name, applicable=True, rejection_reasons=(),
        scientific_score=round(sci, _ROUND), operational_score=round(op, _ROUND),
        total_score=round(total, _ROUND), subscores={k: round(v, _ROUND) for k, v in subs.items()},
        penalties=penalties, contributions=contributions,
        metrics_used=tuple(used), metrics_missing=tuple(missing),
        uncertainty=round(unc, _ROUND), rank=None, tie_break_key=snap.name, explanation=expl,
    )


_HEURISTIC_WARNING = (
    "Heuristic ranking of hardware suitability from reported calibration/queue "
    "metrics; NOT a probability of quantum-computational success."
)


def _build_explanation(snap, subs, contributions, penalties, missing, warnings,
                       req, weights, unc, total) -> dict[str, Any]:
    # favourable = real weighted contribution (weight x subscore), not raw subscore,
    # so a subscore of 1 on a 0.05-weight dim cannot leapfrog a 0.20 contribution,
    # and a zero-weight dimension can never appear as a favourable factor.
    fav = sorted(((d, contributions[d]) for d in _ALL_DIMS if contributions[d] > 0),
                 key=lambda kv: (-kv[1], kv[0]))[:3]
    pen = sorted(penalties.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    age_s = req.decision_ts - snap.data_ts
    return {
        "compatible": True,
        "top_favourable": [{"dimension": d, "subscore": round(subs[d], 4),
                            "weight": getattr(weights, d), "contribution": round(c, 4)}
                           for d, c in fav],
        "top_penalties": [{"dimension": d, "subscore": round(subs[d], 4),
                           "weight": getattr(weights, d), "penalty": round(p, 4)}
                          for d, p in pen],
        "contribution_sum": round(sum(contributions.values()), 4),
        "total_score": total,
        "missing_metrics": list(missing),
        "warnings": list(warnings),
        "metric_age_seconds": round(age_s, 3),
        "uncertainty": round(unc, 4),
        "weights": weights.as_dict(),
        "heuristic_warning": _HEURISTIC_WARNING,
    }


# --------------------------------------------------------------------------- #
# Strict JSON-native serialisation
# --------------------------------------------------------------------------- #
def _json_native(obj: Any, path: str = "$") -> Any:
    """Return a JSON-native, finite copy of ``obj`` or raise.

    Accepts dict (str keys), list, tuple (-> list, explicit policy), str, bool,
    int, finite float, None. Refuses set, bytes, datetime, arbitrary objects,
    non-string keys, NaN and Inf. No ``default=str`` coercion anywhere.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{path}: non-finite float is not JSON-native")
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"{path}: dict key {k!r} is not a string")
            out[k] = _json_native(v, f"{path}.{k}")
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_native(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    raise ValueError(f"{path}: {type(obj).__name__} is not JSON-native")


def _canonical(obj: Any) -> str:
    return json.dumps(_json_native(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------- #
# Shared workload commitment (binds a Phase-4 decision to a downstream workload)
# --------------------------------------------------------------------------- #
def workload_commitment(req: WorkloadRequirements) -> str:
    """Canonical SHA-256 of the full workload requirements.

    A downstream phase (e.g. hardware-validation) can recompute this from the SAME
    :class:`WorkloadRequirements` and require it to equal the ``workload_sha256``
    embedded in a decision record — so a decision made for one workload can never
    be reused for a different (or stronger) one.
    """
    return "sha256:" + hashlib.sha256(_canonical(asdict(req)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Ranking + provenance
# --------------------------------------------------------------------------- #
def _validate_uncertainty_block(v: Any) -> float:
    return _req_finite(v, "uncertainty_block", lo=0.0, hi=1.0)


def rank_backends(
    req: WorkloadRequirements,
    snapshots: list[BackendSnapshot],
    *,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    uncertainty_block: float = DEFAULT_UNCERTAINTY_BLOCK,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank backends deterministically and return an explainable decision record.

    ``selected`` is the rank-1 applicable backend UNLESS none is applicable or the
    best one's uncertainty exceeds ``uncertainty_block`` — in which case selection
    is withheld (never silent). Ties break by backend name for byte-stable output.

    A backend's identity is its (whitespace-stripped) name; two snapshots with the
    same name are refused rather than silently merged. The decision hash covers the
    ENTIRE record (schema, workload, snapshots, weights, block, ranking, rejects,
    explanations and any supplied provenance); only ``decision_sha256`` itself is
    excluded from the hashed content.
    """
    _validate_uncertainty_block(uncertainty_block)
    seen: set[str] = set()
    for s in snapshots:
        if s.name in seen:
            raise ValueError(f"duplicate backend name {s.name!r}: name is the identity in a decision")
        seen.add(s.name)

    assessments = [assess_backend(req, s, weights=weights) for s in snapshots]
    applicable = [a for a in assessments if a.applicable]
    # deterministic order: highest total first, then name asc (stable tie-break)
    applicable.sort(key=lambda a: (-(a.total_score or 0.0), a.tie_break_key))

    ranked = [_with_rank(a, i + 1) for i, a in enumerate(applicable)]
    rejected = sorted((a for a in assessments if not a.applicable), key=lambda a: a.tie_break_key)

    selected: str | None = None
    selection_note: str
    if not ranked:
        selection_note = "no_applicable_backend"
    elif ranked[0].uncertainty > uncertainty_block:
        selection_note = (f"withheld: top candidate uncertainty {ranked[0].uncertainty} "
                          f"exceeds block threshold {uncertainty_block}")
    else:
        selected = ranked[0].name
        selection_note = "top-ranked applicable backend"

    # Build the FULL record first (including provenance), then hash it. Provenance
    # is deep-copied so a later mutation of the caller's dict cannot change the
    # record, and it is validated JSON-native.
    record: dict[str, Any] = {
        "schema": SCORING_SCHEMA_VERSION,
        "selected": selected,
        "selection_note": selection_note,
        "decision_ts": req.decision_ts,
        "weights": weights.as_dict(),
        "uncertainty_block": uncertainty_block,
        "workload": asdict(req),
        "workload_sha256": workload_commitment(req),
        "ranking": [_assessment_row(a) for a in ranked],
        "rejected": [_assessment_row(a) for a in rejected],
        # embedded snapshots sorted by name so the decision hash is independent of
        # the caller's input ordering (byte-for-byte determinism)
        "snapshots": [asdict(s) for s in sorted(snapshots, key=lambda s: s.name)],
        "provenance": _validate_provenance_record(provenance),
    }
    record["decision_sha256"] = "sha256:" + hashlib.sha256(
        _canonical(record).encode("utf-8")
    ).hexdigest()
    return record


def _validate_provenance_record(provenance: dict[str, Any] | None) -> dict[str, Any] | None:
    if provenance is None:
        return None
    # JSON-native (deep) validation + a defensive deep copy.
    return copy.deepcopy(_json_native(provenance))


def verify_decision_hash(record: dict[str, Any]) -> bool:
    """Recompute the decision hash over everything except ``decision_sha256``.

    Returns True iff the stored hash matches. Refuses records whose content is not
    JSON-native / finite (``_canonical`` raises), so a tampered non-native payload
    cannot pass silently.
    """
    claimed = record.get("decision_sha256")
    if not isinstance(claimed, str):
        return False
    payload = {k: v for k, v in record.items() if k != "decision_sha256"}
    recomputed = "sha256:" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return claimed == recomputed


def _with_rank(a: BackendAssessment, rank: int) -> BackendAssessment:
    return BackendAssessment(**{**asdict(a), "rank": rank,
                                "rejection_reasons": a.rejection_reasons,
                                "metrics_used": a.metrics_used,
                                "metrics_missing": a.metrics_missing})


def _assessment_row(a: BackendAssessment) -> dict[str, Any]:
    return {
        "name": a.name, "applicable": a.applicable, "rank": a.rank,
        "total_score": a.total_score, "scientific_score": a.scientific_score,
        "operational_score": a.operational_score, "subscores": a.subscores,
        "penalties": a.penalties, "contributions": a.contributions,
        "uncertainty": a.uncertainty,
        "metrics_used": list(a.metrics_used), "metrics_missing": list(a.metrics_missing),
        "rejection_reasons": [dict(r) for r in a.rejection_reasons],
        "tie_break_key": a.tie_break_key, "explanation": a.explanation,
    }


def compare(a: BackendAssessment, b: BackendAssessment) -> dict[str, Any]:
    """Explain why ``a`` is (or is not) ranked ahead of ``b``."""
    if not a.applicable or not b.applicable:
        return {"verdict": "one_or_both_inapplicable",
                "a_applicable": a.applicable, "b_applicable": b.applicable}
    diffs = sorted(
        ((d, round(a.subscores.get(d, 0.0) - b.subscores.get(d, 0.0), _ROUND)) for d in _ALL_DIMS),
        key=lambda kv: -abs(kv[1]),
    )
    ahead = (a.total_score or 0.0) > (b.total_score or 0.0) or (
        (a.total_score or 0.0) == (b.total_score or 0.0) and a.tie_break_key < b.tie_break_key
    )
    return {
        "ahead": a.name if ahead else b.name,
        "total_delta": round((a.total_score or 0.0) - (b.total_score or 0.0), _ROUND),
        "top_dimension_deltas": [{"dimension": d, "a_minus_b": v} for d, v in diffs[:3]],
        "tie_broken_by_name": (a.total_score == b.total_score),
    }


# --------------------------------------------------------------------------- #
# Optional post-transpilation refinement + injectable data source
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TranspilationMetrics:
    isa_depth: int
    two_qubit_gate_count: int
    swap_count: int
    layout_ok: bool

    def __post_init__(self) -> None:
        _req_int(self.isa_depth, "isa_depth", lo=0)
        _req_int(self.two_qubit_gate_count, "two_qubit_gate_count", lo=0)
        _req_int(self.swap_count, "swap_count", lo=0)
        _req_bool(self.layout_ok, "layout_ok")


def refine_with_transpilation(
    assessment: BackendAssessment, metrics: TranspilationMetrics
) -> BackendAssessment:
    """Attach real post-transpilation metrics to an assessment's explanation.

    This does NOT silently rewrite the heuristic score; it records the measured
    ISA cost so the UI/mémoire can show the true routing overhead alongside the
    pre-transpilation estimate. The returned assessment does not share mutable
    structures with ``assessment``.
    """
    expl = copy.deepcopy(assessment.explanation)
    expl["post_transpilation"] = {
        "isa_depth": metrics.isa_depth, "two_qubit_gate_count": metrics.two_qubit_gate_count,
        "swap_count": metrics.swap_count, "layout_ok": metrics.layout_ok,
    }
    return BackendAssessment(**{**asdict(assessment), "explanation": expl,
                                "rejection_reasons": assessment.rejection_reasons,
                                "metrics_used": assessment.metrics_used,
                                "metrics_missing": assessment.metrics_missing})


@runtime_checkable
class BackendDataSource(Protocol):
    """Injectable collector that yields snapshots; kept out of the scoring logic."""

    def snapshots(self) -> list[BackendSnapshot]: ...
