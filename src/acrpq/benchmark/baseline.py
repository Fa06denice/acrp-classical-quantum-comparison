"""Scientific baseline comparison — the four ACRP objective levels and the
deltas/ratios between them.

The benchmark produces one :class:`~acrpq.benchmark.export.RunRecord` per solver
leg. This module reads those records (never re-runs anything) and derives the
*scientific* comparison the thesis needs, written to a SEPARATE artifact so the
per-record CSV/JSON schema stays untouched:

* ``original_ampl``      — continuous solution of the historical Code/NC_ACRP.mod
                           (AMPL + Couenne). A *proven* optimum only when Couenne
                           reports ``solved`` (Status.OPTIMAL); a ``solved?``/
                           feasible result is a feasible incumbent, not the optimum.
* ``pyomo_minlp``        — the Python rebuild of the same continuous model.
* ``discrete_reference`` — grid optimum; a proven optimum only in its exact mode.
* approximate legs       — qubo_exact / qubo_annealing / quantum_qaoa / D-Wave.

From these it computes, using the **rescored** (common-geometry) objective of
each leg — never the solver's self-reported number:

* ``delta_adaptation``     = pyomo_minlp        − original_ampl   (Python rebuild cost)
* ``delta_discretization`` = discrete_reference − original_ampl   (grid cost)
* ``delta_algorithm``      = approximate        − discrete_reference (per method)

An OFFICIAL delta anchors ONLY on a **proven optimum**: ``original_ampl`` must be
``Status.OPTIMAL``, feasible under the shared geometry, finite AND ``integrity_ok``;
``discrete_reference`` must likewise be a proven ``optimal``. A merely ``feasible``
anchor (a Couenne ``solved?`` incumbent, or the discrete reference in its heuristic
mode) yields a ``None`` official delta plus a clearly-labelled
``provisional_vs_feasible_incumbent`` entry — never presented as the official gap.
``unavailable`` (AMPL not installed), ``unknown`` (integrity divergence),
``timeout``, ``infeasible`` and ``error`` contribute nothing. Pyomo can therefore
never silently stand in for a missing or unproven original-AMPL optimum.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .export import RunRecord

# Canonical display / export order for the scientific legs.
BASELINE_LEG_ORDER: tuple[str, ...] = (
    "original_ampl",
    "classical_minlp",  # pyomo_minlp
    "classical_discrete",  # discrete_reference
    "qubo_exact",
    "qubo_annealing",
    "quantum_qaoa",
    "qubo_dwave",
    "qubo_dwave_sa",
)

# Legs treated as approximate methods measured against the discrete reference.
_APPROXIMATE_KINDS: tuple[str, ...] = (
    "qubo_exact",
    "qubo_annealing",
    "quantum_qaoa",
    "qubo_dwave",
    "qubo_dwave_sa",
)

# A feasible incumbent (usable only for *provisional*, clearly-labelled metrics).
_INCUMBENT_STATUSES = frozenset({"optimal", "feasible"})

# The supervised protocol's AMPL solver — the ONLY one that anchors an official
# continuous delta. Any other AMPL solver (e.g. couenne) is kept as a separate,
# explicitly-named validation leg and never merged under `original_ampl`.
SUPERVISED_AMPL_SOLVER = "gurobi"


def _ampl_solver_of(rec: RunRecord) -> str | None:
    """Extract the AMPL solver from an original_ampl record's backend tag.

    ``run_original_ampl`` tags the record backend ``ampl:<solver>``; return
    ``<solver>`` (e.g. ``gurobi``/``couenne``) or ``None`` if unknown.
    """
    backend = rec.repro.backend or ""
    for sep in (":", "+"):
        if backend.startswith(f"ampl{sep}"):
            return backend.split(sep, 1)[1] or None
    return None


@dataclass(frozen=True)
class LegSummary:
    """The audit-relevant facts about one solver leg, straight from its record."""

    solver_kind: str
    status: str
    feasible: bool
    objective: float | None  # rescored (common-geometry); None if not finite
    provider_objective: float | None  # solver self-report, for audit only
    integrity_ok: bool | None  # provider vs rescored agreement (AMPL/Pyomo only)
    is_incumbent: bool  # feasible + finite: a usable incumbent, NOT a proven optimum
    is_optimal_anchor: bool  # proven optimum: may anchor an OFFICIAL delta
    ampl_solver: str | None = None  # the AMPL solver (gurobi/couenne/…) for AMPL legs

    @staticmethod
    def from_record(rec: RunRecord) -> "LegSummary":
        obj = rec.common.objective
        obj_val = obj if isinstance(obj, (int, float)) and math.isfinite(obj) else None
        prov = rec.extra.get("provider_objective")
        prov_val = prov if isinstance(prov, (int, float)) and math.isfinite(prov) else None
        integ = rec.extra.get("integrity_ok")
        integrity = None if integ is None else bool(integ)
        ampl_solver = _ampl_solver_of(rec) if rec.solver_kind == "original_ampl" else None
        incumbent = (
            rec.status in _INCUMBENT_STATUSES
            and bool(rec.common.feasible)
            and obj_val is not None
        )
        # An OFFICIAL anchor must be a PROVEN optimum: status 'optimal', feasible
        # under the shared geometry, finite, AND integrity-checked. A leg that
        # self-reports an objective (original_ampl / pyomo_minlp) must have
        # integrity_ok TRUE; a leg with no provider objective (discrete_reference)
        # only needs it to be non-False. A merely 'feasible' incumbent — a heuristic
        # discrete reference or a Couenne feasible-but-not-proven point — never
        # anchors an official delta.
        integrity_holds = (integrity is True) if prov_val is not None else (integrity is not False)
        optimal_anchor = rec.status == "optimal" and incumbent and integrity_holds
        return LegSummary(
            solver_kind=rec.solver_kind,
            status=rec.status,
            feasible=bool(rec.common.feasible),
            objective=obj_val,
            provider_objective=prov_val,
            integrity_ok=integrity,
            is_incumbent=incumbent,
            is_optimal_anchor=optimal_anchor,
            ampl_solver=ampl_solver,
        )


@dataclass(frozen=True)
class Delta:
    """A signed difference and its ratio, plus a note when either is undefined."""

    value: str  # which leg is being measured
    base: str  # against which anchor
    delta: float | None
    ratio: float | None
    note: str = ""


@dataclass
class BaselineComparison:
    """The scientific comparison for one scenario, derived from its records."""

    scenario_id: str
    instance_name: str
    family: str
    n: int
    grid_k: int
    legs: dict[str, LegSummary]
    # OFFICIAL scientific deltas: anchored ONLY on a proven optimum. Null otherwise.
    delta_adaptation: Delta
    delta_discretization: Delta
    delta_algorithm: list[Delta]
    # Explicitly-labelled, NON-official metrics computed against a merely-feasible
    # incumbent (original_ampl or discrete_reference) when no proven optimum exists.
    provisional_vs_feasible_incumbent: list[Delta] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _delta(value: str, base: str, num: float | None, den: float | None) -> Delta:
    """Signed ``num - den`` and ratio ``num / den``, with honest None + note."""
    if num is None and den is None:
        return Delta(value, base, None, None, note="both legs missing/untrustworthy")
    if num is None:
        return Delta(value, base, None, None, note=f"{value} not a trustworthy anchor")
    if den is None:
        return Delta(value, base, None, None, note=f"{base} not a trustworthy anchor")
    diff = num - den
    if not math.isfinite(diff):
        return Delta(value, base, None, None, note="non-finite difference")
    ratio: float | None
    if den == 0.0:
        ratio = None
        note = "ratio undefined: zero denominator"
    else:
        ratio = num / den
        if not math.isfinite(ratio):
            ratio = None
            note = "ratio non-finite"
        else:
            note = ""
    return Delta(value, base, diff, ratio, note=note)


def _optimal_base(leg: LegSummary | None) -> float | None:
    """Objective usable as an OFFICIAL anchor: only a proven optimum qualifies."""
    return leg.objective if (leg is not None and leg.is_optimal_anchor) else None


def _incumbent_obj(leg: LegSummary | None) -> float | None:
    """Objective of a feasible incumbent (numerator side of a delta)."""
    return leg.objective if (leg is not None and leg.is_incumbent) else None


def _feasible_only_base(leg: LegSummary | None) -> float | None:
    """Objective of a leg that is a feasible incumbent but NOT a proven optimum."""
    if leg is not None and leg.is_incumbent and not leg.is_optimal_anchor:
        return leg.objective
    return None


def compute_baseline(records: list[RunRecord]) -> BaselineComparison | None:
    """Derive the scientific baseline comparison from one scenario's records.

    Returns ``None`` when ``records`` is empty. Records are assumed to belong to a
    single scenario; the first record supplies the scenario/instance identity.

    OFFICIAL deltas anchor ONLY on a proven optimum (``original_ampl`` for the
    adaptation/discretisation gaps, ``discrete_reference`` for the algorithmic
    gap). When an anchor is merely feasible, the official delta is ``None`` and a
    clearly-labelled ``provisional_vs_feasible_incumbent`` entry is recorded
    instead — never presented as the official gap.
    """
    if not records:
        return None
    head = records[0]
    legs: dict[str, LegSummary] = {}
    for rec in records:
        summary = LegSummary.from_record(rec)
        key = rec.solver_kind
        if key == "original_ampl" and summary.ampl_solver not in (None, SUPERVISED_AMPL_SOLVER):
            # A non-supervised AMPL solver (e.g. couenne) is a SEPARATE validation
            # leg — never merged under `original_ampl`, never the official anchor.
            key = f"original_ampl_{summary.ampl_solver}"
        legs.setdefault(key, summary)

    original = legs.get("original_ampl")   # the supervised (gurobi) anchor
    pyomo = legs.get("classical_minlp")
    discrete = legs.get("classical_discrete")

    # OFFICIAL anchors — proven optima only.
    o_base = _optimal_base(original)
    d_base = _optimal_base(discrete)

    delta_adaptation = _delta("pyomo_minlp", "original_ampl", _incumbent_obj(pyomo), o_base)
    delta_discretization = _delta(
        "discrete_reference", "original_ampl", _incumbent_obj(discrete), o_base
    )

    delta_algorithm: list[Delta] = []
    for kind in _APPROXIMATE_KINDS:
        leg = legs.get(kind)
        if leg is None:
            continue
        delta_algorithm.append(
            _delta(kind, "discrete_reference", _incumbent_obj(leg), d_base)
        )

    # PROVISIONAL metrics vs a feasible-but-not-proven incumbent (never official).
    provisional: list[Delta] = []
    o_feas = _feasible_only_base(original)
    if o_feas is not None:
        provisional.append(
            _delta("pyomo_minlp", "original_ampl_feasible_incumbent",
                   _incumbent_obj(pyomo), o_feas)
        )
        provisional.append(
            _delta("discrete_reference", "original_ampl_feasible_incumbent",
                   _incumbent_obj(discrete), o_feas)
        )
    d_feas = _feasible_only_base(discrete)
    if d_feas is not None:
        for kind in _APPROXIMATE_KINDS:
            leg = legs.get(kind)
            if leg is None:
                continue
            provisional.append(
                _delta(kind, "discrete_reference_feasible_incumbent",
                       _incumbent_obj(leg), d_feas)
            )

    notes = _build_notes(original, pyomo, discrete, legs)

    return BaselineComparison(
        scenario_id=head.scenario_id,
        instance_name=head.instance_name,
        family=head.family,
        n=head.n,
        grid_k=head.grid_k,
        legs=legs,
        delta_adaptation=delta_adaptation,
        delta_discretization=delta_discretization,
        delta_algorithm=delta_algorithm,
        provisional_vs_feasible_incumbent=provisional,
        notes=notes,
    )


def _build_notes(
    original: LegSummary | None,
    pyomo: LegSummary | None,
    discrete: LegSummary | None,
    legs: dict[str, LegSummary] | None = None,
) -> list[str]:
    notes: list[str] = []
    if original is not None and original.is_optimal_anchor:
        solver = original.ampl_solver or SUPERVISED_AMPL_SOLVER
        notes.append(f"official continuous anchor = original_ampl via {solver}")
    # A separate non-supervised AMPL validation leg (e.g. couenne), if present.
    for key, leg in (legs or {}).items():
        if key.startswith("original_ampl_"):
            notes.append(
                f"{key} present as an OPTIONAL validation leg ({leg.ampl_solver}) — "
                "not the official anchor"
            )
    if original is None:
        notes.append(
            "original_ampl (supervised gurobi) leg absent — historical continuous "
            "baseline not present"
        )
    elif original.status == "unavailable":
        notes.append(
            "original_ampl UNAVAILABLE — AMPL/supervised solver not installed; "
            "no continuous reference optimum"
        )
    elif original.is_optimal_anchor:
        pass  # proven continuous optimum: official continuous deltas are valid
    elif original.is_incumbent:
        notes.append(
            "original_ampl is a FEASIBLE incumbent, NOT a proven continuous optimum; "
            "official adaptation/discretisation deltas are null "
            "(see provisional_vs_feasible_incumbent)"
        )
    else:
        notes.append(f"original_ampl not usable as an anchor (status={original.status})")
    if original is not None and original.integrity_ok is False:
        notes.append("original_ampl provider/rescored objective diverged — untrustworthy")
    if pyomo is not None and pyomo.integrity_ok is False:
        notes.append("pyomo_minlp provider/rescored objective diverged")
    if discrete is not None and discrete.is_incumbent and not discrete.is_optimal_anchor:
        notes.append(
            "discrete_reference is a FEASIBLE heuristic incumbent, NOT a proven "
            "optimum; official delta_algorithm is null "
            "(see provisional_vs_feasible_incumbent)"
        )
    return notes


# --------------------------------------------------------------------------- #
# Serialisation (separate artifact; does not touch the per-record schema)
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    from dataclasses import is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    return obj


def write_baseline_json(comparisons: list[BaselineComparison], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([_jsonable(c) for c in comparisons], indent=2), encoding="utf-8"
    )
    return p


_BASELINE_COLUMNS = [
    "scenario_id",
    "instance_name",
    "family",
    "n",
    "grid_k",
    "original_ampl_objective",
    "original_ampl_status",
    "original_ampl_solver",
    "original_ampl_optimal",
    "pyomo_minlp_objective",
    "pyomo_minlp_status",
    "discrete_reference_objective",
    "discrete_reference_status",
    "discrete_reference_optimal",
    "delta_adaptation",
    "adaptation_ratio",
    "delta_discretization",
    "discretization_ratio",
    "best_algorithm_kind",
    "best_algorithm_objective",
    "delta_algorithm",
    "algorithm_ratio",
    "n_provisional",
    "notes",
]


def _num(v: float | None) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return ""
    return v


def _best_algorithm(c: BaselineComparison) -> Delta | None:
    """The approximate leg with the smallest finite delta (closest to reference)."""
    scored = [d for d in c.delta_algorithm if d.delta is not None]
    return min(scored, key=lambda d: d.delta) if scored else None  # type: ignore[arg-type,return-value]


def write_baseline_csv(comparisons: list[BaselineComparison], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_BASELINE_COLUMNS)
        writer.writeheader()
        for c in comparisons:
            o = c.legs.get("original_ampl")
            p_leg = c.legs.get("classical_minlp")
            d = c.legs.get("classical_discrete")
            best = _best_algorithm(c)
            writer.writerow(
                {
                    "scenario_id": c.scenario_id,
                    "instance_name": c.instance_name,
                    "family": c.family,
                    "n": c.n,
                    "grid_k": c.grid_k,
                    "original_ampl_objective": _num(o.objective) if o else "",
                    "original_ampl_status": o.status if o else "absent",
                    "original_ampl_solver": (o.ampl_solver or "") if o else "",
                    "original_ampl_optimal": bool(o.is_optimal_anchor) if o else "",
                    "pyomo_minlp_objective": _num(p_leg.objective) if p_leg else "",
                    "pyomo_minlp_status": p_leg.status if p_leg else "absent",
                    "discrete_reference_objective": _num(d.objective) if d else "",
                    "discrete_reference_status": d.status if d else "absent",
                    "discrete_reference_optimal": bool(d.is_optimal_anchor) if d else "",
                    "delta_adaptation": _num(c.delta_adaptation.delta),
                    "adaptation_ratio": _num(c.delta_adaptation.ratio),
                    "delta_discretization": _num(c.delta_discretization.delta),
                    "discretization_ratio": _num(c.delta_discretization.ratio),
                    "best_algorithm_kind": best.value if best else "",
                    "best_algorithm_objective": (
                        _num(c.legs[best.value].objective)
                        if best and best.value in c.legs
                        else ""
                    ),
                    "delta_algorithm": _num(best.delta) if best else "",
                    "algorithm_ratio": _num(best.ratio) if best else "",
                    "n_provisional": len(c.provisional_vs_feasible_incumbent),
                    "notes": "; ".join(c.notes),
                }
            )
    return p
