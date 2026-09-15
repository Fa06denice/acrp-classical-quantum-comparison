"""Auditable explanations for the maneuver QUBO.

The functions in this module are deliberately pure Python.  They derive every
number from :class:`ManeuverQUBO` itself, so the dashboard cannot drift into a
second, presentation-only interpretation of the optimisation model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..objectives import option_cost_vector
from .memguard import max_safe_qubits, safe_budget_bytes, statevector_bytes
from .qubo import ManeuverQUBO, _as_bit_dict

_GIB = 1024**3
# Past ~60 qubits no host can hold the statevector, so the exact ``2**N``
# integers only bloat the JSON payload (and can overflow ``float`` past ~1050).
# Report them exactly inside the simulable range and ``None`` beyond it; the
# ``*_gib`` floats stay usable much further out for the "does it fit?" verdict.
_MAX_EXACT_QUBITS = 63


def _exact_amplitudes(n: int) -> int | None:
    """``2**n`` while it is a sane JSON integer, else ``None`` (unbounded)."""
    return 2**n if n <= _MAX_EXACT_QUBITS else None


def _exact_bytes(n: int) -> int | None:
    """Exact statevector byte count inside the simulable range, else ``None``."""
    return statevector_bytes(n) if n <= _MAX_EXACT_QUBITS else None


def _gib(n: int) -> float | None:
    """Statevector size in GiB as a compact float; ``None`` only past overflow."""
    exp = n - 26  # 2**n * 16 / 2**30
    return 2.0**exp if exp <= 1023 else None


def qubo_components(qubo: ManeuverQUBO) -> list[tuple[int, ...]]:
    """Return connected bit components of the non-zero QUBO interaction graph.

    Same-aircraft one-hot couplings connect every option block.  Consequently,
    distinct returned components are mathematically independent sub-QUBOs and
    may be optimised separately without changing the global optimum.
    """
    neighbours: list[set[int]] = [set() for _ in range(qubo.n_qubits)]
    for (a, b), coeff in qubo.quadratic.items():
        if not math.isclose(coeff, 0.0, abs_tol=1e-12):
            neighbours[a].add(b)
            neighbours[b].add(a)

    unseen = set(range(qubo.n_qubits))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: list[int] = []
        while stack:
            bit = stack.pop()
            component.append(bit)
            linked = neighbours[bit] & unseen
            unseen.difference_update(linked)
            stack.extend(linked)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda c: (-len(c), c))


def split_qubo_components(
    qubo: ManeuverQUBO,
) -> list[tuple[tuple[int, ...], ManeuverQUBO]]:
    """Build local-index sub-QUBOs for exact disconnected decomposition.

    The original constant is intentionally excluded from every component.  For
    any assignment, ``original.energy == original.constant + sum(local.energy)``.
    """
    split = []
    for component in qubo_components(qubo):
        old_to_new = {old: new for new, old in enumerate(component)}
        local_linear = {
            old_to_new[old]: coeff
            for old, coeff in qubo.linear.items()
            if old in old_to_new
        }
        local_quadratic = {
            (old_to_new[a], old_to_new[b]): coeff
            for (a, b), coeff in qubo.quadratic.items()
            if a in old_to_new and b in old_to_new
        }
        local = ManeuverQUBO(
            discrete=qubo.discrete,
            linear=local_linear,
            quadratic=local_quadratic,
            constant=0.0,
            n_qubits=len(component),
            penalty_onehot=qubo.penalty_onehot,
            penalty_conflict=qubo.penalty_conflict,
            meta={"parent_n_qubits": float(qubo.n_qubits)},
            objective_id=qubo.objective_id,  # components inherit the parent objective
        )
        split.append((component, local))
    return split


def resource_analysis(qubo: ManeuverQUBO) -> dict[str, Any]:
    """Explain raw and exactly decomposed statevector resource requirements."""
    components = qubo_components(qubo)
    d = qubo.discrete
    component_rows = []
    for idx, bits in enumerate(components, 1):
        aircraft = sorted({d.inv_index(bit)[0] for bit in bits})
        size = len(bits)
        component_rows.append(
            {
                "id": idx,
                "aircraft": aircraft,
                "n_aircraft": len(aircraft),
                "n_qubits": size,
                "statevector_bytes": _exact_bytes(size),
                "statevector_gib": _gib(size),
            }
        )

    largest = max((len(c) for c in components), default=0)
    reduction_log2 = max(0, qubo.n_qubits - largest)
    reduction_factor = 2.0**reduction_log2 if reduction_log2 <= 1023 else None
    safe_q = max_safe_qubits()
    return {
        "raw_qubits": qubo.n_qubits,
        "raw_amplitudes": _exact_amplitudes(qubo.n_qubits),
        "raw_statevector_bytes": _exact_bytes(qubo.n_qubits),
        "raw_statevector_gib": _gib(qubo.n_qubits),
        "safe_budget_bytes": safe_budget_bytes(),
        "safe_budget_gib": safe_budget_bytes() / _GIB,
        "max_safe_qubits": safe_q,
        "raw_fits_statevector": qubo.n_qubits <= safe_q,
        "n_exact_components": len(components),
        "largest_component_qubits": largest,
        "decomposed_peak_bytes": _exact_bytes(largest),
        "decomposed_peak_gib": _gib(largest),
        "decomposed_fits_statevector": largest <= safe_q,
        "exact_decomposition_available": len(components) > 1,
        "memory_reduction_factor": reduction_factor,
        "memory_reduction_log2": reduction_log2,
        "components": component_rows,
        "note": (
            "Components are disconnected in the actual QUBO interaction graph; "
            "separate minimisation preserves the global optimum exactly."
        ),
    }


def energy_breakdown(
    qubo: ManeuverQUBO, bits: Mapping[int, int] | Sequence[int] | str
) -> dict[str, Any]:
    """Decompose ``H(x)`` into its three documented physical/model terms.

    The objective term uses the QUBO's OWN objective (``qubo.objective_id``), so
    the identity ``objective_cost + penalties == qubo.energy`` holds for both
    objectives — not only the quadratic one.
    """
    x = _as_bit_dict(bits, qubo.n_qubits)
    d = qubo.discrete
    cost_vec = option_cost_vector(qubo.objective_id, d.grid, d.instance.w)

    objective_cost = 0.0
    active_by_aircraft: dict[int, list[int]] = {}
    for i in d.instance.aircraft():
        active = [
            o for o in range(d.k)
            if x.get(d.var_index(i, o), 0)
        ]
        active_by_aircraft[i] = active
        objective_cost += sum(cost_vec[o] for o in active)

    conflict_activations = 0
    active_conflicts: list[dict[str, int]] = []
    for (i, j), matrix in d.conflict.items():
        for oi in active_by_aircraft[i]:
            for oj in active_by_aircraft[j]:
                if matrix[oi][oj]:
                    conflict_activations += 1
                    active_conflicts.append({"i": i, "oi": oi, "j": j, "oj": oj})

    onehot_residual = sum(
        (len(active_by_aircraft[i]) - 1) ** 2
        for i in d.instance.aircraft()
    )
    conflict_penalty = qubo.penalty_conflict * conflict_activations
    onehot_penalty = qubo.penalty_onehot * onehot_residual
    total = objective_cost + conflict_penalty + onehot_penalty
    encoded = qubo.energy(bits)
    return {
        "objective_id": qubo.objective_id,
        "objective_cost": objective_cost,
        # backward-compatible alias (now objective-aware, not always quadratic)
        "maneuver_cost": objective_cost,
        "conflict_activations": conflict_activations,
        "conflict_penalty": conflict_penalty,
        "onehot_residual": onehot_residual,
        "onehot_penalty": onehot_penalty,
        "total": total,
        "encoded_total": encoded,
        "identity_verified": math.isclose(total, encoded, rel_tol=1e-9, abs_tol=1e-7),
        "active_conflicts": active_conflicts,
    }


def bitstring_explanation(
    qubo: ManeuverQUBO, bits: Mapping[int, int] | Sequence[int] | str
) -> dict[str, Any]:
    """Map a raw bit assignment to aircraft/options without hiding repairs."""
    from .decode import decode_assignment

    x = _as_bit_dict(bits, qubo.n_qubits)
    d = qubo.discrete
    cost_vec = option_cost_vector(qubo.objective_id, d.grid, d.instance.w)
    repaired = decode_assignment(qubo, bits)
    rows = []
    raw_onehot = True
    for i in d.instance.aircraft():
        active = [o for o in range(d.k) if x.get(d.var_index(i, o), 0)]
        valid = len(active) == 1
        raw_onehot &= valid
        chosen = repaired[i]
        option = d.grid.options[chosen]
        rows.append(
            {
                "aircraft": i,
                "bit_range": [d.var_index(i, 0), d.var_index(i, d.k - 1)],
                "active_options": active,
                "raw_onehot_valid": valid,
                "decoded_option": chosen,
                "decoded_label": option.label,
                "q": option.q,
                "theta": option.theta,
                "cost": cost_vec[chosen],
                "repair": (
                    None if valid
                    else "NOOP fallback" if not active
                    else "cheapest active option"
                ),
            }
        )
    choice = tuple(repaired[i] for i in d.instance.aircraft())
    return {
        "objective_id": qubo.objective_id,
        "bitstring": "".join(str(x.get(b, 0)) for b in range(qubo.n_qubits)),
        "selected_bits": sorted(x),
        "raw_onehot_valid": raw_onehot,
        "repair_applied": not raw_onehot,
        "decoded_choice": list(choice),
        "decoded_conflicts": d.assignment_conflicts(choice),
        "decoded_feasible": d.assignment_conflicts(choice) == 0,
        "aircraft": rows,
    }


def sample_explanations(
    qubo: ManeuverQUBO, counts: Mapping[str, int], *, limit: int = 10
) -> list[dict[str, Any]]:
    """Summarise the most probable QAOA/annealer samples, bounded for JSON/UI."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be an integer >= 1")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts.values()):
        raise ValueError("sample counts must be non-negative integers")
    total = sum(counts.values())
    if total == 0:
        return []
    rows = []
    for bitstring, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        decoded = bitstring_explanation(qubo, bitstring)
        breakdown = energy_breakdown(qubo, bitstring)
        rows.append(
            {
                "bitstring": bitstring,
                "count": count,
                "probability": count / total,
                "energy": breakdown["total"],
                "raw_onehot_valid": decoded["raw_onehot_valid"],
                "decoded_feasible": decoded["decoded_feasible"],
                "decoded_conflicts": decoded["decoded_conflicts"],
                "decoded_choice": decoded["decoded_choice"],
            }
        )
    return rows


def coefficient_explanation(qubo: ManeuverQUBO) -> dict[str, Any]:
    """Return the semantic origin of every displayed QUBO coefficient."""
    d = qubo.discrete
    cost_vec = option_cost_vector(qubo.objective_id, d.grid, d.instance.w)
    linear = []
    for bit, coeff in sorted(qubo.linear.items()):
        i, o = d.inv_index(bit)
        option = d.grid.options[o]
        linear.append(
            {
                "bit": bit,
                "aircraft": i,
                "option": o,
                "label": option.label,
                "coefficient": coeff,
                "objective_cost": cost_vec[o],
                # backward-compatible alias (now objective-aware)
                "maneuver_cost": cost_vec[o],
                "onehot_linear": -qubo.penalty_onehot,
                "formula": "objective_cost - lambda_onehot",
            }
        )

    quadratic = []
    for (a, b), coeff in sorted(qubo.quadratic.items()):
        ia, oa = d.inv_index(a)
        ib, ob = d.inv_index(b)
        same_aircraft = ia == ib
        quadratic.append(
            {
                "a": a,
                "b": b,
                "aircraft_a": ia,
                "option_a": oa,
                "aircraft_b": ib,
                "option_b": ob,
                "coefficient": coeff,
                "kind": "onehot" if same_aircraft else "conflict",
                "formula": (
                    "2 * lambda_onehot" if same_aircraft else "lambda_conflict"
                ),
            }
        )
    return {"objective_id": qubo.objective_id, "linear": linear, "quadratic": quadratic}
