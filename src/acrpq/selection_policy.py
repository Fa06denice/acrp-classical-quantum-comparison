"""Versioned, deterministic tie-break policy for equally-scored candidates.

Why this exists (MoE counter-review, Expert A + the closure mission): the one-hot
*repair* in :func:`acrpq.quantum.decode.decode_assignment` had to choose among several
set bits, and it broke ties on the **quadratic** grid cost regardless of the QUBO's
``objective_id``. At the default ``w = 0.5`` the NOOP option is the unique zero-cost
option under both objectives, so the choice was accidentally right; at ``w = 0`` the
quadratic cost degenerates to ``(1 - q)**2`` and every ``q = 1`` option ties at zero, so
the repair could return a **non-NOOP** option and inflate the deviated-aircraft count.
``w`` is settable through the dashboard API, so this was reachable.

Contract (deliberately narrow — this is a *selection* rule, never an objective):

1. **Optimality is defined exclusively by the primary objective.** This module never
   changes an objective value, and never makes a solution "better".
2. It only orders candidates that the primary objective already considers **equal**.
3. It is **deterministic**: the order never depends on dict/set iteration order.
4. It is **published** as ``selection_policy`` (id + ordered criteria) in artefacts and
   exports, never folded silently into a cost.
5. No hidden epsilon: comparisons use exact equality on the primary objective, and the
   tie-break keys are exact too.

Ordered criteria of :data:`SELECTION_POLICY_ID`:

1. feasibility first (a feasible candidate always precedes an infeasible one);
2. fewest deviated aircraft (options that are NOOP win — this is the criterion the
   quadratic cost failed to express at ``w in {0, 1}``);
3. smallest unweighted total variation ``|theta| + |1 - q|`` (scale-free, so it does not
   re-introduce the weighting of the historical objective);
4. canonical order: lowest option index / lexicographically smallest bitstring.
"""
from __future__ import annotations

from typing import Any

SELECTION_POLICY_ID = "acrpq-selection-policy/1"
SELECTION_POLICY_CRITERIA: tuple[str, ...] = (
    "feasible_first",
    "fewest_deviated_aircraft",
    "smallest_unweighted_total_variation",
    "canonical_lowest_option_index",
)


def selection_policy_doc() -> dict[str, Any]:
    """JSON-native description to publish alongside any decoded result."""
    return {
        "selection_policy_id": SELECTION_POLICY_ID,
        "criteria": list(SELECTION_POLICY_CRITERIA),
        "note": (
            "Tie-break only. Optimality is defined solely by the primary objective; this "
            "policy orders candidates the objective already scores equally and never "
            "changes an objective value."
        ),
    }


def option_selection_key(grid: Any, option_index: int) -> tuple:
    """Deterministic tie-break key for one grid option (lower is preferred).

    Applied ONLY among options the primary objective scores identically.
    """
    opt = grid.options[option_index]
    deviated = 0 if opt.is_noop else 1
    variation = abs(float(opt.theta)) + abs(1.0 - float(opt.q))
    # Honesty note (verified by mutation testing): `deviated` is REDUNDANT with
    # `variation` on every grid this repo builds, because NOOP is exactly the option
    # with (q, theta) == (1, 0) and is therefore the unique zero-variation option.
    # Neutralising `deviated` alone changes no outcome (an equivalent mutant), while
    # neutralising `variation` breaks 3 tests. It is kept because it states criterion 2
    # of the published contract explicitly and would remain correct if a future grid
    # ever offered a non-NOOP option at zero variation.
    return (deviated, variation, option_index)


def choose_option(grid: Any, option_indices: Any) -> int:
    """Pick one option among ``option_indices`` under the versioned policy.

    ``option_indices`` may be any iterable (including a set or dict view): the result is
    independent of its iteration order because the key ends in the option index.
    """
    candidates = sorted(int(o) for o in option_indices)
    if not candidates:
        raise ValueError("choose_option needs at least one candidate option")
    return min(candidates, key=lambda o: option_selection_key(grid, o))
