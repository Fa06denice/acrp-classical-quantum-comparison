"""EXPERIMENTAL — instance-scalability analysis of the conflict graph and its
QUBO decomposition, for ALL 1042 library instances.

Status: ``experimental = True``, ``official_benchmark = False``, ``real_qpu =
False`` on every artefact this module or its scripts produce.

This module is ISOLATED: it imports only read-only helpers from ``acrpq``
(``geometry``, ``model``, ``objectives``) plus the sibling experimental
``adaptive_grid``/``adaptive_qpu`` modules (``LocalGrid``, ``global_grid``,
``official_k_grid_options``, ``conflict_table``, ``exact_local_solve``,
``build_hetero_qubo``, ``build_fixed_angle_circuit``, ``transpile_isa``). It
never imports the official ``quantum/qubo.py`` and builds no official QUBO.

Method summary (see ``docs/COMPACT_ENCODING_STUDY.md`` §13 and
``results/experimental_adaptive_grid_v1/moe_reports/expert_D_decomposition.md``
for the theory this reuses):

* the "any-option" conflict graph ``G_K`` has an edge ``(i, j)`` iff SOME pair
  of options (one per aircraft, drawn from a K-level grid) conflicts. Two
  aircraft in different connected components of ``G_K`` never interact in the
  QUBO built on that grid: the QUBO factorises exactly and each component can
  be solved independently without loss of optimality (expert D, candidate 6).
* the "NOOP-only" / initial graph is the special case ``K=1`` (theta=0 for
  everyone) and is NEVER used to decide a decomposition (expert D: it can
  under-estimate the real coupling).
* an isolated aircraft in the K3 any-option graph can be fixed to NOOP: NOOP
  costs 0 (a lower bound on cost) and, having no edges, participates in no
  conflict term whatever it or anyone else does — dropping it from the QUBO
  loses nothing.
* the heterogeneous-block width keeps every isolated aircraft at ``K_i = 1``
  (forced NOOP) and applies the expert-D EXACT dominance rule (candidate 4) to
  every other aircraft, PER OBJECTIVE (the dominance test uses the option
  cost, which is objective-dependent).

Nothing here claims heuristic decompositions preserve optimality; only the
``exact_*``-labelled quantities do (proven in the report above).
"""

from __future__ import annotations

import math
import signal
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .. import geometry
from ..model import Instance
from ..objectives import ObjectiveId, coerce_objective_id, primary_objective
from . import adaptive_grid as ag
from . import adaptive_qpu as aq

EXPERIMENTAL = True
OFFICIAL_BENCHMARK = False
REAL_QPU = False
ARTIFACT_FLAGS = {"experimental": True, "official_benchmark": False, "real_qpu": False}

# Fixed heuristic neighbourhood widths (LNS-style destroy sets on a K3 grid) —
# constants, independent of any instance; NEVER claimed exact.
HEURISTIC_NEIGHBOURHOOD_SIZES = (3, 5, 7)
HEURISTIC_NEIGHBOURHOOD_K = 3

# Hardware-pilot plausibility thresholds, calibrated on the real campaign
# (docs/ADAPTIVE_QPU_GO_NO_GO.md §3): CP3/CP4 (9/12 logical qubits) gave 9.5%/1.0%
# feasible shots; CP5 (15 qubits) gave 0%. 2Q gate counts from the same table
# (CP3 108, CP4 207/414 cumulative, CP5-CP7 grow to 700).
HARDWARE_PILOT_MAX_LOGICAL = 12
HARDWARE_PILOT_MAX_2Q = 210
HARDWARE_IMPLAUSIBLE_MIN_LOGICAL = 15
HARDWARE_IMPLAUSIBLE_MIN_2Q = 350
AER_STATEVECTOR_MAX_Q = 30
ISA_COMPILABLE_MAX_LOGICAL = 156  # FakeMarrakesh / ibm_marrakesh qubit count


# --------------------------------------------------------------------------- #
# Conflict graphs
# --------------------------------------------------------------------------- #
def any_option_graph(inst: Instance, levels: Sequence[float]) -> tuple[tuple[int, int], ...]:
    """Edges ``(i, j)``, ``i < j``, iff SOME pair of options in ``levels`` conflicts.

    ``levels`` is a single global list of heading options offered to every
    aircraft (``q = 1`` fixed, matching the rest of this experimental study).
    For the official K=3 grid, pass ``adaptive_grid.official_k_grid_options(inst, 3)``.
    """
    if not levels:
        raise ValueError("levels must not be empty")
    edges = []
    for i, j in inst.pairs():
        conflict = False
        for ti in levels:
            for tj in levels:
                if geometry.in_conflict(inst, i, j, 1.0, ti, 1.0, tj):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            edges.append((i, j))
    return tuple(edges)


def initial_conflict_graph(inst: Instance) -> tuple[tuple[int, int], ...]:
    """NOOP-NOOP conflict graph (``theta = 0`` for everyone). Diagnostic only —
    never a valid basis for a solver decomposition (expert D)."""
    return any_option_graph(inst, (0.0,))


# --------------------------------------------------------------------------- #
# Graph algorithms (pure, 1-based aircraft ids 1..n)
# --------------------------------------------------------------------------- #
def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """Connected components of the undirected graph on ``{1..n}``.

    Returns tuples of aircraft ids, sorted by DECREASING size then by their
    smallest member (deterministic).
    """
    parent = list(range(n + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j in edges:
        union(i, j)
    groups: dict[int, list[int]] = {}
    for v in range(1, n + 1):
        groups.setdefault(find(v), []).append(v)
    comps = [tuple(sorted(g)) for g in groups.values()]
    comps.sort(key=lambda c: (-len(c), c[0]))
    return tuple(comps)


def degrees(n: int, edges: Iterable[tuple[int, int]]) -> dict[int, int]:
    deg = {v: 0 for v in range(1, n + 1)}
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    return deg


def density(n: int, n_edges: int) -> float:
    max_edges = n * (n - 1) // 2
    return 0.0 if max_edges == 0 else n_edges / max_edges


def _adjacency(n: int, edges: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {v: [] for v in range(1, n + 1)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def articulation_points_and_bridges(
    n: int, edges: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Tarjan's DFS articulation points / bridges (handles disconnected graphs)."""
    adj = _adjacency(n, edges)
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    ap: set[int] = set()
    bridges: list[tuple[int, int]] = []
    timer = [0]

    def dfs(root: int) -> None:
        stack = [(root, iter(adj[root]))]
        parent[root] = None
        disc[root] = low[root] = timer[0]
        timer[0] += 1
        child_count: dict[int, int] = {root: 0}
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v not in disc:
                    parent[v] = u
                    disc[v] = low[v] = timer[0]
                    timer[0] += 1
                    child_count[u] = child_count.get(u, 0) + 1
                    child_count[v] = 0
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] > disc[p]:
                    a, b = (p, u) if p < u else (u, p)
                    bridges.append((a, b))
                if parent[p] is not None and low[u] >= disc[p]:
                    ap.add(p)
                elif parent[p] is None and child_count.get(p, 0) > 1:
                    ap.add(p)

    for v in range(1, n + 1):
        if v not in disc:
            dfs(v)
    return tuple(sorted(ap)), tuple(sorted(bridges))


def structure_label(n: int, n_edges: int, n_components: int, dens: float, degs: dict[int, int]) -> str:
    """One of clique / quasi-clique / chain / clustered / sparse (best-effort)."""
    if n <= 1:
        return "trivial"
    if dens >= 1.0 - 1e-12:
        return "clique"
    if dens >= 0.8:
        return "quasi-clique"
    max_deg = max(degs.values(), default=0)
    if n_components == 1 and max_deg <= 2 and n_edges == n - 1:
        return "chain"
    if n_components > 1:
        return "clustered"
    return "sparse"


# --------------------------------------------------------------------------- #
# Dominance (expert D, candidate 4) on the official K3 grid, per objective
# --------------------------------------------------------------------------- #
def dominance_survivors(
    inst: Instance, objective_id: ObjectiveId | str, k3_levels: Sequence[float],
    isolated: set[int], table: dict[tuple[int, int], tuple[tuple[bool, ...], ...]] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Per-aircraft surviving PHYSICAL heading options: ``(0.0,)`` if isolated
    (forced NOOP), else the K3 options surviving the EXACT dominance rule
    (option ``a`` dominates ``a'`` iff ``cost_a <= cost_a'`` and, for every
    neighbour, ``conflict_row[a] subseteq conflict_row[a']``). Never removes an
    optimum (expert D, candidate 4, proof by exchange argument)."""
    oid = coerce_objective_id(objective_id)
    n = inst.n
    k = len(k3_levels)
    cost = [primary_objective(oid, (1.0,), (t,), inst.w) for t in k3_levels]
    neighbours: dict[int, list[int]] = {i: [] for i in range(1, n + 1)}
    for i, j in inst.pairs():
        neighbours[i].append(j)
        neighbours[j].append(i)
    if table is None:
        grid = ag.LocalGrid(centers=(0.0,) * n, delta=math.nan, include_noop=0.0 in k3_levels,
                            options=tuple(tuple(k3_levels) for _ in range(n)))
        table = ag.conflict_table(inst, grid)

    def conflict_row(i: int, a: int, j: int) -> tuple[bool, ...]:
        if i < j:
            return table[(i, j)][a]
        return tuple(table[(j, i)][b][a] for b in range(k))

    def _row_subset(row_a: tuple[bool, ...], row_ap: tuple[bool, ...]) -> bool:
        """True iff every option flagged in ``row_a`` is also flagged in ``row_ap``."""
        return all((not ba) or bb for ba, bb in zip(row_a, row_ap))

    survivors: list[tuple[float, ...]] = []
    for i in range(1, n + 1):
        if i in isolated:
            survivors.append((0.0,))
            continue
        dominated: set[int] = set()
        # rows[a][t] = boolean row (over neighbour t's k options) for aircraft i's option a
        rows = {a: tuple(conflict_row(i, a, j) for j in neighbours[i]) for a in range(k)}
        for a in range(k):
            if a in dominated:
                continue
            for ap in range(k):
                if a == ap or ap in dominated:
                    continue
                if cost[a] > cost[ap]:
                    continue
                subset_everywhere = all(
                    _row_subset(rows[a][t], rows[ap][t]) for t in range(len(neighbours[i]))
                )
                if not subset_everywhere:
                    continue
                if cost[a] == cost[ap] and all(
                    _row_subset(rows[ap][t], rows[a][t]) for t in range(len(neighbours[i]))
                ):
                    # fully interchangeable (equal cost, equal conflict rows both ways):
                    # keep only the smaller index by convention.
                    if ap > a:
                        dominated.add(ap)
                    continue
                # cost[a] <= cost[ap] and subset_everywhere, and not the tie above:
                # a strictly dominates ap.
                dominated.add(ap)
        survivors.append(tuple(sorted(k3_levels[a] for a in range(k) if a not in dominated)))
    return tuple(survivors)


def dominance_survivor_counts(
    inst: Instance, objective_id: ObjectiveId | str, k3_levels: Sequence[float],
    isolated: set[int], table: dict[tuple[int, int], tuple[tuple[bool, ...], ...]] | None = None,
) -> tuple[int, ...]:
    """``K_i`` per aircraft (see :func:`dominance_survivors` for the rule)."""
    return tuple(len(opts) for opts in dominance_survivors(inst, objective_id, k3_levels, isolated, table))


def hetero_exact_grid(
    inst: Instance, objective_id: ObjectiveId | str, k3_levels: Sequence[float], isolated: set[int],
    table: dict[tuple[int, int], tuple[tuple[bool, ...], ...]] | None = None,
) -> "ag.LocalGrid":
    """The heterogeneous-block :class:`~acrpq.experimental.adaptive_grid.LocalGrid`
    obtained by keeping only the dominance survivors of every aircraft (isolated
    aircraft forced to the single NOOP option)."""
    opts = dominance_survivors(inst, objective_id, k3_levels, isolated, table)
    return ag.LocalGrid(centers=(0.0,) * inst.n, delta=math.nan, include_noop=True, options=opts)


# --------------------------------------------------------------------------- #
# Resource proxies
# --------------------------------------------------------------------------- #
def cross_component_qubo_terms(qubo: "aq.HeteroQUBO", components: Sequence[tuple[int, ...]]) -> int:
    """Number of quadratic QUBO terms whose two bits belong to DIFFERENT
    aircraft-components. Must be 0 for a valid exact decomposition."""
    comp_of: dict[int, int] = {}
    for ci, comp in enumerate(components):
        for a in comp:
            comp_of[a] = ci
    n_cross = 0
    for (b1, b2) in qubo.quadratic:
        i1, _ = qubo.inv_index(b1)
        i2, _ = qubo.inv_index(b2)
        if comp_of[i1] != comp_of[i2]:
            n_cross += 1
    return n_cross


def max_interaction_degree(qubo: "aq.HeteroQUBO") -> int:
    deg: dict[int, int] = {}
    for (b1, b2) in qubo.quadratic:
        deg[b1] = deg.get(b1, 0) + 1
        deg[b2] = deg.get(b2, 0) + 1
    return max(deg.values(), default=0)


def statevector_bytes(n_qubits: int) -> int:
    """16 bytes/amplitude (complex128) * 2**n_qubits statevector entries."""
    return 16 * (2**n_qubits)


def statevector_practicality(n_qubits: int) -> str:
    if n_qubits <= AER_STATEVECTOR_MAX_Q:
        return "feasible"
    if n_qubits <= 36:
        return "large"
    return "infeasible"


class _TimeoutError(RuntimeError):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _TimeoutError("wall-clock timeout")


def local_exact_feasible(
    inst: Instance, objective_id: ObjectiveId | str, grid: ag.LocalGrid, *,
    timeout_s: int = 60, max_states: int = 2_000_000,
) -> tuple[bool, str]:
    """Whether ``adaptive_grid.exact_local_solve`` completes on ``grid`` within
    ``timeout_s`` (SIGALRM) and ``max_states`` (search-space guard). Returns
    ``(feasible_within_budget, reason)``; ``reason`` explains a False result.

    SIGALRM only works on the main thread of a POSIX process; callers running
    this off the main thread must catch the resulting ``ValueError`` themselves
    (documented, not silently swallowed here).
    """
    if grid.search_space() > max_states:
        return False, f"search_space {grid.search_space()} exceeds max_states={max_states}"
    old_handler = None
    have_alarm = hasattr(signal, "SIGALRM")
    try:
        if have_alarm:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_s)
        ag.exact_local_solve(inst, objective_id, grid, max_states=max_states)
        return True, "ok"
    except _TimeoutError:
        return False, f"timeout after {timeout_s}s"
    finally:
        if have_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
CLASSES = (
    "DECOMPOSABLE_FORMALLY",
    "WIDTH_REDUCED",
    "NEWLY_LOCAL_EXACT_ACCESSIBLE",
    "NEWLY_STATEVECTOR_ACCESSIBLE",
    "LOCAL_HEURISTIC_ONLY",
    "ISA_COMPILABLE_ONLY",
    "HARDWARE_PILOT_PLAUSIBLE",
    "HARDWARE_SCIENTIFICALLY_IMPLAUSIBLE",
)


@dataclass(frozen=True)
class ClassificationInput:
    n: int
    q_global_k3: int
    q_largest_component: int
    largest_component_size: int
    n_components: int
    cross_component_terms: int
    local_exact_feasible_global: bool
    local_exact_feasible_component: bool
    expected_isa_2q_component: int | None
    q_hetero_exact: int | None = None


def classify_instance(ci: ClassificationInput) -> list[str]:
    """Accessibility labels (schema v2). Vocabulary is deliberately narrow:

    * DECOMPOSABLE_FORMALLY — >= 2 components and 0 cross-component QUBO terms (K3 grid);
      it never means 'accessible'.
    * WIDTH_REDUCED — the largest component or the heterogeneous exact block count is
      strictly below 3n; a reduced width does not imply an exact global solve.
    * NEWLY_LOCAL_EXACT_ACCESSIBLE — the largest component is exactly solvable within the
      study's budget while the monolithic problem is not.
    * NEWLY_STATEVECTOR_ACCESSIBLE — largest component <= 30 logical qubits while 3n > 30.
    * LOCAL_HEURISTIC_ONLY — even the largest component is outside the exact budget.
    * ISA_COMPILABLE_ONLY — 30 < largest component <= 156 logical qubits; compilable in
      width, never 'exploitable'.
    * HARDWARE_PILOT_PLAUSIBLE — largest component <= 12 logical qubits AND a MEASURED
      offline ISA 2Q count <= 210 (regime where the real campaign produced feasible shots).
    * HARDWARE_SCIENTIFICALLY_IMPLAUSIBLE — >= 15 logical qubits or >= 350 2Q gates.

    Logical qubits only; physical qubits come solely from transpile layouts.
    """
    labels: list[str] = []
    if ci.n_components >= 2 and ci.cross_component_terms == 0:
        labels.append("DECOMPOSABLE_FORMALLY")
    reduced = ci.q_largest_component < ci.q_global_k3 or (
        ci.q_hetero_exact is not None and ci.q_hetero_exact < ci.q_global_k3)
    if reduced:
        labels.append("WIDTH_REDUCED")
    if ci.local_exact_feasible_component and not ci.local_exact_feasible_global:
        labels.append("NEWLY_LOCAL_EXACT_ACCESSIBLE")
    if ci.q_largest_component <= AER_STATEVECTOR_MAX_Q < ci.q_global_k3:
        labels.append("NEWLY_STATEVECTOR_ACCESSIBLE")
    if not ci.local_exact_feasible_component:
        labels.append("LOCAL_HEURISTIC_ONLY")
    if AER_STATEVECTOR_MAX_Q < ci.q_largest_component <= ISA_COMPILABLE_MAX_LOGICAL:
        labels.append("ISA_COMPILABLE_ONLY")
    plausible_2q = (ci.expected_isa_2q_component is not None
                    and ci.expected_isa_2q_component <= HARDWARE_PILOT_MAX_2Q)
    if ci.q_largest_component <= HARDWARE_PILOT_MAX_LOGICAL and plausible_2q:
        labels.append("HARDWARE_PILOT_PLAUSIBLE")
    if (ci.q_largest_component >= HARDWARE_IMPLAUSIBLE_MIN_LOGICAL
            or (ci.expected_isa_2q_component or 0) >= HARDWARE_IMPLAUSIBLE_MIN_2Q):
        labels.append("HARDWARE_SCIENTIFICALLY_IMPLAUSIBLE")
    return labels


__all__ = [
    "ARTIFACT_FLAGS", "CLASSES", "ClassificationInput",
    "HEURISTIC_NEIGHBOURHOOD_SIZES", "HEURISTIC_NEIGHBOURHOOD_K",
    "any_option_graph", "articulation_points_and_bridges", "classify_instance",
    "connected_components", "cross_component_qubo_terms", "degrees", "density",
    "dominance_survivor_counts", "dominance_survivors", "hetero_exact_grid",
    "initial_conflict_graph", "local_exact_feasible",
    "max_interaction_degree", "statevector_bytes", "statevector_practicality",
    "structure_label",
]
