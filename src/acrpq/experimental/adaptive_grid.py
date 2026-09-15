"""EXPERIMENTAL — adaptive iterative heading grid for the discrete ACRP.

Status: ``experimental = True``, ``official_benchmark = False``, ``real_qpu = False``.
This module is ISOLATED: it imports only read-only helpers from ``acrpq``
(instance model, geometry kernel, objectives) and is imported by nothing in the
official pipeline. It never builds an official ``ManeuverQUBO``.

Idea
----
Instead of one global one-hot grid with ``K`` heading levels shared by all
aircraft (``n*K`` qubits), each round ``r`` gives aircraft ``i`` a small LOCAL
grid centred on its current heading::

    Theta_i^(r) = clip{ c_i^(r) - Delta_r,  c_i^(r),  c_i^(r) + Delta_r }   (+ NOOP optionally)
    c_i^(r+1)   = theta_i^*(r)            (round-r solution)
    Delta_(r+1) = rho * Delta_r,  0 < rho < 1

so the one-hot register width stays ~``3n`` (``4n`` with the NOOP option kept)
per round while the implicit angular resolution shrinks geometrically.

Two DISTINCT warm-start notions (never confused here):

* warm-start of the CONTROLS / GRID: the initial centres ``c^(0)`` (NOOP, or a
  classical continuous solution). This is what this module does.
* warm-start of the QAOA STATE / ANGLES: not implemented, not modelled here.

Guarantees (see docs/COMPACT_ENCODING_STUDY.md):

* Round-0 with ``c = 0`` and ``Delta_0 = HMAX`` is EXACTLY the official K=3 grid.
* The centre ``c_i^(r+1) = theta_i^*(r)`` is always inside ``[HMIN, HMAX]`` so
  it is never clipped away: the previous solution is ALWAYS a point of the next
  grid (``previous_solution_in_grid`` is asserted and recorded per round).
* Hence with an EXACT round solver ``f_best^(r+1) <= f_best^(r)``. With a
  heuristic round solver only the elitist archive is monotone.
* The procedure is NOT exhaustive over the implicit fine grid: it may converge
  to a local optimum. Nothing here claims global optimality beyond round 0.

Speed is kept fixed at ``q = 1`` (heading-only model, matching the official
``n_q = 1`` grids).
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .. import geometry
from ..model import Instance
from ..objectives import ObjectiveId, coerce_objective_id, primary_objective

EXPERIMENTAL = True
OFFICIAL_BENCHMARK = False
REAL_QPU = False

ARTIFACT_FLAGS = {"experimental": True, "official_benchmark": False, "real_qpu": False}

NOOP_THETA = 0.0
_TIE_EPS = 0.0  # exact ties only; lexicographic tie-break on the choice tuple


# --------------------------------------------------------------------------- #
# Local (per-aircraft) heading grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LocalGrid:
    """One round's per-aircraft heading options (sorted, deduplicated, clipped)."""

    centers: tuple[float, ...]
    delta: float
    include_noop: bool
    options: tuple[tuple[float, ...], ...]  # options[i-1] = sorted thetas of aircraft i

    @property
    def n(self) -> int:
        return len(self.options)

    @property
    def n_qubits(self) -> int:
        """One-hot width = total number of options across aircraft."""
        return sum(len(o) for o in self.options)

    @property
    def max_k(self) -> int:
        return max((len(o) for o in self.options), default=0)

    def search_space(self) -> int:
        s = 1
        for o in self.options:
            s *= len(o)
        return s

    def contains(self, theta: Sequence[float]) -> bool:
        """True iff every ``theta[i]`` is an exact option of aircraft ``i``."""
        if len(theta) != self.n:
            return False
        return all(t in opts for t, opts in zip(theta, self.options))

    def thetas(self, choice: Sequence[int]) -> tuple[float, ...]:
        if len(choice) != self.n:
            raise ValueError("choice must have length n")
        out = []
        for opts, c in zip(self.options, choice):
            if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < len(opts):
                raise ValueError(f"invalid option index {c!r} for {len(opts)} options")
            out.append(opts[c])
        return tuple(out)

    def n_onehot_quadratic_terms(self) -> int:
        return sum(len(o) * (len(o) - 1) // 2 for o in self.options)


def _check_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def build_local_grid(
    inst: Instance,
    centers: Sequence[float],
    delta: float,
    *,
    include_noop: bool = False,
) -> LocalGrid:
    """Build ``{c-D, c, c+D}`` per aircraft, clipped to the instance heading box.

    Options equal after clipping are deduplicated (an aircraft may end up with
    fewer than 3 options). ``include_noop`` adds ``theta = 0`` to every aircraft
    (needed for ``maneuver_count_v1`` to be able to *un-move* an aircraft).
    """
    if len(centers) != inst.n:
        raise ValueError(f"centers must have length n={inst.n}, got {len(centers)}")
    delta = _check_finite(delta, "delta")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta!r}")
    lo, hi = inst.hmin, inst.hmax
    opts: list[tuple[float, ...]] = []
    cs: list[float] = []
    for c in centers:
        c = _check_finite(c, "center")
        if not lo <= c <= hi:
            raise ValueError(f"center {c!r} outside heading bounds [{lo}, {hi}]")
        cs.append(c)
        cand = {min(hi, max(lo, c - delta)), c, min(hi, max(lo, c + delta))}
        if include_noop:
            cand.add(NOOP_THETA)
        opts.append(tuple(sorted(cand)))
    return LocalGrid(centers=tuple(cs), delta=delta, include_noop=include_noop,
                     options=tuple(opts))


def official_k_grid_options(inst: Instance, k: int) -> tuple[float, ...]:
    """Heading levels of the official ``ManeuverGrid.build(n_theta=k, n_q=1)``.

    Re-derived here (linspace + snap nearest to 0) so the experimental module
    does not import the official grid builder; equality is asserted in tests.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1")
    lo, hi = inst.hmin, inst.hmax
    if k == 1:
        vals = [lo]
    else:
        step = (hi - lo) / (k - 1)
        vals = [lo + step * i for i in range(k)]
    j = min(range(len(vals)), key=lambda i: abs(vals[i]))
    vals[j] = 0.0
    return tuple(vals)


def global_grid(inst: Instance, k: int) -> LocalGrid:
    """The official global K-level grid expressed as a (uniform) LocalGrid."""
    levels = official_k_grid_options(inst, k)
    return LocalGrid(centers=(0.0,) * inst.n, delta=math.nan, include_noop=0.0 in levels,
                     options=tuple(levels for _ in range(inst.n)))


# --------------------------------------------------------------------------- #
# Conflict table and scoring (single geometry kernel, q fixed to 1)
# --------------------------------------------------------------------------- #
def conflict_table(inst: Instance, grid: LocalGrid) -> dict[tuple[int, int], tuple[tuple[bool, ...], ...]]:
    """``table[(i,j)][a][b]`` iff aircraft i option a and j option b conflict."""
    table: dict[tuple[int, int], tuple[tuple[bool, ...], ...]] = {}
    for i, j in inst.pairs():
        oi, oj = grid.options[i - 1], grid.options[j - 1]
        rows = []
        for ti in oi:
            rows.append(tuple(geometry.in_conflict(inst, i, j, 1.0, ti, 1.0, tj) for tj in oj))
        table[(i, j)] = tuple(rows)
    return table


def objective_value(inst: Instance, objective_id: ObjectiveId | str, theta: Sequence[float]) -> float:
    oid = coerce_objective_id(objective_id)
    q = (1.0,) * inst.n
    return primary_objective(oid, q, tuple(theta), inst.w)


def count_conflicts(inst: Instance, theta: Sequence[float]) -> int:
    return geometry.count_conflicts(inst, (1.0,) * inst.n, tuple(theta))


def n_conflict_terms(table) -> int:
    return sum(sum(1 for row in mat for v in row if v) for mat in table.values())


# --------------------------------------------------------------------------- #
# Round solvers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoundSolve:
    """Result of one local-grid solve. ``choice is None`` iff no feasible point."""

    solver: str
    choice: tuple[int, ...] | None
    theta: tuple[float, ...] | None
    objective: float | None
    n_evaluations: int
    exact: bool
    wall_time_s: float
    seed: int | None = None


def _per_option_cost(inst: Instance, oid: ObjectiveId, grid: LocalGrid) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(primary_objective(oid, (1.0,), (t,), inst.w) for t in opts) for opts in grid.options
    )


def exact_local_solve(inst: Instance, objective_id: ObjectiveId | str, grid: LocalGrid,
                      table=None, *, max_states: int = 5_000_000,
                      branch_order: str = "index") -> RoundSolve:
    """EXACT optimum over the local grid by branch-and-bound on conflicts + cost.

    Costs are non-negative and separable, conflicts pairwise, so a partial
    assignment with a conflict (or a partial cost >= incumbent) can be pruned.
    Ties are broken lexicographically on the choice tuple (deterministic, no
    epsilon). Equivalent to exhaustive enumeration (asserted in tests).

    ``branch_order="cost"`` explores cheap options first (much faster pruning on
    fine grids); the optimum VALUE is identical, only the tie representative may
    differ (documented, used for fine-grid references only).
    """
    t0 = time.perf_counter()
    oid = coerce_objective_id(objective_id)
    if grid.search_space() > max_states:
        raise ValueError(f"local search space {grid.search_space()} exceeds max_states={max_states}")
    if table is None:
        table = conflict_table(inst, grid)
    n = grid.n
    cost = _per_option_cost(inst, oid, grid)
    # pairs indexed by the later aircraft so we check conflicts incrementally
    later: dict[int, list[tuple[int, tuple[tuple[bool, ...], ...]]]] = {j: [] for j in range(1, n + 1)}
    for (i, j), mat in table.items():
        later[j].append((i, mat))
    best_choice: list[int] | None = None
    best_val = math.inf
    evals = 0
    choice = [0] * n
    if branch_order == "cost":
        def _cost_order(i: int) -> list[int]:
            return sorted(range(len(grid.options[i])), key=lambda a: (cost[i][a], a))
        orders = [_cost_order(i) for i in range(n)]
    elif branch_order == "index":
        orders = [list(range(len(grid.options[i]))) for i in range(n)]
    else:
        raise ValueError(f"unknown branch_order {branch_order!r}")

    def rec(i: int, partial: float) -> None:
        nonlocal best_choice, best_val, evals
        if i > n:
            evals += 1
            # strict improvement, or equal value with lexicographically smaller tuple
            if partial < best_val or (partial == best_val and best_choice is not None
                                      and choice < best_choice):
                best_val, best_choice = partial, list(choice)
            return
        for a in orders[i - 1]:
            val = partial + cost[i - 1][a]
            if val > best_val:
                if orders[i - 1][0] != a and branch_order == "cost":
                    break  # cost-sorted: every remaining option is at least as expensive
                continue
            ok = True
            for (h, mat) in later[i]:
                if mat[choice[h - 1]][a]:
                    ok = False
                    break
            if not ok:
                continue
            choice[i - 1] = a
            rec(i + 1, val)
        choice[i - 1] = 0

    rec(1, 0.0)
    if best_choice is None:
        return RoundSolve("exact_bb", None, None, None, evals, True, time.perf_counter() - t0)
    ch = tuple(best_choice)
    th = grid.thetas(ch)
    return RoundSolve("exact_bb", ch, th, objective_value(inst, oid, th), evals, True,
                      time.perf_counter() - t0)


def exhaustive_local_solve(inst: Instance, objective_id: ObjectiveId | str, grid: LocalGrid,
                           table=None, *, max_states: int = 2_000_000) -> RoundSolve:
    """Plain product enumeration (test oracle for :func:`exact_local_solve`)."""
    t0 = time.perf_counter()
    oid = coerce_objective_id(objective_id)
    if grid.search_space() > max_states:
        raise ValueError("search space too large for exhaustive oracle")
    if table is None:
        table = conflict_table(inst, grid)
    best: tuple[float, tuple[int, ...]] | None = None
    evals = 0
    for ch in itertools.product(*(range(len(o)) for o in grid.options)):
        evals += 1
        if any(mat[ch[i - 1]][ch[j - 1]] for (i, j), mat in table.items()):
            continue
        th = grid.thetas(ch)
        v = objective_value(inst, oid, th)
        if best is None or v < best[0] or (v == best[0] and ch < best[1]):
            best = (v, ch)
    if best is None:
        return RoundSolve("exhaustive", None, None, None, evals, True, time.perf_counter() - t0)
    th = grid.thetas(best[1])
    return RoundSolve("exhaustive", best[1], th, best[0], evals, True, time.perf_counter() - t0)


def sa_local_solve(inst: Instance, objective_id: ObjectiveId | str, grid: LocalGrid,
                   table=None, *, seed: int = 0, n_sweeps: int = 200, restarts: int = 4,
                   t_start: float = 1.0, t_end: float = 0.01,
                   start_choice: tuple[int, ...] | None = None) -> RoundSolve:
    """Heuristic simulated annealing on ``cost + lam * conflicts`` over the local grid.

    One-hot preserving (one option per aircraft). Returns the best CONFLICT-FREE
    state visited (or ``None``). Deterministic given ``seed``. No optimality claim.
    """
    t0 = time.perf_counter()
    oid = coerce_objective_id(objective_id)
    if table is None:
        table = conflict_table(inst, grid)
    n = grid.n
    cost = _per_option_cost(inst, oid, grid)
    lam = 10.0 * (sum(max(c) for c in cost) + 1.0)
    rng = random.Random(seed)
    pairs = list(table.items())

    def energy(ch: list[int]) -> tuple[float, int]:
        conf = sum(1 for (i, j), mat in pairs if mat[ch[i - 1]][ch[j - 1]])
        return sum(cost[i][ch[i]] for i in range(n)) + lam * conf, conf

    best: tuple[float, tuple[int, ...]] | None = None
    evals = 0
    for r in range(restarts):
        if r == 0 and start_choice is not None:
            ch = list(start_choice)
        else:
            ch = [rng.randrange(len(o)) for o in grid.options]
        e, conf = energy(ch)
        evals += 1
        if conf == 0 and (best is None or e < best[0] or (e == best[0] and tuple(ch) < best[1])):
            best = (e, tuple(ch))
        for s in range(n_sweeps):
            frac = s / max(1, n_sweeps - 1)
            temp = t_start * (t_end / t_start) ** frac
            for i in range(n):
                k = len(grid.options[i])
                if k == 1:
                    continue
                old = ch[i]
                new = rng.randrange(k)
                if new == old:
                    continue
                ch[i] = new
                e2, conf2 = energy(ch)
                evals += 1
                d = e2 - e
                if d <= 0 or rng.random() < math.exp(-d / max(temp, 1e-12)):
                    e, conf = e2, conf2
                    if conf == 0 and (best is None or e < best[0]
                                      or (e == best[0] and tuple(ch) < best[1])):
                        best = (e, tuple(ch))
                else:
                    ch[i] = old
    if best is None:
        return RoundSolve("sa", None, None, None, evals, False, time.perf_counter() - t0, seed)
    th = grid.thetas(best[1])
    return RoundSolve("sa", best[1], th, objective_value(inst, oid, th), evals, False,
                      time.perf_counter() - t0, seed)


# --------------------------------------------------------------------------- #
# Adaptive loop
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdaptiveConfig:
    objective_id: str
    delta0: float | None = None  # None -> inst.hmax so round 0 == official K3 grid when centres = 0
    rho: float = 0.5
    max_rounds: int = 8
    include_noop: bool = False
    stagnation_patience: int = 2  # stop after this many consecutive rounds without archive improvement
    min_delta: float = 1e-6
    expand_on_infeasible: bool = False  # if a round is infeasible, retry with delta/rho once (guarded)
    solver: str = "exact"  # "exact" | "sa"
    seed: int = 0
    sa_sweeps: int = 200
    sa_restarts: int = 4
    max_states: int = 5_000_000  # cap on the exact round solver's search space (B&B prunes far below)

    def validate(self) -> None:
        if isinstance(self.max_states, bool) or not isinstance(self.max_states, int) or self.max_states < 1:
            raise ValueError("max_states must be an integer >= 1")
        coerce_objective_id(self.objective_id)
        if isinstance(self.rho, bool) or not isinstance(self.rho, (int, float)) \
                or not math.isfinite(self.rho) or not 0.0 < self.rho < 1.0:
            raise ValueError(f"rho must be a finite number in (0, 1), got {self.rho!r}")
        if self.delta0 is not None:
            d = _check_finite(self.delta0, "delta0")
            if d <= 0:
                raise ValueError("delta0 must be > 0")
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int) or self.max_rounds < 1:
            raise ValueError("max_rounds must be an integer >= 1")
        if self.stagnation_patience < 1:
            raise ValueError("stagnation_patience must be >= 1")
        if self.solver not in ("exact", "sa"):
            raise ValueError(f"unknown solver {self.solver!r}")
        if not math.isfinite(self.min_delta) or self.min_delta <= 0:
            raise ValueError("min_delta must be finite and > 0")


@dataclass
class RoundRecord:
    round: int
    delta: float
    centers: tuple[float, ...]
    options: tuple[tuple[float, ...], ...]
    n_qubits: int
    max_k: int
    search_space: int
    n_onehot_quadratic_terms: int
    n_conflict_quadratic_terms: int
    solver: str
    exact: bool
    previous_solution_in_grid: bool
    feasible: bool
    theta: tuple[float, ...] | None
    objective: float | None
    n_conflicts: int | None
    n_moved: int | None
    n_evaluations: int
    wall_time_s: float
    archive_objective_after: float | None
    archive_improved: bool
    note: str = ""


@dataclass
class AdaptiveResult:
    instance: str
    n: int
    objective_id: str
    init_mode: str
    config: dict
    rounds: list[RoundRecord]
    best_theta: tuple[float, ...] | None
    best_objective: float | None
    best_feasible: bool
    best_round: int | None
    stop_reason: str
    n_rounds: int
    n_solver_calls: int
    total_evaluations: int
    max_qubits: int
    width_times_rounds: int
    total_wall_time_s: float
    center_trajectory: list[tuple[float, ...]]
    delta_trajectory: list[float]
    flags: dict = field(default_factory=lambda: dict(ARTIFACT_FLAGS))

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _clip_centers(inst: Instance, theta: Sequence[float]) -> tuple[float, ...]:
    return tuple(min(inst.hmax, max(inst.hmin, float(t))) for t in theta)


def run_adaptive(
    inst: Instance,
    config: AdaptiveConfig,
    *,
    init_mode: str = "noop",
    init_theta: Sequence[float] | None = None,
) -> AdaptiveResult:
    """Run the adaptive iterative grid on ``inst``.

    ``init_mode``:
      * ``"noop"``       — centres ``c^(0) = 0`` (initial trajectories). With the
                           default ``delta0`` round 0 is exactly the official K=3 grid.
      * ``"continuous"`` — centres given by ``init_theta`` (e.g. a classical
                           continuous solution, read-only). This is a warm-start of
                           the GRID/controls, not of any QAOA state.
    """
    config.validate()
    oid = coerce_objective_id(config.objective_id)
    t_all = time.perf_counter()
    if init_mode == "noop":
        centers = (0.0,) * inst.n
    elif init_mode == "continuous":
        if init_theta is None or len(init_theta) != inst.n:
            raise ValueError("init_mode='continuous' requires init_theta of length n")
        for t in init_theta:
            _check_finite(t, "init_theta")
        centers = _clip_centers(inst, init_theta)
    else:
        raise ValueError(f"unknown init_mode {init_mode!r}")

    delta = float(inst.hmax) if config.delta0 is None else float(config.delta0)
    rounds: list[RoundRecord] = []
    best_theta: tuple[float, ...] | None = None
    best_obj: float | None = None
    best_round: int | None = None
    prev_theta: tuple[float, ...] | None = None
    stagnation = 0
    solver_calls = 0
    total_evals = 0
    stop_reason = "max_rounds"
    seen_states: set[tuple[tuple[float, ...], float]] = set()
    expanded_once = False
    r = 0
    while r < config.max_rounds:
        if delta < config.min_delta:
            stop_reason = "delta_below_min"
            break
        key = (centers, delta)
        if key in seen_states:
            stop_reason = "cycle_detected"
            break
        seen_states.add(key)
        grid = build_local_grid(inst, centers, delta, include_noop=config.include_noop)
        table = conflict_table(inst, grid)
        prev_in = prev_theta is None or grid.contains(prev_theta)
        if config.solver == "exact":
            rs = exact_local_solve(inst, oid, grid, table, max_states=config.max_states)
        else:
            start = None
            if prev_theta is not None and prev_in:
                start = tuple(opts.index(t) for opts, t in zip(grid.options, prev_theta))
            rs = sa_local_solve(inst, oid, grid, table, seed=config.seed + r,
                                n_sweeps=config.sa_sweeps, restarts=config.sa_restarts,
                                start_choice=start)
        solver_calls += 1
        total_evals += rs.n_evaluations
        improved = False
        note = ""
        if rs.theta is not None:
            nconf = count_conflicts(inst, rs.theta)
            feasible = nconf == 0
            nmoved = sum(1 for t in rs.theta if t != NOOP_THETA)
            obj = rs.objective
            assert obj is not None  # a solved round always carries its objective
            if feasible and (best_obj is None or obj < best_obj):
                best_theta, best_obj, best_round, improved = rs.theta, obj, r, True
            elif feasible and best_obj is not None and obj == best_obj and best_theta != rs.theta:
                note = "tie_with_archive_kept_archive"
        else:
            nconf, feasible, nmoved, obj = None, False, None, None
        rounds.append(RoundRecord(
            round=r, delta=delta, centers=centers, options=grid.options,
            n_qubits=grid.n_qubits, max_k=grid.max_k, search_space=grid.search_space(),
            n_onehot_quadratic_terms=grid.n_onehot_quadratic_terms(),
            n_conflict_quadratic_terms=n_conflict_terms(table), solver=rs.solver,
            exact=rs.exact, previous_solution_in_grid=prev_in, feasible=feasible,
            theta=rs.theta, objective=obj, n_conflicts=nconf, n_moved=nmoved,
            n_evaluations=rs.n_evaluations, wall_time_s=rs.wall_time_s,
            archive_objective_after=best_obj, archive_improved=improved, note=note,
        ))
        r += 1
        if not feasible:
            if best_theta is None:
                if config.expand_on_infeasible and not expanded_once:
                    expanded_once = True
                    delta = min(delta / config.rho, float(inst.hmax - inst.hmin))
                    rounds[-1].note = "infeasible_expanded_delta"
                    continue
                stop_reason = "infeasible_no_incumbent"
                break
            # keep incumbent as centre, shrink anyway (deterministic recentring policy)
            centers = best_theta
            prev_theta = best_theta
            rounds[-1].note = rounds[-1].note or "infeasible_recentre_on_archive"
        else:
            # recentre on the ROUND solution when it is at least as good as the archive,
            # else on the archive (a heuristic round may be worse than the incumbent)
            if best_theta is not None and obj is not None and best_obj is not None and obj > best_obj:
                centers = best_theta
                prev_theta = best_theta
                rounds[-1].note = rounds[-1].note or "round_worse_than_archive_recentre_on_archive"
            else:
                centers = rs.theta  # type: ignore[assignment]
                prev_theta = rs.theta
        stagnation = 0 if improved else stagnation + 1
        if stagnation >= config.stagnation_patience:
            stop_reason = "stagnation"
            break
        delta = delta * config.rho
    if r >= config.max_rounds and stop_reason == "max_rounds":
        pass
    max_q = max((rr.n_qubits for rr in rounds), default=0)
    return AdaptiveResult(
        instance=inst.name, n=inst.n, objective_id=oid.value, init_mode=init_mode,
        config=asdict(config), rounds=rounds, best_theta=best_theta, best_objective=best_obj,
        best_feasible=best_theta is not None, best_round=best_round, stop_reason=stop_reason,
        n_rounds=len(rounds), n_solver_calls=solver_calls, total_evaluations=total_evals,
        max_qubits=max_q, width_times_rounds=sum(rr.n_qubits for rr in rounds),
        total_wall_time_s=time.perf_counter() - t_all,
        center_trajectory=[rr.centers for rr in rounds], delta_trajectory=[rr.delta for rr in rounds],
    )


def archive_is_monotone(result: AdaptiveResult) -> bool:
    """Elitist-archive monotonicity: archive objective never increases across rounds."""
    prev = None
    for rr in result.rounds:
        cur = rr.archive_objective_after
        if prev is not None and cur is not None and cur > prev:
            return False
        if cur is not None:
            prev = cur
    return True


def round_solutions_monotone(result: AdaptiveResult) -> bool:
    """Stronger property (exact solver only): each feasible round objective <= previous.

    Holds when every round is exact AND the previous solution is in the grid.
    Returns False when it is violated; callers must not claim it for heuristics.
    """
    prev = None
    for rr in result.rounds:
        if rr.objective is None:
            continue
        if prev is not None and rr.objective > prev:
            return False
        prev = rr.objective
    return True


# --------------------------------------------------------------------------- #
# Classical rounding baseline (to isolate the 'quantum step' contribution)
# --------------------------------------------------------------------------- #
def snap_to_grid_baseline(inst: Instance, objective_id: ObjectiveId | str,
                          theta_cont: Sequence[float], k: int) -> dict:
    """Nearest official K-level point of a continuous solution, no repair.

    Reports feasibility and objective of the snapped point; a fair 'purely
    classical' comparator for the continuous-centre initialisation.
    """
    levels = official_k_grid_options(inst, k)
    snapped = tuple(min(levels, key=lambda lvl: (abs(lvl - t), lvl)) for t in theta_cont)
    nconf = count_conflicts(inst, snapped)
    return {"k": k, "theta": snapped, "n_conflicts": nconf, "feasible": nconf == 0,
            "objective": objective_value(inst, objective_id, snapped)}


def snap_and_greedy_repair_baseline(inst: Instance, objective_id: ObjectiveId | str,
                                    theta_cont: Sequence[float], k: int,
                                    max_passes: int | None = None) -> dict:
    """Purely classical control baseline: snap to the official K grid, then greedy repair.

    Repair loop (deterministic): while conflicts remain, scan aircraft in id
    order; for each aircraft involved in a conflict, move it to the grid level
    minimising ``(total conflicts, objective)``; accept only strict conflict
    reductions. Stops when no single-aircraft move reduces conflicts (reported
    as infeasible) or after ``max_passes`` (default ``n*k``) passes.
    """
    oid = coerce_objective_id(objective_id)
    levels = official_k_grid_options(inst, k)
    cur = [min(levels, key=lambda lvl: (abs(lvl - t), lvl)) for t in theta_cont]
    n = inst.n
    passes = 0
    limit = n * k if max_passes is None else max_passes
    n_moves = 0
    evals = 1
    nconf = count_conflicts(inst, cur)
    while nconf > 0 and passes < limit:
        passes += 1
        improved = False
        reports = geometry.conflict_reports(inst, (1.0,) * n, tuple(cur))
        involved = sorted({r.i for r in reports if r.in_conflict} | {r.j for r in reports if r.in_conflict})
        for i in involved:
            best = (nconf, objective_value(inst, oid, cur), cur[i - 1])
            for lvl in levels:
                if lvl == cur[i - 1]:
                    continue
                trial = list(cur)
                trial[i - 1] = lvl
                c = count_conflicts(inst, trial)
                evals += 1
                cand = (c, objective_value(inst, oid, trial), lvl)
                if cand < best:
                    best = cand
            if best[0] < nconf:
                cur[i - 1] = best[2]
                nconf = best[0]
                n_moves += 1
                improved = True
                if nconf == 0:
                    break
        if not improved:
            break
    theta = tuple(cur)
    return {"k": k, "theta": theta, "n_conflicts": nconf, "feasible": nconf == 0,
            "objective": objective_value(inst, oid, theta), "n_moves": n_moves,
            "passes": passes, "n_evaluations": evals}


def reach_bound(delta0: float, rho: float) -> float:
    """Maximum distance a centre can travel from its initial value: sum_r Delta_r = Delta_0/(1-rho).

    Any heading farther than this from the initial centre is unreachable, however
    many rounds are run (geometric series). With ``c=0`` and ``Delta_0 = HMAX``
    the bound covers the whole box iff ``rho >= 1/2``.
    """
    delta0 = _check_finite(delta0, "delta0")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    return delta0 / (1.0 - rho)
