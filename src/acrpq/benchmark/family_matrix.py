"""Applicability matrix across the CP / RCP / FP / GP instance families (Phase 3A).

Pure METADATA inventory — no solver is run here. For each (instance, grid) cell it
records the problem shape (aircraft, initial conflicts, conflict-graph density,
grid, variables/qubits, QUBO term counts, K**n) and, for every method, whether it
is applicable and why — reusing ``preflight_payload`` (the dashboard's single
source of truth for the enumeration/RAM caps) so the matrix can never disagree
with what the solvers actually enforce. The Gurobi discrete MILP (a compact
n*K-binary program that scales past the enumeration caps) is added on top.

Method keys mirror the benchmark legs:
  reference       exhaustive in-grid (exact under its cap, else greedy fallback)
  qubo_exact      exact min H(x)
  qubo_annealing  internal simulated annealing
  dwave_sa        Ocean/D-Wave reference SA (CPU)
  qaoa_aer        QAOA on the Aer simulator
  gurobi_milp     classical discrete compatibility MILP (Gurobi via AMPL)
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
from dataclasses import asdict, dataclass, field

from .. import geometry
from ..discretize import DiscreteACRP
from ..io.loader import InstanceLoader
from ..model import Family, ManeuverGrid

METHOD_KEYS = ["reference", "qubo_exact", "qubo_annealing", "dwave_sa", "qaoa_aer", "gurobi_milp"]
SCALAR_COLS = ["family", "instance", "n_aircraft", "n_pairs", "initial_conflicts",
               "conflict_density", "grid_k", "n_theta", "n_q", "n_vars",
               "n_onehot_terms", "n_conflict_terms", "search_space"]


def rows_to_csv(rows: list[dict]) -> str:
    """Flatten matrix rows to CSV (one applicable flag per method)."""
    buf = io.StringIO()
    cols = SCALAR_COLS + [f"{m}_applicable" for m in METHOD_KEYS]
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        flat = {c: r[c] for c in SCALAR_COLS}
        for m in METHOD_KEYS:
            flat[f"{m}_applicable"] = r["methods"].get(m, {}).get("applicable")
        w.writerow(flat)
    return buf.getvalue()


def assert_json_csv_parity(rows: list[dict], csv_text: str) -> None:
    """Fail-closed check that the CSV faithfully mirrors the JSON rows — EVERY
    scalar column plus every per-method applicable flag (not a subset)."""
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(csv_rows) != len(rows):
        raise ValueError(f"JSON/CSV row-count mismatch: {len(rows)} vs {len(csv_rows)}")
    for jr, cr in zip(rows, csv_rows):
        for col in SCALAR_COLS:
            if str(jr[col]) != cr[col]:
                raise ValueError(
                    f"JSON/CSV parity mismatch at {jr['instance']} col={col}: "
                    f"{jr[col]!r} != {cr[col]!r}")
        for m in METHOD_KEYS:
            if str(jr["methods"][m]["applicable"]) != cr[f"{m}_applicable"]:
                raise ValueError(f"JSON/CSV parity mismatch at {jr['instance']} {m}")


def canonical_matrix_sha256(matrix: dict) -> str:
    """Deterministic sha over the matrix content (metadata only; no wall-time)."""
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()
                    if k not in ("matrix_sha256", "matrix_sha256_note")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj
    blob = json.dumps(strip(matrix), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

# preflight method key -> our benchmark method key
_PF_RENAME = {
    "reference": "reference", "qubo-exact": "qubo_exact",
    "qubo-annealing": "qubo_annealing", "dwave": "dwave_sa", "qaoa": "qaoa_aer",
}


def _amplpy_available() -> bool:
    try:
        return importlib.util.find_spec("amplpy") is not None
    except Exception:
        return False


@dataclass(frozen=True)
class MatrixCell:
    family: str
    instance: str
    n_aircraft: int
    n_pairs: int
    initial_conflicts: int
    conflict_density: float          # initial-conflict pairs / n_pairs
    grid_k: int
    n_theta: int
    n_q: int
    n_vars: int                      # == qubits == n * K
    n_onehot_terms: int
    n_conflict_terms: int
    search_space: str                # K**n as a string (can be astronomically large)
    methods: dict[str, dict] = field(default_factory=dict)


def cell(loader: InstanceLoader, name: str, *, n_theta: int, n_q: int = 1) -> MatrixCell:
    """One (instance, grid) applicability row. No solver is executed."""
    from ..dashboard.api import preflight_payload  # local: pulls in FastAPI-free helpers

    inst = loader.load(name)
    n_pairs = inst.n_pairs()
    initial = geometry.count_initial_conflicts(inst)
    grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
    discrete = DiscreteACRP.build(inst, grid)  # cheap: O(n_pairs * K^2), no QUBO/enum
    k = grid.k
    n_onehot = inst.n * (k * (k - 1) // 2)
    n_conflict = sum(
        1 for (_i, _j), mat in discrete.conflict.items()
        for oi in range(k) for oj in range(k) if mat[oi][oj])

    pf = preflight_payload(loader, name, n_theta=n_theta, n_q=n_q)
    methods: dict[str, dict] = {}
    for m in pf["methods"]:
        key = _PF_RENAME.get(m["method"])
        if key is None:
            continue
        methods[key] = {"applicable": bool(m["applicable"]),
                        "resolution": m.get("resolution"), "reason": m["reason"]}
    # Gurobi discrete MILP: a compact MILP (n*K binaries + one-hot + incompatible
    # pairs); no enumeration/RAM ceiling — applicable whenever AMPL/Gurobi is present.
    ampl = _amplpy_available()
    methods["gurobi_milp"] = {
        "applicable": ampl,
        "resolution": "exact" if ampl else None,
        "reason": (f"discrete compatibility MILP: {inst.n}*{k}={inst.n * k} binaries, "
                   f"one-hot + pairwise-incompatibility constraints"
                   if ampl else "not available: AMPL/Gurobi (amplpy) not importable"),
    }

    return MatrixCell(
        family=inst.family.value, instance=name, n_aircraft=inst.n, n_pairs=n_pairs,
        initial_conflicts=initial,
        conflict_density=(initial / n_pairs) if n_pairs else 0.0,
        grid_k=k, n_theta=n_theta, n_q=n_q, n_vars=inst.n * k,
        n_onehot_terms=n_onehot, n_conflict_terms=n_conflict,
        search_space=str(k**inst.n), methods=methods)


def _natural_key(name: str) -> tuple:
    """Full natural sort: split on digit runs so every embedded integer sorts
    numerically (RCP_10_* < RCP_50_* < RCP_100_*, and RCP_50_9@FL5 places by 50
    then 9 then 5). A per-family cap therefore keeps the *smallest* (cheapest)
    instances first — a defensible, deterministic affordable subset — for every
    family, not just the single-index ones. The full name breaks any tie.
    """
    tokens = tuple((int(t), "") if t.isdigit() else (-1, t)
                   for t in re.split(r"(\d+)", name) if t)
    return (tokens, name)


def family_names(loader: InstanceLoader, family: Family | str) -> list[str]:
    return sorted(loader.list_names(family), key=_natural_key)


def build_matrix(
    loader: InstanceLoader,
    *,
    families: list[str] | None = None,
    k_all: int = 3,
    k_sensitivity: tuple[int, ...] = (5, 7),
    sensitivity_max_aircraft: int = 8,
    per_family_limit: dict[str, int] | int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build the matrix rows plus a per-family/limit log of what was covered.

    Every instance is inventoried at ``k_all`` (the common K=3 grid). The finer
    ``k_sensitivity`` grids are added only for instances with at most
    ``sensitivity_max_aircraft`` aircraft (so the sensitivity study stays
    affordable). ``per_family_limit`` optionally caps instances per family (an int
    applies to all; a dict maps family -> cap, absent families are uncapped). The
    cap and how many instances were dropped are recorded per family — never a
    silent truncation.
    """
    fams = families or [f.value for f in Family]

    def _limit(fam: str) -> int | None:
        if isinstance(per_family_limit, dict):
            return per_family_limit.get(fam)
        return per_family_limit

    rows: list[dict] = []
    coverage: list[dict] = []
    for fam in fams:
        names = family_names(loader, fam)
        lim = _limit(fam)
        used = names if lim is None else names[:lim]
        coverage.append({"family": fam, "available": len(names), "covered": len(used),
                         "dropped": len(names) - len(used), "cap": lim})
        for name in used:
            base = cell(loader, name, n_theta=k_all)
            rows.append(asdict(base))
            if base.n_aircraft <= sensitivity_max_aircraft:
                for k in k_sensitivity:
                    rows.append(asdict(cell(loader, name, n_theta=k)))
    return rows, coverage
