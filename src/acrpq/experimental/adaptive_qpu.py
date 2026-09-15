"""EXPERIMENTAL — heterogeneous-grid QUBO, sampling decoder and hash-chained
adaptive rounds for the ACRP (gate study, no submission code).

Flags: ``experimental = True``, ``official_benchmark = False``. ``real_qpu`` is
never set by this module: it can only be true for an artefact carrying an IBM
``job_id`` and authenticated raw counts, which this module does not produce.

This module is ISOLATED from the official pipeline (it imports only read-only
helpers: geometry kernel, objectives, instance model, and the sibling
experimental ``adaptive_grid``). It deliberately re-implements — and tests
against — the official conventions:

* QUBO: ``H = Σ cost·x + λ_pen·Σ conflicts + λ_oh·Σ_i (Σ_o x_{i,o} − 1)²`` with
  ``λ_pen = pen_scale·(n·cost_max + 1)``, ``λ_oh = max(n, maxdeg+1)·λ_pen``
  (``acrpq.quantum.qubo.build_qubo``). For a uniform K3 grid the coefficients
  are identical term by term (asserted in tests).
* Variable layout for heterogeneous blocks: ``off_i = Σ_{j<i} K_j``,
  ``bit(i, o) = off_i + o``.
* Circuit: fixed-angle QAOA exactly as ``hardware_validation.build_parameterized_qaoa``
  (H, RZ(2γh), RZZ(2γJ), barrier, RX(2β), barrier, measure b→b).
* Bit order: Qiskit count keys are little-endian; QUBO order is the reversed
  string (``bits[b]`` = value of qubit ``b``).

Chain identity: every round carries ``round_hash_r = sha256(canonical(payload_r)
|| round_hash_{r-1} || selected_candidate_hash_{r-1})`` and a driver kind in
``{exact_local, sa, aer_sim, qpu_result}``. A classically driven trajectory can
never be labelled ``qpu_result``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..model import Instance
from ..objectives import ObjectiveId, coerce_objective_id, primary_objective
from . import adaptive_grid as ag

EXPERIMENTAL = True
OFFICIAL_BENCHMARK = False
SCHEMA_CHAIN = "acrpq-experimental-adaptive-chain/1"
SCHEMA_ROUND = "acrpq-experimental-adaptive-round/1"
ADAPTIVE_PROTOCOL_LOCAL_EXACT = "LOCAL_EXACT_ADAPTIVE_V1"
ADAPTIVE_PROTOCOL_QPU_PILOT = "QPU_DRIVEN_ADAPTIVE_PILOT_V1"
DRIVER_KINDS = ("exact_local", "sa", "aer_sim", "qpu_result")
GENESIS_HASH = "0" * 64
NO_FEASIBLE_POLICIES = ("stop", "keep_center_shrink", "keep_center_retry_once")
ANGLE_RULES = ("fixed", "penalty_normalised")

FLAGS = {"experimental": True, "official_benchmark": False}


class ChainIntegrityError(RuntimeError):
    """Fail-closed refusal: the chain state does not allow the requested step."""


# --------------------------------------------------------------------------- #
# Canonical hashing
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no NaN/Inf, repr-exact floats."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not canonically serialisable: {type(o).__name__}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def instance_sha256(inst: Instance) -> str:
    payload = {
        "name": inst.name, "family": inst.family.value, "n": inst.n, "d": inst.d,
        "radius": inst.radius, "v0": list(inst.v0), "cap": list(inst.cap), "x0": list(inst.x0),
        "y0": list(inst.y0), "w": inst.w, "nf": inst.nf, "l0": list(inst.l0), "level": inst.level,
        "qmin": inst.qmin, "qmax": inst.qmax, "hmin": inst.hmin, "hmax": inst.hmax,
    }
    return sha256_obj(payload)


def grid_hash(grid: ag.LocalGrid) -> str:
    """Hash of the ORDERED per-aircraft grids (aircraft id order is load-bearing)."""
    return sha256_obj({"options": [list(o) for o in grid.options], "centers": list(grid.centers),
                       "delta": grid.delta if math.isfinite(grid.delta) else None,
                       "include_noop": grid.include_noop})


# --------------------------------------------------------------------------- #
# Heterogeneous QUBO
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeteroQUBO:
    instance: Instance
    grid: ag.LocalGrid
    objective_id: str
    offsets: tuple[int, ...]  # offsets[i-1]
    n_qubits: int
    linear: Mapping[int, float]
    quadratic: Mapping[tuple[int, int], float]
    constant: float
    penalty_conflict: float
    penalty_onehot: float
    cost: tuple[tuple[float, ...], ...]  # cost[i-1][o] physical cost
    meta: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear", MappingProxyType(dict(self.linear)))
        object.__setattr__(self, "quadratic", MappingProxyType(dict(self.quadratic)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def n(self) -> int:
        return self.instance.n

    def block_size(self, i: int) -> int:
        return len(self.grid.options[i - 1])

    def var_index(self, i: int, o: int) -> int:
        if not 1 <= i <= self.n or not 0 <= o < self.block_size(i):
            raise ValueError(f"invalid (aircraft, option) = ({i}, {o})")
        return self.offsets[i - 1] + o

    def inv_index(self, b: int) -> tuple[int, int]:
        if isinstance(b, bool) or not isinstance(b, int) or not 0 <= b < self.n_qubits:
            raise ValueError(f"bit {b!r} out of range 0..{self.n_qubits - 1}")
        for i in range(self.n, 0, -1):
            if b >= self.offsets[i - 1]:
                return i, b - self.offsets[i - 1]
        raise AssertionError("unreachable")  # pragma: no cover

    def energy(self, bits: Sequence[int] | str) -> float:
        x = _bits_to_list(bits, self.n_qubits)
        e = self.constant
        for b, a in self.linear.items():
            if x[b]:
                e += a
        for (a, c), q in self.quadratic.items():
            if x[a] and x[c]:
                e += q
        return e

    def canonical_hash(self) -> str:
        payload = {
            "schema": "acrpq-experimental-hetero-qubo/1",
            "instance_sha256": instance_sha256(self.instance),
            "objective_id": self.objective_id,
            "grid_hash": grid_hash(self.grid),
            "grids": [list(o) for o in self.grid.options],
            "offsets": list(self.offsets), "n_qubits": self.n_qubits,
            "constant": self.constant,
            "linear": [[b, self.linear[b]] for b in sorted(self.linear)],
            "quadratic": [[a, c, self.quadratic[(a, c)]] for (a, c) in sorted(self.quadratic)],
            "penalty_conflict": self.penalty_conflict, "penalty_onehot": self.penalty_onehot,
        }
        return sha256_obj(payload)

    def to_ising(self) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
        """x = (1 - z)/2 ; returns (h, J, offset) — same convention as acrpq.quantum.ising."""
        h = {b: 0.0 for b in range(self.n_qubits)}
        J: dict[tuple[int, int], float] = {}
        offset = self.constant
        for b, a in self.linear.items():
            offset += a / 2.0
            h[b] -= a / 2.0
        for (a, b), q in self.quadratic.items():
            offset += q / 4.0
            h[a] -= q / 4.0
            h[b] -= q / 4.0
            J[(a, b)] = J.get((a, b), 0.0) + q / 4.0
        h = {b: v for b, v in h.items() if abs(v) > 1e-12}
        return h, J, offset


def _bits_to_list(bits: Sequence[int] | str, n: int) -> list[int]:
    if isinstance(bits, str):
        if len(bits) != n or set(bits) - {"0", "1"}:
            raise ValueError(f"bitstring must be {n} binary characters")
        return [1 if ch == "1" else 0 for ch in bits]
    vals = list(bits)
    if len(vals) != n:
        raise ValueError(f"bit sequence must have length {n}")
    out = []
    for v in vals:
        if isinstance(v, bool):
            out.append(int(v))
        elif v in (0, 1):
            out.append(int(v))
        else:
            raise ValueError("bit values must be binary")
    return out


def build_hetero_qubo(inst: Instance, grid: ag.LocalGrid, objective_id: ObjectiveId | str,
                      *, pen_scale: float = 10.0, table=None) -> HeteroQUBO:
    """Build the heterogeneous-block QUBO with the OFFICIAL penalty rule."""
    oid = coerce_objective_id(objective_id)
    if grid.n != inst.n:
        raise ValueError("grid/instance size mismatch")
    for opts in grid.options:
        if len(opts) < 1:
            raise ValueError("every aircraft needs at least one option")
        if len(set(opts)) != len(opts):
            raise ValueError("duplicated physical option inside a block (dedup violated)")
        if list(opts) != sorted(opts):
            raise ValueError("block options must be sorted ascending")
    if table is None:
        table = ag.conflict_table(inst, grid)
    sizes = [len(o) for o in grid.options]
    offsets = tuple(sum(sizes[:i]) for i in range(inst.n))
    n_qubits = sum(sizes)
    cost = tuple(tuple(primary_objective(oid, (1.0,), (t,), inst.w) for t in opts) for opts in grid.options)
    cost_max = max((c for row in cost for c in row), default=0.0)
    deg: dict[int, int] = {}
    for i, j in inst.pairs():
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    maxdeg = max(deg.values(), default=0)
    lam_pen = pen_scale * (inst.n * cost_max + 1.0)
    lam_oh = max(inst.n, maxdeg + 1) * lam_pen
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0
    for i in inst.aircraft():
        constant += lam_oh
        k = sizes[i - 1]
        for o in range(k):
            b = offsets[i - 1] + o
            linear[b] = linear.get(b, 0.0) + cost[i - 1][o] - lam_oh
        for o1 in range(k):
            for o2 in range(o1 + 1, k):
                key = (offsets[i - 1] + o1, offsets[i - 1] + o2)
                quadratic[key] = quadratic.get(key, 0.0) + 2.0 * lam_oh
    n_conf = 0
    for (i, j), mat in table.items():
        for oi, row in enumerate(mat):
            for oj, v in enumerate(row):
                if v:
                    a, c = offsets[i - 1] + oi, offsets[j - 1] + oj
                    key = (a, c) if a < c else (c, a)
                    quadratic[key] = quadratic.get(key, 0.0) + lam_pen
                    n_conf += 1
    meta = {"n_conflict_terms": float(n_conf),
            "n_onehot_terms": float(sum(k * (k - 1) // 2 for k in sizes)),
            "cost_max": cost_max, "max_conflict_degree": float(maxdeg),
            "n_aircraft": float(inst.n), "min_block": float(min(sizes)), "max_block": float(max(sizes)),
            "onehot_optimum_guaranteed": 1.0 if (lam_pen > inst.n * cost_max and lam_oh > lam_pen * maxdeg) else 0.0}
    return HeteroQUBO(instance=inst, grid=grid, objective_id=oid.value, offsets=offsets, n_qubits=n_qubits,
                      linear=linear, quadratic=quadratic, constant=constant, penalty_conflict=lam_pen,
                      penalty_onehot=lam_oh, cost=cost, meta=meta)


# --------------------------------------------------------------------------- #
# Encoding / decoding
# --------------------------------------------------------------------------- #
def encode_choice(qubo: HeteroQUBO, choice: Sequence[int]) -> str:
    """One-hot bitstring in QUBO order for a per-aircraft option choice."""
    if len(choice) != qubo.n:
        raise ValueError("choice length must be n")
    bits = ["0"] * qubo.n_qubits
    for i, o in enumerate(choice, start=1):
        bits[qubo.var_index(i, o)] = "1"
    return "".join(bits)


def to_qubo_order(raw: str, n: int, bit_order: str) -> str:
    if bit_order not in ("qiskit", "qubo"):
        raise ValueError("bit_order must be 'qiskit' or 'qubo'")
    if len(raw) != n or set(raw) - {"0", "1"}:
        raise ValueError(f"bitstring {raw!r} is not {n} binary characters")
    return raw[::-1] if bit_order == "qiskit" else raw


@dataclass(frozen=True)
class DecodedSample:
    bitstring_qubo: str
    weight: float  # probability
    onehot_valid: bool
    n_onehot_violations: int
    choice: tuple[int, ...]  # option index per aircraft after repair
    repaired: bool
    theta: tuple[float, ...]
    n_conflicts: int
    feasible: bool  # onehot_valid AND zero conflicts
    objective: float
    energy: float
    penalty: float  # energy - objective (0 on a valid, conflict-free state)


def _repair_block(qubo: HeteroQUBO, i: int, set_opts: list[int]) -> tuple[int, bool]:
    """Repair policy for a block: one bit -> that option; none -> the CENTRE
    option (always present in a local grid; the closest-to-zero option for a
    global grid); several -> objective-minimal, then smallest |theta|, then index."""
    opts = qubo.grid.options[i - 1]
    if len(set_opts) == 1:
        return set_opts[0], False
    if not set_opts:
        c = qubo.grid.centers[i - 1]
        if c in opts:
            return opts.index(c), True
        return min(range(len(opts)), key=lambda o: (abs(opts[o]), o)), True
    cost = qubo.cost[i - 1]
    best = min(cost[o] for o in set_opts)
    tied = [o for o in set_opts if cost[o] == best]
    return min(tied, key=lambda o: (abs(opts[o]), o)), True


def decode_bitstring(qubo: HeteroQUBO, bits_qubo: str, weight: float = 1.0) -> DecodedSample:
    x = _bits_to_list(bits_qubo, qubo.n_qubits)
    choice: list[int] = []
    violations = 0
    repaired_any = False
    for i in qubo.instance.aircraft():
        k = qubo.block_size(i)
        set_opts = [o for o in range(k) if x[qubo.offsets[i - 1] + o]]
        if len(set_opts) != 1:
            violations += 1
        o, rep = _repair_block(qubo, i, set_opts)
        repaired_any |= rep
        choice.append(o)
    theta = qubo.grid.thetas(choice)
    nconf = ag.count_conflicts(qubo.instance, theta)
    obj = ag.objective_value(qubo.instance, qubo.objective_id, theta)
    energy = qubo.energy(bits_qubo)
    return DecodedSample(bitstring_qubo=bits_qubo, weight=weight, onehot_valid=violations == 0,
                         n_onehot_violations=violations, choice=tuple(choice), repaired=repaired_any,
                         theta=theta, n_conflicts=nconf, feasible=(violations == 0 and nconf == 0),
                         objective=obj, energy=energy, penalty=energy - obj)


def normalise_raw(raw: Mapping[Any, Any], n: int, *, bit_order: str = "qiskit", kind: str = "counts",
                  shots: int | None = None) -> tuple[dict[str, float], int]:
    """Validate Sampler output and fold to {qubo_bitstring: probability}.

    Accepts str keys (optionally with register spaces) and int keys (converted to
    ``n``-bit strings in the given bit order). Refuses non-binary, wrong-length,
    bool/negative/non-finite values and zero totals.
    """
    if kind not in ("counts", "quasi"):
        raise ValueError("kind must be 'counts' or 'quasi'")
    if not isinstance(raw, Mapping):
        raise ValueError("raw results must be a mapping")
    if not raw:
        raise ValueError("empty result")
    norm: dict[str, float] = {}
    for key, val in raw.items():
        if isinstance(key, bool):
            raise ValueError("bool key")
        if isinstance(key, int):
            if key < 0 or key >= 2**n:
                raise ValueError(f"int key {key} out of range for {n} bits")
            ks = format(key, f"0{n}b")  # big-endian text == qiskit little-endian key convention
        elif isinstance(key, str):
            ks = key.replace(" ", "")
        else:
            raise ValueError(f"unsupported key type {type(key).__name__}")
        if kind == "counts":
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"count for {key!r} must be a non-negative int")
            fv = float(val)
        else:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"probability for {key!r} must be finite and >= 0")
            fv = float(val)
        q = to_qubo_order(ks, n, bit_order)
        norm[q] = norm.get(q, 0.0) + fv
    total = sum(norm.values())
    if total <= 0:
        raise ValueError("zero total weight (zero shots)")
    if kind == "counts":
        if shots is not None and int(total) != shots:
            raise ValueError(f"shots {shots} != count sum {int(total)}")
        n_shots = int(total)
    else:
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"quasi-distribution must sum to 1 (got {total})")
        n_shots = shots or 0
    return {k: v / total for k, v in norm.items()}, n_shots


@dataclass(frozen=True)
class DecodedCounts:
    n_shots: int
    n_unique: int
    samples: tuple[DecodedSample, ...]  # sorted by (energy, bitstring)
    best_feasible: DecodedSample | None
    best_raw: DecodedSample
    modal: DecodedSample
    feasibility_rate: float
    onehot_valid_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shots": self.n_shots, "n_unique": self.n_unique,
            "feasibility_rate": self.feasibility_rate, "onehot_valid_rate": self.onehot_valid_rate,
            "best_feasible": None if self.best_feasible is None else asdict(self.best_feasible),
            "best_raw": asdict(self.best_raw), "modal": asdict(self.modal),
            "samples": [asdict(s) for s in self.samples],
        }


def decode_counts(qubo: HeteroQUBO, raw: Mapping[Any, Any], *, bit_order: str = "qiskit",
                  kind: str = "counts", shots: int | None = None) -> DecodedCounts:
    prob, n_shots = normalise_raw(raw, qubo.n_qubits, bit_order=bit_order, kind=kind, shots=shots)
    samples = [decode_bitstring(qubo, bs, p) for bs, p in prob.items()]
    samples.sort(key=lambda s: (s.energy, s.bitstring_qubo))
    feas = [s for s in samples if s.feasible]
    # documented rule: lowest PHYSICAL objective, then lexicographic bitstring. Energy is
    # deliberately not used (for feasible samples it equals the objective up to fp noise).
    best_feasible = min(feas, key=lambda s: (s.objective, s.bitstring_qubo)) if feas else None
    modal = max(samples, key=lambda s: (s.weight, -samples.index(s)))
    return DecodedCounts(n_shots=n_shots, n_unique=len(samples), samples=tuple(samples),
                         best_feasible=best_feasible, best_raw=samples[0], modal=modal,
                         feasibility_rate=sum(s.weight for s in feas),
                         onehot_valid_rate=sum(s.weight for s in samples if s.onehot_valid))


def candidate_hash(sample: DecodedSample) -> str:
    return sha256_obj({"bitstring_qubo": sample.bitstring_qubo, "theta": list(sample.theta),
                       "objective": sample.objective, "feasible": sample.feasible})


# --------------------------------------------------------------------------- #
# Fixed-angle QAOA circuit (mirror of hardware_validation.build_parameterized_qaoa)
# --------------------------------------------------------------------------- #
def build_fixed_angle_circuit(qubo: HeteroQUBO, gammas: Sequence[float], betas: Sequence[float]):
    from qiskit import QuantumCircuit

    if len(gammas) != len(betas) or not gammas:
        raise ValueError("gammas and betas must have equal non-zero length")
    for v in list(gammas) + list(betas):
        if not math.isfinite(v):
            raise ValueError("angles must be finite")
    h, J, _ = qubo.to_ising()
    n = qubo.n_qubits
    qc = QuantumCircuit(n, n, name="acrpq_adaptive_qaoa")
    qc.h(range(n))
    for g, beta in zip(gammas, betas):
        # insertion order (NOT sorted): identical to hardware_validation.build_parameterized_qaoa,
        # so the routing heuristic sees the same gate sequence and the ISA matches the campaign.
        for b, hv in h.items():
            if hv:
                qc.rz(2.0 * g * hv, b)
        for (a, c), j in J.items():
            if j:
                qc.rzz(2.0 * g * j, a, c)
        qc.barrier()
        for b in range(n):
            qc.rx(2.0 * beta, b)
        qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def circuit_fingerprint(circuit) -> str:
    """Order-exact structural hash: (op name, qubit indices, clbit indices, params)."""
    ops = []
    qidx = {q: i for i, q in enumerate(circuit.qubits)}
    cidx = {c: i for i, c in enumerate(circuit.clbits)}
    for inst in circuit.data:
        params = []
        for p in inst.operation.params:
            try:
                params.append(repr(round(float(p), 12)))  # same tolerance as the official fingerprint
            except TypeError:
                params.append(str(p))
        ops.append([inst.operation.name, [qidx[q] for q in inst.qubits], [cidx[c] for c in inst.clbits], params])
    return sha256_obj({"n_qubits": circuit.num_qubits, "n_clbits": circuit.num_clbits, "ops": ops})


def circuit_metrics(circuit) -> dict[str, Any]:
    ops = circuit.count_ops()
    twoq = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    oneq = sum(1 for inst in circuit.data if inst.operation.num_qubits == 1
               and inst.operation.name not in ("measure", "barrier", "reset", "delay"))
    return {"depth": int(circuit.depth()), "size": int(circuit.size()), "n_2q": twoq, "n_1q": oneq,
            "ops": {k: int(v) for k, v in sorted(ops.items())}, "num_qubits": circuit.num_qubits}


@dataclass(frozen=True)
class IsaReport:
    backend_name: str
    is_fake: bool
    seed_transpiler: int
    optimization_level: int
    n_logical: int
    layout_physical: tuple[int, ...]
    depth: int
    n_2q: int
    n_1q: int
    ops: dict[str, int]
    basis_ok: bool
    isa_hash: str
    reproducible: bool
    isa_hash_second: str
    wall_time_s: float


def transpile_isa(circuit, backend, *, seed_transpiler: int = 1, optimization_level: int = 0,
                  check_reproducible: bool = True):
    """Transpile offline to ``backend`` (fake or read-only real) and report ISA metrics.

    Never submits anything. Runs the pass manager twice with the same seed and
    reports whether the two ISA fingerprints coincide.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.perf_counter()
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed_transpiler)
    isa = pm.run(circuit)
    h1 = circuit_fingerprint(isa)
    h2 = h1
    if check_reproducible:
        pm2 = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                           seed_transpiler=seed_transpiler)
        h2 = circuit_fingerprint(pm2.run(circuit))
    target_ops = set(backend.target.operation_names)
    used = {inst.operation.name for inst in isa.data}
    basis_ok = used <= (target_ops | {"barrier"})
    layout = _final_layout(isa, circuit.num_qubits)
    m = circuit_metrics(isa)
    name = getattr(backend, "name", None) or str(backend)
    is_fake = "fake" in str(name).lower() or type(backend).__name__.lower().startswith("fake")
    rep = IsaReport(backend_name=str(name), is_fake=is_fake, seed_transpiler=seed_transpiler,
                    optimization_level=optimization_level, n_logical=circuit.num_qubits,
                    layout_physical=layout, depth=m["depth"], n_2q=m["n_2q"], n_1q=m["n_1q"], ops=m["ops"],
                    basis_ok=basis_ok, isa_hash=h1, reproducible=(h1 == h2), isa_hash_second=h2,
                    wall_time_s=time.perf_counter() - t0)
    return isa, rep


def _final_layout(isa, n_logical: int) -> tuple[int, ...]:
    layout = getattr(isa, "layout", None)
    if layout is None:
        return tuple(range(n_logical))
    try:
        idx = layout.final_index_layout(filter_ancillas=True)
        return tuple(int(v) for v in idx)
    except Exception:  # pragma: no cover - qiskit API variations
        try:
            init = layout.initial_layout.get_virtual_bits()
            return tuple(int(v) for q, v in list(init.items())[:n_logical])
        except Exception:
            return tuple()


# --------------------------------------------------------------------------- #
# Preregistration and hash chain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Preregistration:
    adaptive_protocol_id: str
    instance_name: str
    instance_sha256: str
    objective_id: str
    initialization_kind: str  # "noop" | "continuous_q1"
    init_theta: tuple[float, ...] | None
    rho: float
    delta0: float
    max_rounds: int
    driver_kind: str  # exact_local | sa | aer_sim | qpu_result
    shots_per_round: int
    backend: str
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    seed_transpiler: int
    optimization_level: int
    include_noop: bool
    no_feasible_policy: str
    stagnation_patience: int
    reference_identity: dict[str, Any]
    source_commit: str
    sampler_seed: int = 0
    preregistered_utc: str = ""
    # "fixed": the same (gamma, beta) every round. "penalty_normalised": gamma_r =
    # gamma * lam_pen(round 0) / lam_pen(round r) so the penalty-driven RZZ phases
    # are identical across rounds (round 0 == the historical fixed angles).
    angle_rule: str = "penalty_normalised"
    min_delta: float = 1e-6  # floor: a round is never built below this half-width (no float underflow)

    def validate(self) -> None:
        coerce_objective_id(self.objective_id)
        if not isinstance(self.min_delta, (int, float)) or not math.isfinite(self.min_delta) or self.min_delta <= 0:
            raise ValueError("min_delta must be finite and > 0")
        if self.angle_rule not in ANGLE_RULES:
            raise ValueError(f"angle_rule must be one of {ANGLE_RULES}")
        if self.driver_kind not in DRIVER_KINDS:
            raise ValueError(f"driver_kind must be one of {DRIVER_KINDS}")
        if self.adaptive_protocol_id not in (ADAPTIVE_PROTOCOL_LOCAL_EXACT, ADAPTIVE_PROTOCOL_QPU_PILOT):
            raise ValueError("unknown adaptive_protocol_id")
        if self.adaptive_protocol_id == ADAPTIVE_PROTOCOL_LOCAL_EXACT and self.driver_kind != "exact_local":
            raise ValueError("LOCAL_EXACT_ADAPTIVE_V1 requires driver_kind exact_local")
        if self.adaptive_protocol_id == ADAPTIVE_PROTOCOL_QPU_PILOT and self.driver_kind == "exact_local":
            raise ValueError("QPU_DRIVEN_ADAPTIVE_PILOT_V1 is a sampling protocol")
        if not (0.0 < self.rho < 1.0) or not math.isfinite(self.rho):
            raise ValueError("rho must be in (0,1)")
        if not math.isfinite(self.delta0) or self.delta0 <= 0:
            raise ValueError("delta0 must be finite > 0")
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int) or self.max_rounds < 1:
            raise ValueError("max_rounds must be int >= 1")
        if isinstance(self.shots_per_round, bool) or not isinstance(self.shots_per_round, int) or self.shots_per_round < 1:
            raise ValueError("shots_per_round must be int >= 1")
        if len(self.gammas) != len(self.betas) or not self.gammas:
            raise ValueError("gammas/betas must have equal non-zero length")
        if self.no_feasible_policy not in NO_FEASIBLE_POLICIES:
            raise ValueError(f"no_feasible_policy must be one of {NO_FEASIBLE_POLICIES}")
        if self.initialization_kind not in ("noop", "continuous_q1"):
            raise ValueError("initialization_kind must be 'noop' or 'continuous_q1'")
        if self.initialization_kind == "continuous_q1" and self.init_theta is None:
            raise ValueError("continuous_q1 initialisation requires init_theta")
        if self.driver_kind == "qpu_result" and self.backend != "ibm_marrakesh":
            raise ValueError("real backend must be ibm_marrakesh")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["init_theta"] = None if self.init_theta is None else list(self.init_theta)
        d["gammas"] = list(self.gammas)
        d["betas"] = list(self.betas)
        return d

    def chain_id(self) -> str:
        return sha256_obj({"schema": SCHEMA_CHAIN, **self.to_dict()})


def effective_angles(prereg: Preregistration, penalty_reference: float,
                     penalty_round: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Angles actually bound in round r under the preregistered rule."""
    if not (math.isfinite(penalty_reference) and math.isfinite(penalty_round)) or penalty_round <= 0:
        raise ValueError("penalties must be finite and > 0")
    if prereg.angle_rule == "fixed":
        return tuple(prereg.gammas), tuple(prereg.betas)
    scale = penalty_reference / penalty_round
    return tuple(g * scale for g in prereg.gammas), tuple(prereg.betas)


def round_hash(payload: dict[str, Any], parent_round_hash: str, parent_selected_hash: str) -> str:
    return sha256_text(canonical_json(payload) + parent_round_hash + parent_selected_hash)


def atomic_write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


ROUND_PAYLOAD_KEYS = (
    "schema", "experimental", "official_benchmark", "real_qpu", "adaptive_protocol_id", "objective_id",
    "instance_sha256", "block_sizes", "penalty_conflict", "penalty_onehot", "angle_rule", "shots",
    "n_evaluations", "n_quadratic_terms", "n_linear_terms", "timing_s", "decoded_summary", "isa_report",
    "round_id", "chain_id", "driver_kind", "center_before", "delta", "grids", "grid_hash",
    "qubo_hash", "n_qubits", "logical_circuit_hash", "isa_hash", "submission_config_hash", "job_id",
    "raw_result_hash", "decoded_result_hash", "selected_candidate_hash", "archive_hash",
    "selected_theta", "selected_objective", "archive_theta", "archive_objective", "status",
    "no_feasible_action", "penalty_reference", "effective_gammas", "effective_betas", "next_delta",
)


def compute_round_hash(record: dict[str, Any]) -> str:
    payload = {k: record.get(k) for k in ROUND_PAYLOAD_KEYS}
    return round_hash(payload, record["parent_round_hash"], record["parent_selected_candidate_hash"])


# --------------------------------------------------------------------------- #
# Chain store (fail-closed)
# --------------------------------------------------------------------------- #
class ChainStore:
    """Directory layout: chain_manifest.json, preregistration.json, round_000/round.json ..."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "chain_manifest.json"

    def round_dir(self, r: int) -> Path:
        return self.root / f"round_{r:03d}"

    def round_path(self, r: int) -> Path:
        return self.round_dir(r) / "round.json"

    def writer_lock(self):
        """Exclusive single-writer lock on the chain directory (fcntl; non-blocking).

        The store assumes ONE sequential writer per chain directory. A second
        concurrent writer fails closed instead of racing on manifest/round files.
        """
        store = self

        class _Lock:
            def __enter__(self_inner):
                store.root.mkdir(parents=True, exist_ok=True)
                self_inner.fd = os.open(store.root / "chain.lock", os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(self_inner.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    os.close(self_inner.fd)
                    raise ChainIntegrityError("another writer holds the chain lock") from exc
                return self_inner

            def __exit__(self_inner, *exc):
                fcntl.flock(self_inner.fd, fcntl.LOCK_UN)
                os.close(self_inner.fd)
                return False

        return _Lock()

    def init(self, prereg: Preregistration) -> dict[str, Any]:
        prereg.validate()
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
            if existing.get("chain_id") != prereg.chain_id():
                raise ChainIntegrityError("chain directory already holds a DIFFERENT preregistration")
            return existing
        manifest = {"schema": SCHEMA_CHAIN, **FLAGS, "adaptive_driver": prereg.driver_kind,
                    "real_qpu": False, "chain_id": prereg.chain_id(),
                    "preregistration_hash": sha256_obj(prereg.to_dict()), "rounds_closed": 0}
        atomic_write_json(self.root / "preregistration.json",
                          {"schema": SCHEMA_CHAIN, **FLAGS, "real_qpu": False, **prereg.to_dict()})
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def load_prereg(self) -> Preregistration:
        d = json.loads((self.root / "preregistration.json").read_text())
        for k in ("schema", "experimental", "official_benchmark", "real_qpu"):
            d.pop(k, None)
        d["init_theta"] = None if d["init_theta"] is None else tuple(d["init_theta"])
        d["gammas"] = tuple(d["gammas"])
        d["betas"] = tuple(d["betas"])
        p = Preregistration(**d)
        p.validate()
        return p

    def load_round(self, r: int) -> dict[str, Any] | None:
        p = self.round_path(r)
        if not p.exists():
            return None
        rec = json.loads(p.read_text())
        if rec.get("round_hash") != compute_round_hash(rec):
            raise ChainIntegrityError(f"round {r} is corrupted: stored round_hash differs from recomputed")
        return rec

    def n_rounds(self) -> int:
        r = 0
        while self.round_path(r).exists():
            r += 1
        return r

    def verify(self, *, deep: bool = True) -> list[str]:
        """Fail-closed verification of the chain; returns problems (empty == ok).

        Structural pass: hash links, parent/child consistency, flags, angle rule.
        ``deep`` pass (default): reloads the instance and RECOMPUTES the physics of
        every round — local grid from (centre, delta), grid/QUBO hashes, raw and
        decoded file hashes, re-decoding of the raw counts (selected candidate must
        be the best feasible sample), and for deterministic drivers (exact_local, sa)
        a re-solve that must reproduce the selected candidate. What the chain does
        NOT protect against: an author with full write access who re-runs the whole
        pipeline; the only external anchor is publishing ``chain_id`` beforehand.
        """
        problems: list[str] = []
        if not self.manifest_path.exists():
            return ["missing chain_manifest.json"]
        manifest = json.loads(self.manifest_path.read_text())
        try:
            prereg = self.load_prereg()
        except Exception as exc:  # noqa: BLE001
            return [f"preregistration invalid: {exc}"]
        raw_prereg = json.loads((self.root / "preregistration.json").read_text())
        if raw_prereg.get("experimental") is not True or raw_prereg.get("official_benchmark") is not False:
            problems.append("preregistration.json flags are not experimental=true/official_benchmark=false")
        if manifest.get("preregistration_hash") != sha256_obj(prereg.to_dict()):
            problems.append("manifest preregistration_hash != sha256 of preregistration.json content")
        if manifest.get("experimental") is not True or manifest.get("official_benchmark") is not False:
            problems.append("manifest flags are not experimental=true/official_benchmark=false")
        if manifest.get("chain_id") != prereg.chain_id():
            problems.append("manifest chain_id != preregistration chain_id")
        inst = None
        if deep:
            try:
                inst = _instance_loader().load(prereg.instance_name)
                if instance_sha256(inst) != prereg.instance_sha256:
                    problems.append("instance content sha256 != preregistration instance_sha256")
                    inst = None
            except Exception as exc:  # noqa: BLE001
                problems.append(f"cannot reload instance for deep verification: {exc}")
        if manifest.get("adaptive_driver") != prereg.driver_kind:
            problems.append("manifest adaptive_driver != preregistration driver_kind")
        prev_hash, prev_sel = GENESIS_HASH, GENESIS_HASH
        prev_rec: dict[str, Any] | None = None
        r = 0
        while self.round_path(r).exists():
            try:
                rec = self.load_round(r)
            except ChainIntegrityError as exc:
                problems.append(str(exc))
                break
            assert rec is not None
            if rec.get("round_id") != r:
                problems.append(f"round dir {r} holds round_id {rec.get('round_id')}")
            if rec.get("experimental") is not True or rec.get("official_benchmark") is not False:
                problems.append(f"round {r}: flags are not experimental=true/official_benchmark=false")
            if rec.get("objective_id") != prereg.objective_id or rec.get("instance_sha256") != prereg.instance_sha256:
                problems.append(f"round {r}: objective_id/instance_sha256 differ from preregistration")
            if rec.get("adaptive_protocol_id") != prereg.adaptive_protocol_id or rec.get("angle_rule") != prereg.angle_rule:
                problems.append(f"round {r}: protocol/angle_rule differ from preregistration")
            if rec.get("chain_id") != prereg.chain_id():
                problems.append(f"round {r}: chain_id mismatch")
            if deep and inst is not None:
                problems.extend(self._deep_check_round(r, rec, prereg, inst))
            if rec.get("driver_kind") != prereg.driver_kind:
                problems.append(f"round {r}: driver_kind mismatch")
            if rec.get("parent_round_hash") != prev_hash:
                problems.append(f"round {r}: parent_round_hash mismatch")
            if rec.get("parent_selected_candidate_hash") != prev_sel:
                problems.append(f"round {r}: parent_selected_candidate_hash mismatch")
            if prev_rec is not None:
                expected_center = prev_rec.get("archive_theta") if prev_rec.get("status") == "closed_no_feasible" \
                    else prev_rec.get("selected_theta")
                if rec.get("center_before") != expected_center:
                    problems.append(f"round {r}: center_before does not match parent selection")
                if not math.isclose(rec.get("delta", -1.0), prev_rec.get("next_delta", -2.0), rel_tol=0, abs_tol=0):
                    problems.append(f"round {r}: delta does not match parent's next_delta")
            try:
                pref, pconf = rec.get("penalty_reference"), rec.get("penalty_conflict")
                if not isinstance(pref, (int, float)) or not isinstance(pconf, (int, float)):
                    raise ValueError("penalty fields missing")
                exp_g, exp_b = effective_angles(prereg, float(pref), float(pconf))
                if list(exp_g) != rec.get("effective_gammas") or list(exp_b) != rec.get("effective_betas"):
                    problems.append(f"round {r}: effective angles violate the preregistered angle rule")
            except (TypeError, ValueError):
                problems.append(f"round {r}: missing/invalid penalty fields")
            if r == 0 and rec.get("penalty_reference") != rec.get("penalty_conflict"):
                problems.append("round 0: penalty_reference must equal its own penalty_conflict")
            if prev_rec is not None and rec.get("penalty_reference") != prev_rec.get("penalty_reference"):
                problems.append(f"round {r}: penalty_reference changed along the chain")
            if prev_rec is not None:
                prev_arch = prev_rec.get("archive_objective")
                sel = rec.get("selected_objective")
                cands = [v for v in (prev_arch, sel) if v is not None]
                expected_arch = min(cands) if cands else None
                if rec.get("archive_objective") != expected_arch:
                    problems.append(f"round {r}: archive_objective is not the running minimum")
            elif rec.get("archive_objective") != rec.get("selected_objective"):
                problems.append("round 0: archive_objective must equal its selection")
            if rec.get("real_qpu"):
                if not rec.get("job_id") or not rec.get("raw_result_hash") or rec.get("driver_kind") != "qpu_result":
                    problems.append(f"round {r}: real_qpu=true without job_id/raw hash/qpu driver")
            if rec.get("driver_kind") == "qpu_result" and rec.get("status") == "closed" and not rec.get("job_id"):
                problems.append(f"round {r}: qpu_result round closed without job_id")
            prev_hash, prev_sel = rec["round_hash"], rec["selected_candidate_hash"]
            prev_rec = rec
            r += 1
        if manifest.get("rounds_closed") != r:
            problems.append(f"manifest rounds_closed {manifest.get('rounds_closed')} != {r} round files")
        return problems

    def _deep_check_round(self, r: int, rec: dict[str, Any], prereg: Preregistration, inst: Instance) -> list[str]:
        """Recompute the physics of round ``r`` and compare with what is recorded."""
        probs: list[str] = []
        try:
            grid = ag.build_local_grid(inst, tuple(rec["center_before"]), float(rec["delta"]),
                                       include_noop=prereg.include_noop)
        except Exception as exc:  # noqa: BLE001
            return [f"round {r}: cannot rebuild local grid from centre/delta: {exc}"]
        if [list(o) for o in grid.options] != rec.get("grids"):
            probs.append(f"round {r}: recorded grids differ from grid rebuilt from centre/delta")
        if grid_hash(grid) != rec.get("grid_hash"):
            probs.append(f"round {r}: grid_hash differs from recomputed")
        table = ag.conflict_table(inst, grid)
        qubo = build_hetero_qubo(inst, grid, prereg.objective_id, table=table)
        if qubo.canonical_hash() != rec.get("qubo_hash"):
            probs.append(f"round {r}: qubo_hash differs from recomputed")
        if qubo.n_qubits != rec.get("n_qubits") or [len(o) for o in grid.options] != rec.get("block_sizes"):
            probs.append(f"round {r}: n_qubits/block_sizes differ from recomputed")
        if qubo.penalty_conflict != rec.get("penalty_conflict") or qubo.penalty_onehot != rec.get("penalty_onehot"):
            probs.append(f"round {r}: penalties differ from recomputed")
        raw_path = self.round_dir(r) / "raw_counts.json"
        dec_path = self.round_dir(r) / "decoded.json"
        if rec.get("raw_result_hash") is not None:
            if not raw_path.exists():
                probs.append(f"round {r}: raw_counts.json missing")
            else:
                raw_doc = json.loads(raw_path.read_text())
                counts = {str(k): int(v) for k, v in raw_doc.get("counts", {}).items()}
                if sha256_obj({"bit_order": "qiskit", "counts": dict(sorted(counts.items()))}) != rec["raw_result_hash"]:
                    probs.append(f"round {r}: raw_counts.json content does not match raw_result_hash")
                else:
                    try:
                        decoded = decode_counts(qubo, counts, bit_order="qiskit", kind="counts")
                    except ValueError as exc:
                        probs.append(f"round {r}: raw counts undecodable: {exc}")
                        decoded = None
                    if decoded is not None:
                        if sha256_obj(decoded.to_dict()) != rec.get("decoded_result_hash"):
                            probs.append(f"round {r}: decoded_result_hash differs from re-decoding")
                        best = decoded.best_feasible
                        if best is None:
                            if rec.get("status") != "closed_no_feasible" or rec.get("selected_theta") is not None:
                                probs.append(f"round {r}: no feasible sample but a selection is recorded")
                        else:
                            if rec.get("selected_theta") != list(best.theta) or rec.get("selected_objective") != best.objective:
                                probs.append(f"round {r}: selected candidate is not the best feasible sample of the raw counts")
                            if rec.get("selected_candidate_hash") != candidate_hash(best):
                                probs.append(f"round {r}: selected_candidate_hash differs from re-decoded candidate")
                # decoded.json is a RECONSTRUCTIBLE expansion of raw_counts.json (see
                # reconstruct_decoded); it may be absent from a compact pack. When present it
                # must match decoded_result_hash; when absent, the re-decoding above is the check.
                if dec_path.exists():
                    dec_doc = json.loads(dec_path.read_text())
                    dec_doc = {k: v for k, v in dec_doc.items() if k not in ("schema", "experimental", "official_benchmark")}
                    if sha256_obj(dec_doc) != rec.get("decoded_result_hash"):
                        probs.append(f"round {r}: decoded.json content does not match decoded_result_hash")
        # deterministic drivers: the selection must be reproducible by re-solving
        if prereg.driver_kind in ("exact_local", "sa") and rec.get("status") == "closed":
            if prereg.driver_kind == "exact_local":
                rs = ag.exact_local_solve(inst, prereg.objective_id, grid, table)
            else:
                rs = ag.sa_local_solve(inst, prereg.objective_id, grid, table, seed=prereg.sampler_seed + r)
            if rs.theta is None or list(rs.theta) != rec.get("selected_theta") or rs.objective != rec.get("selected_objective"):
                probs.append(f"round {r}: deterministic re-solve does not reproduce the selected candidate")
        # archive must be the running minimum of selections
        return probs

    def assert_can_start(self, r: int, prereg: Preregistration) -> tuple[str, str, dict[str, Any] | None]:
        """Return (parent_round_hash, parent_selected_hash, parent_record) or raise."""
        problems = self.verify()
        if problems:
            raise ChainIntegrityError("chain verification failed: " + "; ".join(problems))
        if self.round_path(r).exists():
            raise ChainIntegrityError(f"round {r} already exists (idempotent driver refuses to rebuild)")
        if r == 0:
            return GENESIS_HASH, GENESIS_HASH, None
        parent = self.load_round(r - 1)
        if parent is None:
            raise ChainIntegrityError(f"parent round {r - 1} missing")
        if parent.get("status") not in ("closed", "closed_no_feasible"):
            raise ChainIntegrityError(f"parent round {r - 1} not closed (status {parent.get('status')})")
        if parent.get("objective_id") != prereg.objective_id or parent.get("instance_sha256") != prereg.instance_sha256:
            raise ChainIntegrityError("parent round belongs to a different objective/instance")
        return parent["round_hash"], parent["selected_candidate_hash"], parent

    def record_round(self, rec: dict[str, Any]) -> None:
        rec["round_hash"] = compute_round_hash(rec)
        atomic_write_json(self.round_path(rec["round_id"]), rec)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["rounds_closed"] = self.n_rounds()
        manifest["last_round_hash"] = rec["round_hash"]
        atomic_write_json(self.manifest_path, manifest)


# --------------------------------------------------------------------------- #
# Local drivers (exact / SA / Aer). No QPU driver exists in this module.
# --------------------------------------------------------------------------- #
def _sample_aer(circuit, shots: int, seed: int) -> dict[str, int]:
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=seed)
    job = sim.run(circuit, shots=shots)
    return {str(k): int(v) for k, v in job.result().get_counts().items()}  # qiskit order keys


def run_round(store: ChainStore, prereg: Preregistration, r: int, *, backend=None,
              transpile: bool = True) -> dict[str, Any]:
    """Execute round ``r`` under the preregistration with a LOCAL driver and persist it.

    ``driver_kind`` exact_local: exact B&B on the local grid (the 'sample' is the optimum).
    ``sa``: adaptive_grid.sa_local_solve (deterministic seed = sampler_seed + r).
    ``aer_sim``: fixed-angle QAOA circuit sampled on AerSimulator (seed = sampler_seed + r),
    decoded with the same decoder a QPU result would go through.
    ``qpu_result``: REFUSED here — no submission code exists in this module.
    """
    prereg.validate()
    if prereg.driver_kind == "qpu_result":
        raise ChainIntegrityError("this module has no QPU driver; a qpu_result round needs an "
                                  "authenticated job_id and raw counts produced elsewhere")
    with store.writer_lock():
        return _run_round_locked(store, prereg, r, backend=backend, transpile=transpile)


def _run_round_locked(store: ChainStore, prereg: Preregistration, r: int, *, backend, transpile: bool) -> dict[str, Any]:
    inst_loader = _instance_loader()
    inst = inst_loader.load(prereg.instance_name)
    if instance_sha256(inst) != prereg.instance_sha256:
        raise ChainIntegrityError("instance sha256 differs from preregistration")
    parent_hash, parent_sel, parent = store.assert_can_start(r, prereg)
    t0 = time.perf_counter()
    if r == 0:
        centers = (0.0,) * inst.n if prereg.initialization_kind == "noop" else \
            tuple(min(inst.hmax, max(inst.hmin, float(t))) for t in (prereg.init_theta or ()))
        delta = prereg.delta0
        archive_theta, archive_obj = None, None
    else:
        assert parent is not None
        centers = tuple(parent["archive_theta"] if parent["status"] == "closed_no_feasible" else parent["selected_theta"])
        delta = float(parent["next_delta"])
        archive_theta = None if parent["archive_theta"] is None else tuple(parent["archive_theta"])
        archive_obj = parent["archive_objective"]
    if delta < prereg.min_delta:
        raise ChainIntegrityError(f"round {r}: delta {delta!r} below preregistered min_delta {prereg.min_delta!r}")
    for k in range(r):
        prev = store.load_round(k)
        if prev is not None and prev.get("center_before") == list(centers) and prev.get("delta") == delta:
            raise ChainIntegrityError(f"round {r}: state (centres, delta) already visited at round {k} (cycle)")
    grid = ag.build_local_grid(inst, centers, delta, include_noop=prereg.include_noop)
    table = ag.conflict_table(inst, grid)
    qubo = build_hetero_qubo(inst, grid, prereg.objective_id, table=table)
    penalty_reference = qubo.penalty_conflict if r == 0 else float(parent["penalty_reference"])  # type: ignore[index]
    eff_gammas, eff_betas = effective_angles(prereg, penalty_reference, qubo.penalty_conflict)
    rec: dict[str, Any] = {
        "schema": SCHEMA_ROUND, **FLAGS, "real_qpu": False, "adaptive_protocol_id": prereg.adaptive_protocol_id,
        "round_id": r, "chain_id": prereg.chain_id(), "driver_kind": prereg.driver_kind,
        "objective_id": prereg.objective_id, "instance_sha256": prereg.instance_sha256,
        "parent_round_hash": parent_hash, "parent_selected_candidate_hash": parent_sel,
        "center_before": list(centers), "delta": delta, "grids": [list(o) for o in grid.options],
        "grid_hash": grid_hash(grid), "qubo_hash": qubo.canonical_hash(), "n_qubits": qubo.n_qubits,
        "block_sizes": [len(o) for o in grid.options],
        "penalty_conflict": qubo.penalty_conflict, "penalty_onehot": qubo.penalty_onehot,
        "penalty_reference": penalty_reference, "effective_gammas": list(eff_gammas),
        "effective_betas": list(eff_betas), "angle_rule": prereg.angle_rule,
        "n_quadratic_terms": len(qubo.quadratic), "n_linear_terms": len(qubo.linear),
        "logical_circuit_hash": None, "isa_hash": None, "isa_report": None,
        "submission_config_hash": None, "job_id": None, "raw_result_hash": None,
        "decoded_result_hash": None, "selected_candidate_hash": GENESIS_HASH, "archive_hash": GENESIS_HASH,
        "selected_theta": None, "selected_objective": None, "archive_theta": None if archive_theta is None else list(archive_theta),
        "archive_objective": archive_obj, "status": "built", "no_feasible_action": None,
        "timing_s": {}, "shots": None, "decoded_summary": None,
    }
    circuit = None
    if prereg.driver_kind == "aer_sim" or transpile:
        try:
            circuit = build_fixed_angle_circuit(qubo, eff_gammas, eff_betas)
            rec["logical_circuit_hash"] = circuit_fingerprint(circuit)
            rec["logical_metrics"] = circuit_metrics(circuit)
        except ImportError:
            circuit = None
    if circuit is not None and transpile and backend is not None:
        _, isa_rep = transpile_isa(circuit, backend, seed_transpiler=prereg.seed_transpiler,
                                   optimization_level=prereg.optimization_level)
        rec["isa_hash"] = isa_rep.isa_hash
        rec["isa_report"] = asdict(isa_rep)
        rec["submission_config_hash"] = sha256_obj({
            "backend": prereg.backend, "shots": prereg.shots_per_round, "gammas": list(eff_gammas),
            "betas": list(eff_betas), "angle_rule": prereg.angle_rule, "seed_transpiler": prereg.seed_transpiler,
            "optimization_level": prereg.optimization_level, "isa_hash": isa_rep.isa_hash,
            "qubo_hash": qubo.canonical_hash()})
    t_build = time.perf_counter() - t0
    # --- sampling / solving
    t1 = time.perf_counter()
    if prereg.driver_kind == "exact_local":
        rs = ag.exact_local_solve(inst, prereg.objective_id, grid, table)
        raw = {encode_choice(qubo, rs.choice)[::-1]: 1} if rs.choice is not None else None
        rec["n_evaluations"] = rs.n_evaluations
    elif prereg.driver_kind == "sa":
        rs = ag.sa_local_solve(inst, prereg.objective_id, grid, table, seed=prereg.sampler_seed + r)
        raw = {encode_choice(qubo, rs.choice)[::-1]: 1} if rs.choice is not None else None
        rec["n_evaluations"] = rs.n_evaluations
    else:  # aer_sim
        if circuit is None:
            raise ChainIntegrityError("qiskit unavailable: cannot run an aer_sim round")
        raw = _sample_aer(circuit, prereg.shots_per_round, prereg.sampler_seed + r)
        rec["shots"] = prereg.shots_per_round
    rec["timing_s"] = {"build": t_build, "sample_or_solve": time.perf_counter() - t1}
    decoded = None
    if raw is not None:
        rec["raw_result_hash"] = sha256_obj({"bit_order": "qiskit", "counts": {k: int(v) for k, v in sorted(raw.items())}})
        decoded = decode_counts(qubo, raw, bit_order="qiskit", kind="counts")
        rec["decoded_summary"] = {"n_shots": decoded.n_shots, "n_unique": decoded.n_unique,
                                  "feasibility_rate": decoded.feasibility_rate,
                                  "onehot_valid_rate": decoded.onehot_valid_rate,
                                  "best_raw": asdict(decoded.best_raw),
                                  "best_feasible": None if decoded.best_feasible is None else asdict(decoded.best_feasible)}
        rec["decoded_result_hash"] = sha256_obj(decoded.to_dict())
        atomic_write_json(store.round_dir(r) / "raw_counts.json",
                          {"schema": SCHEMA_ROUND, **FLAGS, "bit_order": "qiskit", "driver_kind": prereg.driver_kind,
                           "counts": dict(sorted(raw.items()))})
        atomic_write_json(store.round_dir(r) / "decoded.json", {"schema": SCHEMA_ROUND, **FLAGS, **decoded.to_dict()})
    # --- selection + archive (elitist)
    best = decoded.best_feasible if decoded is not None else None
    if best is not None:
        rec["selected_theta"] = list(best.theta)
        rec["selected_objective"] = best.objective
        rec["selected_candidate_hash"] = candidate_hash(best)
        if archive_obj is None or best.objective < archive_obj:
            archive_theta, archive_obj = best.theta, best.objective
        rec["status"] = "closed"
        rec["next_delta"] = delta * prereg.rho
    else:
        rec["status"] = "closed_no_feasible"
        if archive_theta is None:
            rec["no_feasible_action"] = "stop_no_incumbent"
            rec["next_delta"] = delta
        elif prereg.no_feasible_policy == "stop":
            rec["no_feasible_action"] = "stop"
            rec["next_delta"] = delta
        elif prereg.no_feasible_policy == "keep_center_shrink":
            rec["no_feasible_action"] = "keep_center_shrink"
            rec["next_delta"] = delta * prereg.rho
        else:  # keep_center_retry_once
            retried = parent is not None and parent.get("no_feasible_action") == "keep_center_retry_once"
            rec["no_feasible_action"] = "stop_after_retry" if retried else "keep_center_retry_once"
            rec["next_delta"] = delta
        # centre for next round is the archive (never invented)
        rec["selected_theta"] = None
        rec["selected_candidate_hash"] = sha256_obj({"no_feasible": True, "round_id": r})
    rec["archive_theta"] = None if archive_theta is None else list(archive_theta)
    rec["archive_objective"] = archive_obj
    rec["archive_hash"] = sha256_obj({"theta": rec["archive_theta"], "objective": archive_obj})
    rec["timing_s"]["total"] = time.perf_counter() - t0
    store.record_round(rec)
    return rec


def reconstruct_decoded(store: ChainStore, r: int, *, write: bool = False) -> dict[str, Any]:
    """Rebuild round ``r``'s ``decoded.json`` from ``raw_counts.json`` + the round's grid.

    Deterministic: same decoder, same grid (rebuilt from centre/delta), same raw counts.
    The result's hash must equal ``decoded_result_hash``; otherwise ChainIntegrityError.
    With ``write=True`` the file is (re)written with the same canonical writer, so the bytes
    are identical to the original expansion.
    """
    prereg = store.load_prereg()
    rec = store.load_round(r)
    if rec is None:
        raise ChainIntegrityError(f"round {r} missing")
    inst = _instance_loader().load(prereg.instance_name)
    grid = ag.build_local_grid(inst, tuple(rec["center_before"]), float(rec["delta"]), include_noop=prereg.include_noop)
    qubo = build_hetero_qubo(inst, grid, prereg.objective_id)
    raw = json.loads((store.round_dir(r) / "raw_counts.json").read_text())
    counts = {str(k): int(v) for k, v in raw["counts"].items()}
    decoded = decode_counts(qubo, counts, bit_order="qiskit", kind="counts")
    doc = decoded.to_dict()
    if sha256_obj(doc) != rec.get("decoded_result_hash"):
        raise ChainIntegrityError(f"round {r}: reconstructed decoding does not match decoded_result_hash")
    full = {"schema": SCHEMA_ROUND, **FLAGS, **doc}
    if write:
        atomic_write_json(store.round_dir(r) / "decoded.json", full)
    return full


def chain_should_continue(rec: dict[str, Any], prereg: Preregistration) -> bool:
    if rec["round_id"] + 1 >= prereg.max_rounds:
        return False
    if rec["no_feasible_action"] in ("stop", "stop_no_incumbent", "stop_after_retry"):
        return False
    return True


def run_chain(root: Path, prereg: Preregistration, *, backend=None, transpile: bool = True) -> dict[str, Any]:
    """Run (or resume) a full local chain. Idempotent: existing rounds are verified, not rebuilt."""
    store = ChainStore(root)
    store.init(prereg)
    r = store.n_rounds()
    last = store.load_round(r - 1) if r > 0 else None
    stagnation = 0
    while True:
        if last is not None and not chain_should_continue(last, prereg):
            break
        if r >= prereg.max_rounds:
            break
        if last is not None and float(last["next_delta"]) < prereg.min_delta:
            break  # preregistered floor reached: stop cleanly, archive kept
        rec = run_round(store, prereg, r, backend=backend, transpile=transpile)
        improved = last is None or rec["archive_objective"] != (last or {}).get("archive_objective")
        stagnation = 0 if improved else stagnation + 1
        last = rec
        r += 1
        if stagnation >= prereg.stagnation_patience:
            break
    problems = store.verify()
    if problems:
        raise ChainIntegrityError("post-run verification failed: " + "; ".join(problems))
    return {"chain_id": prereg.chain_id(), "n_rounds": store.n_rounds(), "archive_theta": last["archive_theta"] if last else None,
            "archive_objective": last["archive_objective"] if last else None, "problems": problems}


def _instance_loader():
    from ..io.loader import InstanceLoader

    return InstanceLoader()


def make_preregistration(inst: Instance, *, protocol: str, driver_kind: str, objective_id: str,
                         rho: float = 0.5, delta0: float | None = None, max_rounds: int = 4,
                         shots_per_round: int = 512, backend: str = "aer_simulator",
                         gammas: Sequence[float] = (0.4,), betas: Sequence[float] = (0.3,),
                         seed_transpiler: int = 1, optimization_level: int = 0,
                         initialization_kind: str = "noop", init_theta: Sequence[float] | None = None,
                         include_noop: bool = False, no_feasible_policy: str = "keep_center_retry_once",
                         stagnation_patience: int = 3, reference_identity: dict[str, Any] | None = None,
                         source_commit: str = "unknown", sampler_seed: int = 0) -> Preregistration:
    p = Preregistration(
        adaptive_protocol_id=protocol, instance_name=inst.name, instance_sha256=instance_sha256(inst),
        objective_id=coerce_objective_id(objective_id).value, initialization_kind=initialization_kind,
        init_theta=None if init_theta is None else tuple(float(t) for t in init_theta), rho=rho,
        delta0=float(inst.hmax) if delta0 is None else float(delta0), max_rounds=max_rounds,
        driver_kind=driver_kind, shots_per_round=shots_per_round, backend=backend,
        gammas=tuple(float(g) for g in gammas), betas=tuple(float(b) for b in betas),
        seed_transpiler=seed_transpiler, optimization_level=optimization_level, include_noop=include_noop,
        no_feasible_policy=no_feasible_policy, stagnation_patience=stagnation_patience,
        reference_identity=dict(reference_identity or {}), source_commit=source_commit,
        sampler_seed=sampler_seed, preregistered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    p.validate()
    return p


__all__ = [n for n in dir() if not n.startswith("_")]
_ = field  # keep dataclasses.field import used for future extension
