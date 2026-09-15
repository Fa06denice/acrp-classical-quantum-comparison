"""Guarded local QPU API routes (Phase 7).

``register_qpu_routes(app, ...)`` attaches the QPU workflow endpoints to an
existing FastAPI ``app`` (see :func:`acrpq.dashboard.api.create_app`). Everything
here is local, fail-closed and side-effect-free with respect to IBM:

* prepare/dry-run build the QUBO (read-only), score backends (Phase 4), prepare a
  Phase-5 artefact against a **fake** backend (never submittable) and an immutable
  execution plan (Phase 6);
* confirmation requires an exact, single-use, expiring nonce bound to the run and
  all its hashes/backend/budget, plus an explicit consent phrase, and only ever
  moves a run to ``awaiting_confirmation`` — it never submits;
* a real submission endpoint is intentionally absent; even a confirmed run cannot
  submit while the server feature flags withhold it (:mod:`qpu_flags`);
* mutating routes require a loopback client.

FastAPI/qiskit are imported lazily inside the handlers that need them.
"""

# NB: no ``from __future__ import annotations`` here — FastAPI must see the real
# ``Request`` type on the route handlers (a stringified annotation would be
# unresolvable because ``Request`` is imported inside ``register_qpu_routes``).

import os
import statistics
import time
from dataclasses import asdict
from typing import Any, Callable

from ..exceptions import ACRPError
from . import ibm_runner, qpu_confirm, qpu_flags, qpu_runs, qpu_store
from .backend_scoring import BackendSnapshot, WorkloadRequirements, rank_backends
from .hardware_validation import (
    HardwareValidationRequest,
    _expected_phase3_params,
    canonical_qubo_hash,
    prepare_hardware_bundle,
)
from .qpu_ibm_factory import GatewayNotEnabled
from .qpu_orchestrator import build_execution_plan, dry_run

_MAX_SNAPSHOTS = 64
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_ALLOWED_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "testclient", "testserver", "[::1]", "::1"})
_MAX_BODY_BYTES = 262_144            # 256 KiB — snapshots are small
_DEFAULT_RATE = (120, 60.0)          # (max requests, window seconds) per client+path

# Known local fake backends (name -> lazy loader). Real backends need an account.
_FAKE_BACKENDS = {
    "fake_manila": "FakeManilaV2",
    "fake_lagos": "FakeLagosV2",
    "fake_sherbrooke": "FakeSherbrooke",
}


class BodyLimitMiddleware:
    """ASGI middleware that caps the REAL received body size for QPU POST routes.

    Counts bytes off the receive channel (not a possibly-lying Content-Length),
    stops as soon as the cap is exceeded, and returns 413 before the handler runs.
    Handles chunked / no-Content-Length bodies. Other paths pass straight through.
    """

    def __init__(self, app: Any, *, max_bytes: int, prefix: str = "/api/qpu/") -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.prefix = prefix

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (scope.get("type") != "http" or scope.get("method") != "POST"
                or not scope.get("path", "").startswith(self.prefix)):
            await self.app(scope, receive, send)
            return
        buffered: list[dict[str, Any]] = []
        total = 0
        over = False
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b""))
            buffered.append(message)
            more = message.get("more_body", False)
            if total > self.max_bytes:
                over = True
                break
        if over:
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body",
                        "body": b'{"detail":"request body too large"}'})
            return
        it = iter(buffered)

        async def replay() -> dict[str, Any]:
            try:
                return next(it)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)


def _strict_int(v: Any, name: str, *, lo: int | None = None, hi: int | None = None) -> int:
    """Reject bool/str/float coercions (no int(True)/int('5')); enforce bounds."""
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name} must be an int (not bool/str/float); got {type(v).__name__}")
    if lo is not None and v < lo:
        raise ValueError(f"{name} must be >= {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"{name} must be <= {hi}")
    return v


def _strict_float(v: Any, name: str, *, lo: float | None = None, hi: float | None = None) -> float:
    import math
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number (no NaN/Inf/bool); got {v!r}")
    fv = float(v)
    if lo is not None and fv < lo:
        raise ValueError(f"{name} must be >= {lo}")
    if hi is not None and fv > hi:
        raise ValueError(f"{name} must be <= {hi}")
    return fv


_PREPARE_BODY_KEYS = frozenset({
    "instance", "n_theta", "n_q", "shots", "transpiler_seed", "backend_requested",
    "gammas", "betas", "snapshots", "max_jobs", "max_quantum_seconds", "optimization_level",
    "mode", "strategy", "angle_points", "objective_id",
})


def _resolve_fake_backend(name: str) -> Any:
    cls_name = _FAKE_BACKENDS.get(name)
    if cls_name is None:
        raise ValueError(f"unknown local fake backend {name!r}; "
                         f"choose one of {sorted(_FAKE_BACKENDS)}")
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, cls_name)()


def _resolve_real_backend(
    name: str, *, env: Callable[[str], str | None] = os.environ.get,
) -> Any:
    """Resolve a real IBM backend through the SDK's private account store.

    This path is read-only and available only when the operator explicitly
    enables the runtime factory.  The token is never accepted by this API.
    """
    from .qpu_ibm_factory import GatewayNotEnabled, runtime_factory_enabled

    if not runtime_factory_enabled(env=env):
        raise GatewayNotEnabled(
            "real IBM backend discovery is disabled "
            "(set ACRPQ_IBM_RUNTIME_FACTORY_ENABLED=true)")
    if not name.startswith("ibm_"):
        raise ValueError("real backend name must start with 'ibm_'")
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService().backend(name)


def _real_backend_snapshot(backend: Any, *, now: float | None = None) -> BackendSnapshot:
    """Collect a conservative, JSON-native snapshot from an IBM backend."""
    ts = time.time() if now is None else now
    status = backend.status()
    props = backend.properties()
    two_q_errors: list[float] = []
    if props is not None:
        for gate in props.gates:
            if len(gate.qubits) != 2:
                continue
            params = {p.name: p.value for p in gate.parameters}
            err = params.get("gate_error")
            if isinstance(err, (int, float)) and 0.0 <= float(err) <= 1.0:
                two_q_errors.append(float(err))
    readout_errors: list[float] = []
    if props is not None:
        for qubit in range(int(backend.num_qubits)):
            try:
                err = props.readout_error(qubit)
            except Exception:  # noqa: BLE001 - unavailable calibration stays unknown
                continue
            if isinstance(err, (int, float)) and 0.0 <= float(err) <= 1.0:
                readout_errors.append(float(err))
    coupling = getattr(backend, "coupling_map", None)
    edges = list(coupling.get_edges()) if coupling is not None else []
    family_raw = getattr(backend, "processor_type", None)
    family = family_raw.get("family", "IBM") if isinstance(family_raw, dict) else "IBM"
    clops_raw = getattr(backend, "clops", None)
    clops = float(clops_raw) if isinstance(clops_raw, (int, float)) else None
    unknown: list[str] = []
    metrics = {
        "pending_jobs": getattr(status, "pending_jobs", None),
        "two_qubit_error": min(two_q_errors) if two_q_errors else None,
        "two_qubit_error_layered": None,
        "readout_error": statistics.median(readout_errors) if readout_errors else None,
        "clops": clops,
        "avg_coupling_degree": (len(edges) / int(backend.num_qubits) if edges else None),
    }
    unknown.extend(k for k, v in metrics.items() if v is None)
    pending_raw = metrics["pending_jobs"]
    pending_jobs = (
        int(pending_raw)
        if isinstance(pending_raw, int) and not isinstance(pending_raw, bool)
        else None
    )
    return BackendSnapshot(
        name=str(backend.name), region="us-east", family=str(family),
        physical_qubits=int(backend.num_qubits), programmable_qubits=int(backend.num_qubits),
        operational=bool(status.operational), maintenance=False,
        pending_jobs=pending_jobs, two_qubit_error=metrics["two_qubit_error"],
        two_qubit_error_layered=None, readout_error=metrics["readout_error"], clops=clops,
        avg_coupling_degree=metrics["avg_coupling_degree"], data_ts=ts,
        provenance={
            "source": "qiskit_ibm_runtime", "backend": str(backend.name),
            "properties_last_update": str(getattr(props, "last_update_date", "unknown")),
        }, unknown_fields=tuple(unknown), synthetic=False,
    )


def _resolve_backend_and_snapshots(
    name: str, client_snapshots: list[dict[str, Any]], *,
    env: Callable[[str], str | None] = os.environ.get,
) -> tuple[Any, list[BackendSnapshot], float]:
    if name in _FAKE_BACKENDS:
        return (_resolve_fake_backend(name),
                [_snapshot_from_dict(s) for s in client_snapshots], 0.0)
    backend = _resolve_real_backend(name, env=env)
    snapshot = _real_backend_snapshot(backend)
    return backend, [snapshot], snapshot.data_ts


def _snapshot_from_dict(d: dict[str, Any]) -> BackendSnapshot:
    if not isinstance(d, dict):
        raise ValueError("each snapshot must be an object")
    allowed = set(BackendSnapshot.__dataclass_fields__)
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"unknown snapshot field(s): {sorted(unknown)}")
    payload = dict(d)
    payload["synthetic"] = True  # the API can only ever see synthetic snapshots
    if "provenance" in payload and not isinstance(payload["provenance"], dict):
        raise ValueError("provenance must be an object")
    for tup in ("unknown_fields",):
        if tup in payload and isinstance(payload[tup], list):
            payload[tup] = tuple(payload[tup])
    return BackendSnapshot(**payload)


def _build_workload(qubo: Any, *, shots: int, decision_ts: float, qaoa_reps: int,
                    max_quantum_seconds: float) -> WorkloadRequirements:
    n = qubo.n_qubits
    pairs = n * (n - 1) / 2 if n > 1 else 1.0
    density = min(1.0, len(qubo.quadratic) / pairs) if pairs else 0.0
    return WorkloadRequirements(
        logical_qubits=n, est_two_qubit_gates=len(qubo.quadratic),
        logical_depth=2 * qaoa_reps, interaction_density=density, shots=shots,
        max_quantum_seconds=max_quantum_seconds, decision_ts=decision_ts)


def _floatlist(v: Any, name: str) -> tuple[float, ...]:
    if not isinstance(v, list) or not v:
        raise ValueError(f"{name} must be a non-empty list of numbers")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        out.append(float(x))
    return tuple(out)


def _prepare_bundle_and_plan(
    loader: Any, body: dict[str, Any], *, decision_ts: float,
    env: Callable[[str], str | None] = os.environ.get,
):
    """Shared prepare path for /prepare and /dry-run (no persistence, no submit)."""
    from ..quantum.qubo import build_qubo

    if not isinstance(body, dict):
        raise ValueError("request body must be an object")
    unknown = set(body) - _PREPARE_BODY_KEYS
    if unknown:
        raise ValueError(f"unknown request field(s): {sorted(unknown)}")
    instance = body.get("instance")
    if not isinstance(instance, str) or not instance.strip():
        raise ValueError("'instance' must be a non-empty string")
    # STRICT: no int(True)/int('5')/float('nan') coercion anywhere
    n_theta = _strict_int(body.get("n_theta", 2), "n_theta", lo=1, hi=25)
    n_q = _strict_int(body.get("n_q", 1), "n_q", lo=1, hi=10)
    shots = _strict_int(body.get("shots", 1024), "shots", lo=1, hi=1_000_000)
    seed = _strict_int(body.get("transpiler_seed", 0), "transpiler_seed", lo=0, hi=2**31 - 1)
    max_jobs = _strict_int(body.get("max_jobs", 1), "max_jobs", lo=1, hi=10_000)
    opt_level = _strict_int(body.get("optimization_level", 0), "optimization_level", lo=0, hi=3)
    max_qs = _strict_float(body.get("max_quantum_seconds", 30.0), "max_quantum_seconds", lo=0.0)
    objective_id = body.get("objective_id", "quadratic_control_cost_v1")
    backend = body.get("backend_requested")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("'backend_requested' must be a non-empty string")
    gammas = _floatlist(body.get("gammas"), "gammas")
    betas = _floatlist(body.get("betas"), "betas")
    snapshots = body.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("'snapshots' must be a non-empty list")
    if len(snapshots) > _MAX_SNAPSHOTS:
        raise ValueError(f"too many snapshots (> {_MAX_SNAPSHOTS})")

    inst = loader.load(instance)
    resolved_backend, snaps, live_ts = _resolve_backend_and_snapshots(
        backend, snapshots, env=env)
    # Hardware capacity, not the local statevector default, is the QUBO budget.
    # The term/depth/2Q budgets still fail closed later in the same pipeline.
    qubo = build_qubo(
        inst, n_theta=n_theta, n_q=n_q,
        max_qubits=int(resolved_backend.num_qubits),
        objective_id=objective_id,
    )
    effective_ts = live_ts or decision_ts
    workload = _build_workload(qubo, shots=shots, decision_ts=effective_ts, qaoa_reps=len(gammas),
                               max_quantum_seconds=max_qs)
    decision = rank_backends(workload, snaps)
    if decision["selected"] != backend:
        raise ValueError(
            f"scored selection {decision['selected']!r} != requested backend {backend!r}")

    request = HardwareValidationRequest(
        instance_id=instance, qubo_sha256=canonical_qubo_hash(qubo), n_binary_vars=qubo.n_qubits,
        variable_meaning=tuple(f"x{b}" for b in range(qubo.n_qubits)), qaoa_reps=len(gammas),
        gammas=gammas, betas=betas, shots=shots, transpiler_seed=seed,
        backend_requested=backend, max_jobs=max_jobs,
        max_quantum_seconds=max_qs, n_theta=n_theta, n_q=n_q,
        objective_id=qubo.objective_id,
        optimization_level=opt_level, decision_ts=effective_ts)
    bundle = prepare_hardware_bundle(request, qubo, resolved_backend, decision=decision,
                                     workload=workload)
    plan = build_execution_plan(bundle)
    return bundle, plan, decision


_FULL_QAOA_BODY_KEYS = frozenset({
    "instance", "n_theta", "n_q", "shots", "transpiler_seed", "backend_requested",
    "snapshots", "max_quantum_seconds", "optimization_level", "strategy", "angle_points",
    "reps", "total_shots_cap",
})


def _prepare_full_qaoa(loader: Any, body: dict[str, Any], *, decision_ts: float):
    """Build N single-point bundles + a bounded full-QAOA plan (no persist/submit)."""
    from ..quantum.qubo import build_qubo
    from .qpu_orchestrator import build_full_qaoa_plan, generate_angle_points

    if not isinstance(body, dict):
        raise ValueError("request body must be an object")
    unknown = set(body) - _FULL_QAOA_BODY_KEYS
    if unknown:
        raise ValueError(f"unknown request field(s): {sorted(unknown)}")
    instance = body.get("instance")
    if not isinstance(instance, str) or not instance.strip():
        raise ValueError("'instance' must be a non-empty string")
    n_theta = _strict_int(body.get("n_theta", 2), "n_theta", lo=1, hi=25)
    n_q = _strict_int(body.get("n_q", 1), "n_q", lo=1, hi=10)
    shots = _strict_int(body.get("shots", 512), "shots", lo=1, hi=1_000_000)
    seed = _strict_int(body.get("transpiler_seed", 1), "transpiler_seed", lo=0, hi=2**31 - 1)
    reps = _strict_int(body.get("reps", 1), "reps", lo=1, hi=8)
    max_qs = _strict_float(body.get("max_quantum_seconds", 60.0), "max_quantum_seconds", lo=0.0)
    total_cap = _strict_int(body.get("total_shots_cap", 10_000), "total_shots_cap", lo=1)
    opt = _strict_int(body.get("optimization_level", 0), "optimization_level", lo=0, hi=3)
    backend = body.get("backend_requested")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("'backend_requested' must be a non-empty string")
    snapshots = body.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots or len(snapshots) > _MAX_SNAPSHOTS:
        raise ValueError("'snapshots' must be a non-empty list within the cap")
    strategy = body.get("strategy", "deterministic_sweep")
    provided = None
    if strategy == "provided_points":
        pts = body.get("angle_points")
        if not isinstance(pts, list) or not pts:
            raise ValueError("provided_points needs a non-empty 'angle_points' list")
        provided = [(_floatlist(p.get("gammas"), "gammas"), _floatlist(p.get("betas"), "betas"))
                    for p in pts]
    points = generate_angle_points(strategy, reps=reps, provided=provided)

    inst = loader.load(instance)
    fake_backend = _resolve_fake_backend(backend)
    qubo = build_qubo(
        inst, n_theta=n_theta, n_q=n_q, max_qubits=int(fake_backend.num_qubits))
    workload = _build_workload(qubo, shots=shots, decision_ts=decision_ts, qaoa_reps=reps,
                               max_quantum_seconds=max_qs)
    decision = rank_backends(workload, [_snapshot_from_dict(s) for s in snapshots])
    if decision["selected"] != backend:
        raise ValueError(f"scored selection {decision['selected']!r} != requested {backend!r}")
    bundles = []
    for gammas, betas in points:
        req = HardwareValidationRequest(
            instance_id=instance, qubo_sha256=canonical_qubo_hash(qubo), n_binary_vars=qubo.n_qubits,
            variable_meaning=tuple(f"x{b}" for b in range(qubo.n_qubits)), qaoa_reps=reps,
            gammas=gammas, betas=betas, shots=shots, transpiler_seed=seed,
            backend_requested=backend, max_jobs=len(points), max_quantum_seconds=max_qs,
            n_theta=n_theta, n_q=n_q, optimization_level=opt, decision_ts=decision_ts)
        bundles.append(prepare_hardware_bundle(req, qubo, fake_backend, decision=decision,
                                               workload=workload))
    plan = build_full_qaoa_plan(bundles, strategy=strategy, total_shots_cap=total_cap,
                                max_quantum_seconds=max_qs, max_jobs=len(points),
                                max_angle_points=len(points))
    return bundles, plan


def register_qpu_routes(
    app: Any,
    *,
    loader: Any,
    runs_dir: str,
    confirm_store: qpu_confirm.DurableConfirmationStore | None = None,
    env: Callable[[str], str | None] = os.environ.get,
    decision_ts: float = 0.0,
    max_body_bytes: int = _MAX_BODY_BYTES,
    rate_limit: tuple[int, float] = _DEFAULT_RATE,
    monotonic: Callable[[], float] = time.monotonic,
    gateway_factory: Callable[[str], Any] | None = None,
    simulate_gateway_factory: Callable[[str], Any] | None = None,
) -> qpu_confirm.DurableConfirmationStore:
    """Attach guarded QPU routes to ``app``; returns the durable confirmation store."""
    from fastapi import Body, HTTPException, Request

    # durable + atomic: the challenge is persisted and the consume survives a restart
    store = confirm_store or qpu_confirm.DurableConfirmationStore(root=runs_dir)
    import threading as _threading
    if not (isinstance(rate_limit, tuple) and len(rate_limit) == 2
            and isinstance(rate_limit[0], int) and rate_limit[0] > 0 and rate_limit[1] > 0):
        raise ValueError("rate_limit must be (positive int, positive window seconds)")
    _rate_hits: dict[tuple[str, str], list[float]] = {}
    _rate_lock = _threading.Lock()
    _RATE_MAX_KEYS = 4096
    # real streaming body-size enforcement (not just a trusting Content-Length check)
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body_bytes)

    def _host_of(header: str | None) -> str | None:
        if not header:
            return None
        h = header.strip().lower()
        if h.startswith("http://"):
            h = h[len("http://"):]
        elif h.startswith("https://"):
            h = h[len("https://"):]
        return h.split("/")[0].rsplit(":", 1)[0] if not h.startswith("[") else h.split("]")[0] + "]"

    def _require_local(request: Request) -> None:
        host = request.client.host if request.client else None
        if host not in _LOOPBACK_HOSTS:
            raise HTTPException(status_code=403, detail="QPU mutations require a loopback client")

    def _require_qpu_enabled() -> None:
        if not qpu_flags.qpu_enabled(env=env):
            raise HTTPException(
                status_code=403,
                detail="QPU workflow is disabled (set ACRPQ_QPU_ENABLED=true to prepare/confirm)")

    def _guard(request: Request, *, require_json: bool = True) -> None:
        """HTTP security for a QPU mutation: loopback + Host/Origin + content-type +
        body-size + local rate limit. Trusts NO forwarded headers."""
        _require_local(request)
        headers = request.headers
        # Host must be a known loopback host (no reverse-proxy trust)
        if _host_of(headers.get("host")) not in _ALLOWED_HOSTS:
            raise HTTPException(status_code=403, detail="Host header not allowed")
        # Origin, if present, must be same-origin/loopback (CSRF defense)
        origin = headers.get("origin")
        if origin is not None and _host_of(origin) not in _ALLOWED_HOSTS:
            raise HTTPException(status_code=403, detail="cross-origin request refused")
        if require_json:
            ctype = (headers.get("content-type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                raise HTTPException(status_code=415, detail="content-type must be application/json")
        clen = headers.get("content-length")
        if clen is not None:
            try:
                if int(clen) > max_body_bytes:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid content-length") from None
        # fixed-window local rate limit per (client, path) — thread-safe + bounded
        limit, window = rate_limit
        key = (request.client.host if request.client else "?", request.url.path)
        now = monotonic()
        with _rate_lock:
            hits = [t for t in _rate_hits.get(key, ()) if now - t < window]
            if len(hits) >= limit:
                raise HTTPException(status_code=429, detail="rate limit exceeded")
            hits.append(now)
            _rate_hits[key] = hits
            if len(_rate_hits) > _RATE_MAX_KEYS:   # evict stale keys to bound memory
                for k in [k for k, ts in _rate_hits.items()
                          if not ts or now - ts[-1] >= window][: len(_rate_hits) - _RATE_MAX_KEYS]:
                    _rate_hits.pop(k, None)

    def _http_valueerror(exc: Exception) -> HTTPException:
        return HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}")

    def _load(run_id: str) -> dict[str, Any]:
        try:
            return qpu_runs.load(run_id, root=runs_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="unknown run") from exc
        except ValueError as exc:  # malformed run_id (e.g. path traversal attempt)
            raise HTTPException(status_code=404, detail="unknown or invalid run id") from exc
        except qpu_runs.CorruptRunError as exc:
            raise HTTPException(status_code=409, detail="run integrity check failed") from exc

    @app.get("/api/qpu/flags")
    def qpu_flags_route() -> dict[str, Any]:
        from .qpu_ibm_factory import runtime_factory_enabled

        return {
            **qpu_flags.flags_snapshot(env=env),
            "runtime_factory_enabled": runtime_factory_enabled(env=env),
        }

    @app.get("/api/qpu/runs")
    def qpu_run_list() -> dict[str, Any]:
        """Integrity-verified summaries for restart-safe UI resumption."""
        runs = qpu_runs.list_runs(root=runs_dir)
        return {
            "runs": [
                {
                    "run_id": r["local_run_id"], "state": r["state"],
                    "terminal": r.get("terminal", False),
                    "backend": r.get("params", {}).get("backend_requested"),
                    "instance": r.get("params", {}).get("instance"),
                    "created_utc": r.get("created_utc"),
                    "has_remote_job": bool(r.get("ibm_job_ids")),
                }
                for r in runs
            ]
        }

    @app.get("/api/qpu/backends/live/{backend_name}")
    def qpu_live_backend(backend_name: str) -> dict[str, Any]:
        """Read a real backend snapshot; never submits and never accepts a token."""
        if not qpu_flags.qpu_enabled(env=env):
            raise HTTPException(status_code=403, detail="QPU workflow is disabled")
        try:
            backend = _resolve_real_backend(backend_name, env=env)
            snapshot = _real_backend_snapshot(backend)
        except Exception as exc:  # noqa: BLE001 - map SDK/auth/backend errors without secrets
            from .ibm_runner import classify_remote_error

            category = classify_remote_error(exc)
            raise HTTPException(
                status_code=503,
                detail=f"IBM backend discovery failed ({category}); check the saved account",
            ) from exc
        return {"backend": asdict(snapshot), "read_only": True, "submitted": False}

    @app.post("/api/qpu/backends/assess")
    def qpu_assess(body: dict = Body(...)) -> dict[str, Any]:
        try:
            shots = _strict_int(body.get("shots", 1024), "shots", lo=1, hi=1_000_000)
            reps = _strict_int(body.get("qaoa_reps", 1), "qaoa_reps", lo=1, hi=100)
            snaps = body.get("snapshots")
            if not isinstance(snaps, list) or not snaps:
                raise ValueError("'snapshots' must be a non-empty list")
            if len(snaps) > _MAX_SNAPSHOTS:
                raise ValueError(f"too many snapshots (> {_MAX_SNAPSHOTS})")
            n = _strict_int(body.get("logical_qubits", 1), "logical_qubits", lo=1)
            density = _strict_float(body.get("interaction_density", 0.0),
                                    "interaction_density", lo=0.0, hi=1.0)
            workload = WorkloadRequirements(
                logical_qubits=n,
                est_two_qubit_gates=_strict_int(body.get("est_two_qubit_gates", 0),
                                                "est_two_qubit_gates", lo=0),
                logical_depth=2 * reps, interaction_density=density, shots=shots,
                decision_ts=decision_ts)
            decision = rank_backends(workload, [_snapshot_from_dict(s) for s in snaps])
        except (ValueError, ACRPError) as exc:
            raise _http_valueerror(exc) from exc
        return decision

    @app.post("/api/qpu/dry-run")
    def qpu_dry_run(request: Request, body: dict = Body(...)) -> dict[str, Any]:
        # dry-run is the SAFE demo path: no persistence, no gateway, no submission,
        # so it stays available even when the QPU workflow is otherwise disabled.
        _guard(request)
        try:
            bundle, plan, _decision = _prepare_bundle_and_plan(
                loader, body, decision_ts=decision_ts, env=env)
            report = dry_run(plan, bundle)
        except GatewayNotEnabled as exc:   # real backend requested, factory disabled: clean 503
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, ACRPError) as exc:
            raise _http_valueerror(exc) from exc
        return {
            "submittable": report.submittable, "plan_sha256": report.plan_sha256,
            "would_submit": list(report.would_submit), "budget": report.budget,
            "notes": list(report.notes), "warnings": list(report.warnings),
            "flags": qpu_flags.flags_snapshot(env=env),
        }

    @app.post("/api/qpu/prepare")
    def qpu_prepare(request: Request, body: dict = Body(...)) -> dict[str, Any]:
        _guard(request)
        _require_qpu_enabled()
        try:
            bundle, plan, decision = _prepare_bundle_and_plan(
                loader, body, decision_ts=decision_ts, env=env)
            p = bundle.prepared
            params = _expected_phase3_params(p)
            run_id = qpu_runs.prepare(params, root=runs_dir)
            # persist the whole chain durably so it survives a restart (Phase 6.1).
            # Store the FULL dataclasses (asdict) so each doc carries its own
            # self-hash (artefact_sha256 / plan_sha256), not just the manifest.
            from dataclasses import asdict as _asdict
            qpu_store.save_decision(run_id, decision, root=runs_dir)
            qpu_store.save_artefact(run_id, _asdict(p), root=runs_dir)
            qpu_store.save_plan(run_id, _asdict(plan), root=runs_dir)
            qpu_store.save_isa_circuit(
                run_id, qpu_store.dump_isa_circuit(bundle.isa_circuit), root=runs_dir)
            _nonce, challenge_public = store.issue(
                run_id=run_id, config_hash=p.config_hash, artefact_sha256=p.artefact_sha256,
                decision_sha256=p.decision_sha256, backend=p.backend, total_shots=plan.total_shots,
                max_jobs=plan.max_jobs, max_quantum_seconds=plan.max_quantum_seconds,
                plan_sha256=plan.plan_sha256,
                isa_fingerprint_sha256=p.isa_circuit["fingerprint_sha256"])
            report = dry_run(plan, bundle)
        except GatewayNotEnabled as exc:   # real backend requested, factory disabled: clean 503
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, ACRPError) as exc:
            raise _http_valueerror(exc) from exc
        return {
            "run_id": run_id, "state": qpu_runs.QpuState.PREPARED.value,
            "config_hash": p.config_hash, "artefact_sha256": p.artefact_sha256,
            "decision_sha256": p.decision_sha256, "plan_sha256": plan.plan_sha256,
            "backend": p.backend, "submittable": p.submittable,
            "budget": {"total_shots": plan.total_shots, "max_jobs": plan.max_jobs,
                       "max_quantum_seconds": plan.max_quantum_seconds},
            "isa": {"depth": p.transpilation["depth_after"],
                    "two_qubit_gates": p.transpilation["two_qubit_gates_after"]},
            "confirmation": challenge_public,
            "dry_run": {"would_submit": list(report.would_submit), "warnings": list(report.warnings)},
            "flags": qpu_flags.flags_snapshot(env=env),
            "warnings": list(p.warnings),
        }

    @app.post("/api/qpu/{run_id}/confirm")
    def qpu_confirm_route(run_id: str, request: Request, body: dict = Body(...)) -> dict[str, Any]:
        _guard(request)
        _require_qpu_enabled()
        run = _load(run_id)
        # prepared -> confirm; awaiting_confirmation -> an idempotent winner retry
        # (e.g. after a lost HTTP response or a crash between consume and transition)
        if run["state"] not in (qpu_runs.QpuState.PREPARED.value,
                                 qpu_runs.QpuState.AWAITING_CONFIRMATION.value):
            raise HTTPException(status_code=409,
                                detail=f"run must be 'prepared' to confirm; is {run['state']!r}")
        b = body if isinstance(body, dict) else {}
        try:
            # atomic, durable, single-winner consume (idempotent for the winner)
            store.consume(
                nonce=b.get("nonce"), run_id=run_id, config_hash=b.get("config_hash"),
                artefact_sha256=b.get("artefact_sha256"), decision_sha256=b.get("decision_sha256"),
                backend=b.get("backend"), total_shots=b.get("total_shots"),
                max_jobs=b.get("max_jobs"), max_quantum_seconds=b.get("max_quantum_seconds"),
                consent=b.get("consent"), current_config_hash=run.get("config_hash"))
        except qpu_confirm.ConfirmationError as exc:
            raise HTTPException(status_code=422, detail=f"confirmation refused: {exc}") from exc
        # explicit confirmation only reaches awaiting_confirmation; it NEVER submits.
        # The atomic consume marker is the commit point; completing the transition on
        # a retry is safe (recovery-safe: a crash here leaves a recoverable run).
        if run["state"] == qpu_runs.QpuState.PREPARED.value:
            qpu_runs.transition(run_id, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs_dir)
        return {
            "run_id": run_id, "state": qpu_runs.QpuState.AWAITING_CONFIRMATION.value,
            "submission_allowed": qpu_flags.submission_allowed(env=env),
            "note": ("confirmed locally; a real submission additionally requires the server "
                     "feature flags and a configured IBM account (not present)."),
        }

    @app.get("/api/qpu/{run_id}")
    def qpu_status(run_id: str) -> dict[str, Any]:
        return _load(run_id)

    @app.get("/api/qpu/{run_id}/events")
    def qpu_events(run_id: str, after_seq: int = -1, limit: int | None = None) -> dict[str, Any]:
        run = _load(run_id)   # 404/409 for unknown/corrupt (also verifies integrity)
        try:
            after = _strict_int(after_seq, "after_seq", lo=-1)
            lim = None if limit is None else _strict_int(limit, "limit", lo=1, hi=10_000)
            stream = qpu_runs.raw_events(run_id, root=runs_dir, after_seq=after, limit=lim)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from exc
        # EXHAUSTIVE: total == n_events; every append-only event kind is present
        return {"run_id": run_id, "state": run["state"], "n_events": run["n_events"],
                "total": stream["total"], "returned": stream["returned"],
                "after_seq": after, "events": stream["events"]}

    @app.post("/api/qpu/{run_id}/cancel")
    def qpu_cancel(run_id: str, request: Request) -> dict[str, Any]:
        _guard(request, require_json=False)
        run = _load(run_id)
        state = run["state"]
        # local cancel only: a run with no remote job can be cancelled outright.
        if run.get("ibm_job_ids"):
            raise HTTPException(status_code=409,
                                detail="run has a remote job; reconcile before cancelling")
        if state in (qpu_runs.QpuState.PREPARED.value,
                     qpu_runs.QpuState.AWAITING_CONFIRMATION.value):
            qpu_runs.transition(run_id, qpu_runs.QpuState.CANCELLED, root=runs_dir)
            return {"run_id": run_id, "state": qpu_runs.QpuState.CANCELLED.value}
        raise HTTPException(status_code=409, detail=f"cannot locally cancel from state {state!r}")

    @app.post("/api/qpu/full-qaoa/dry-run")
    def qpu_full_qaoa_dry_run(request: Request, body: dict = Body(...)) -> dict[str, Any]:
        # bounded full_qaoa_hardware, DRY-RUN only. Real full-QAOA submission stays
        # disabled by default via its own flag (ACRPQ_FULL_QAOA_SUBMISSION_ENABLED).
        _guard(request)
        try:
            _bundles, plan = _prepare_full_qaoa(loader, body, decision_ts=decision_ts)
        except (ValueError, ACRPError) as exc:
            raise _http_valueerror(exc) from exc
        return {
            "protocol": plan.protocol, "strategy": plan.strategy, "backend": plan.backend,
            "submittable": plan.submittable, "plan_sha256": plan.plan_sha256,
            "jobs": [{"job_index": j["job_index"], "local_tag": j["local_tag"],
                      "gammas": j["gammas"], "betas": j["betas"], "shots": j["shots"]}
                     for j in plan.jobs],
            "budget": {"max_jobs": plan.max_jobs, "total_shots": plan.total_shots,
                       "total_shots_cap": plan.total_shots_cap,
                       "max_quantum_seconds": plan.max_quantum_seconds,
                       "max_angle_points": plan.max_angle_points},
            "warnings": list(plan.warnings),
            "full_qaoa_submission_enabled": (
                (env("ACRPQ_FULL_QAOA_SUBMISSION_ENABLED") or "").strip().lower()
                in ("1", "true", "yes", "on")),
            "flags": qpu_flags.flags_snapshot(env=env),
        }

    def _gateway_for(backend: str) -> Any:
        """Build the submission/reconcile gateway ONLY after all guards pass.

        In tests an injected ``gateway_factory`` returns a FAKE gateway. Otherwise
        the default is a flag-gated :class:`RealIbmGatewayFactory` that constructs
        nothing until ``ACRPQ_IBM_RUNTIME_FACTORY_ENABLED`` is set — so a bare
        server fails closed (503) and never builds a ``QiskitRuntimeService`` or
        reads a token. Reached only AFTER every real-submission guard.
        """
        from .qpu_ibm_factory import GatewayNotEnabled, RealIbmGatewayFactory
        factory = gateway_factory or RealIbmGatewayFactory(env=env)
        try:
            return factory(backend)
        except GatewayNotEnabled as exc:
            raise HTTPException(status_code=503,
                                detail=f"real submission gateway not available: {exc}") from exc

    def _validate_real_submission_chain(run_id: str) -> dict[str, Any]:
        """Central guard: refuse ANY non-real submission BEFORE building a gateway.

        Requires: run awaiting_confirmation; a complete+coherent+**real_submittable**
        persisted chain (submittable artefact on a NON-fake backend from a
        NON-synthetic snapshot, self-hashes verified, ISA re-fingerprinted); and a
        cryptographically-verified consumed confirmation marker bound to the whole
        chain. A fake/synthetic/dry-run artefact, or a `{}`/forged marker, is
        refused here — so ``run_sampler`` is never reached for them.
        """
        run = _load(run_id)
        if run["state"] != qpu_runs.QpuState.AWAITING_CONFIRMATION.value:
            raise HTTPException(status_code=409,
                                detail=f"run must be awaiting_confirmation; is {run['state']!r}")
        report = qpu_store.verify_full_run(run_id, root=runs_dir)
        if not report.get("real_submittable"):
            raise HTTPException(
                status_code=409,
                detail=f"chain is not real-submittable (fake/synthetic/incoherent): "
                       f"{report.get('problems')}")
        artefact = qpu_store.load_artefact(run_id, root=runs_dir)
        plan = qpu_store.load_plan(run_id, root=runs_dir)
        try:
            store.verify_consumed_confirmation(
                run_id, expected_config_hash=run["config_hash"],
                expected_decision_sha256=artefact["decision_sha256"],
                expected_artefact_sha256=artefact["artefact_sha256"],
                expected_plan_sha256=plan["plan_sha256"],
                expected_isa_fingerprint=artefact["isa_circuit"]["fingerprint_sha256"],
                expected_backend=artefact["backend"], expected_shots=plan["total_shots"],
                expected_max_jobs=plan["max_jobs"],
                expected_max_quantum_seconds=plan["max_quantum_seconds"])
        except qpu_confirm.ConfirmationError as exc:
            raise HTTPException(status_code=409, detail=f"confirmation not verified: {exc}") from exc
        try:
            isa = qpu_store.load_isa_circuit(run_id, root=runs_dir, as_circuit=True)
        except qpu_store.StoreError as exc:
            raise HTTPException(status_code=409, detail=f"ISA circuit invalid: {exc}") from exc
        return {"run": run, "artefact": artefact, "isa": isa}

    @app.post("/api/qpu/{run_id}/submit")
    def qpu_submit(run_id: str, request: Request) -> dict[str, Any]:
        _guard(request, require_json=False)
        # (1) all three feature flags must align (fail-closed)
        if not qpu_flags.submission_allowed(env=env):
            raise HTTPException(status_code=403,
                                detail="submission disabled by server feature flags")
        # (2) refuse every non-real chain BEFORE any gateway is built
        ctx = _validate_real_submission_chain(run_id)
        # (3) build the gateway ONLY now (real seam / 501 in prod; a REAL gateway
        #     is refused for a dry-run-safe fake so a fake can never ride /submit)
        gateway = _gateway_for(ctx["artefact"]["backend"])
        if getattr(gateway, "dry_run_safe", False) is True:
            raise HTTPException(status_code=409,
                                detail="/submit refuses a dry_run_safe (fake) gateway")
        from .qpu_submit_gate import ApiHardwareAuthorizer

        def _reverify_chain() -> None:
            _validate_real_submission_chain(run_id)

        from .ibm_runner import _skewed_iso

        authorizer = ApiHardwareAuthorizer(
            local_run_id=run_id,   # bound to THIS run (MoE Expert E, IMPORTANT-2)
            tag=run_id,
            # skew the point-of-use dedup window like the main pre-submit search, so a
            # job created just before the run's own timestamp is still found (LOW-4)
            created_after_iso=_skewed_iso(ctx["run"]["created_utc"], 60.0),
            verify_chain=_reverify_chain,
            search=gateway,
        )
        # (4) idempotent submit exactly once — never retried; ambiguity -> submission_unknown
        try:
            verdict = ibm_runner.submit_run(
                run_id, gateway, isa_circuit=ctx["isa"], expected_config_hash=ctx["run"]["config_hash"],
                root=runs_dir, sleeper=lambda _s: None, jitter=lambda: 0.0,
                hardware_authorizer=authorizer)
        except (ibm_runner.SubmissionRefused, qpu_runs.IllegalTransitionError) as exc:
            current = qpu_runs.load(run_id, root=runs_dir)["state"]
            raise HTTPException(
                status_code=409,
                detail=f"submission already claimed or refused; state={current}: {exc}",
            ) from exc
        state = qpu_runs.load(run_id, root=runs_dir)["state"]
        return {"run_id": run_id, "verdict": verdict.get("verdict"), "state": state,
                "job_ids": verdict.get("job_ids") or ([verdict["job_id"]] if verdict.get("job_id") else [])}

    def _fetch_result_if_completed(run_id: str, gateway: Any) -> None:
        """Item 5 policy: once a run is 'completed', fetch the job's counts ONCE and
        persist them as result_raw (idempotent; never fabricates counts). A read
        failure leaves the run 'completed but result not retrieved' (honest)."""
        if not hasattr(gateway, "get_result"):
            return
        run = qpu_runs.load(run_id, root=runs_dir)
        if run["state"] != qpu_runs.QpuState.COMPLETED.value:
            return
        if qpu_store.has(runs_dir, run_id, "result_raw"):
            return                                   # already retrieved (idempotent)
        job_ids = run.get("ibm_job_ids") or []
        if not job_ids:
            return
        try:
            raw = gateway.get_result(gateway.get_job(job_ids[0]))
            qpu_store.save_result_raw(
                run_id, {"counts": raw["counts"], "bit_order": raw.get("bit_order", "qiskit"),
                         "kind": raw.get("kind", "counts")}, root=runs_dir)
        except Exception:  # noqa: BLE001 - never fabricate; leave result unretrieved
            return

    def _simulate_gateway(backend: str) -> Any:
        if simulate_gateway_factory is None:
            raise HTTPException(status_code=404, detail="simulate seam not enabled on this server")
        gw = simulate_gateway_factory(backend)
        if getattr(gw, "dry_run_safe", False) is not True:
            raise HTTPException(status_code=409,
                                detail="simulate requires a dry_run_safe (fake) gateway")
        if isinstance(gw, ibm_runner.RuntimeGateway):
            raise HTTPException(status_code=409, detail="simulate refuses a real RuntimeGateway")
        return gw

    @app.post("/api/qpu/{run_id}/simulate/submit")
    def qpu_simulate_submit(run_id: str, request: Request) -> dict[str, Any]:
        # TEST/DEV seam: exercise the submit MECHANICS with a fake gateway. Verifies
        # the durable confirmation marker (so the confirm flow is real) but permits a
        # fake artefact. NEVER contacts IBM.
        _guard(request, require_json=False)
        run = _load(run_id)
        if run["state"] != qpu_runs.QpuState.AWAITING_CONFIRMATION.value:
            raise HTTPException(status_code=409,
                                detail=f"run must be awaiting_confirmation; is {run['state']!r}")
        artefact = qpu_store.load_artefact(run_id, root=runs_dir)
        plan = qpu_store.load_plan(run_id, root=runs_dir)
        try:
            store.verify_consumed_confirmation(
                run_id, expected_config_hash=run["config_hash"],
                expected_decision_sha256=artefact["decision_sha256"],
                expected_artefact_sha256=artefact["artefact_sha256"],
                expected_plan_sha256=plan["plan_sha256"],
                expected_isa_fingerprint=artefact["isa_circuit"]["fingerprint_sha256"],
                expected_backend=artefact["backend"], expected_shots=plan["total_shots"],
                expected_max_jobs=plan["max_jobs"],
                expected_max_quantum_seconds=plan["max_quantum_seconds"])
        except qpu_confirm.ConfirmationError as exc:
            raise HTTPException(status_code=409, detail=f"confirmation not verified: {exc}") from exc
        isa = qpu_store.load_isa_circuit(run_id, root=runs_dir, as_circuit=True)
        gateway = _simulate_gateway(artefact["backend"])
        try:
            verdict = ibm_runner.submit_run(
                run_id, gateway, isa_circuit=isa, expected_config_hash=run["config_hash"],
                root=runs_dir, sleeper=lambda _s: None, jitter=lambda: 0.0)
        except (ibm_runner.SubmissionRefused, qpu_runs.IllegalTransitionError) as exc:
            current = qpu_runs.load(run_id, root=runs_dir)["state"]
            raise HTTPException(
                status_code=409,
                detail=f"simulation submission already claimed or refused; state={current}: {exc}",
            ) from exc
        _fetch_result_if_completed(run_id, gateway)
        return {"run_id": run_id, "simulated": True, "verdict": verdict.get("verdict"),
                "state": qpu_runs.load(run_id, root=runs_dir)["state"],
                "job_ids": verdict.get("job_ids") or ([verdict["job_id"]] if verdict.get("job_id") else [])}

    @app.post("/api/qpu/{run_id}/simulate/reconcile")
    def qpu_simulate_reconcile(run_id: str, request: Request) -> dict[str, Any]:
        _guard(request, require_json=False)
        run = _load(run_id)
        gateway = _simulate_gateway(run["params"]["backend_requested"])
        verdict = ibm_runner.reconcile_run(run_id, gateway, root=runs_dir,
                                           sleeper=lambda _s: None, jitter=lambda: 0.0)
        _fetch_result_if_completed(run_id, gateway)   # item 5: retrieve counts once
        return {"run_id": run_id, "simulated": True, "verdict": verdict,
                "state": qpu_runs.load(run_id, root=runs_dir)["state"]}

    @app.post("/api/qpu/{run_id}/reconcile")
    def qpu_reconcile(run_id: str, request: Request) -> dict[str, Any]:
        # reconcile stays available even when new submissions are disabled, so an
        # already-existing remote job can always be secured. Read-only remotely.
        _guard(request, require_json=False)
        run = _load(run_id)
        gateway = _gateway_for(run["params"]["backend_requested"])
        verdict = ibm_runner.reconcile_run(run_id, gateway, root=runs_dir,
                                           sleeper=lambda _s: None, jitter=lambda: 0.0)
        # A remote DONE transition and result retrieval are distinct operations.
        # Persist the counts immediately after reconciliation, exactly as the
        # fake lifecycle does, so the UI does not remain "completed/no result".
        _fetch_result_if_completed(run_id, gateway)
        state = qpu_runs.load(run_id, root=runs_dir)["state"]
        return {"run_id": run_id, "verdict": verdict, "state": state}

    @app.get("/api/qpu/{run_id}/result")
    def qpu_result(run_id: str) -> dict[str, Any]:
        run = _load(run_id)
        if not qpu_store.has(runs_dir, run_id, "result_raw"):
            return {"run_id": run_id, "state": run["state"], "available": False,
                    "note": "no raw result persisted yet (no counts fetched / submission seam)"}
        raw = qpu_store.load_result_raw(run_id, root=runs_dir)
        # decode the persisted raw result through the Phase-5 decoder (read-only)
        from ..quantum.qubo import build_qubo
        params = run["params"]
        inst = loader.load(params["instance"])
        qubo = build_qubo(
            inst, n_theta=params["n_theta"], n_q=params.get("n_q", 1),
            objective_id=params.get("objective_id", "quadratic_control_cost_v1"),
        )
        from .hardware_validation import decode_samples
        # Certified reference: the persisted raw doc rarely carries one, which left every
        # recorded optimality/absolute gap null (Agent D finding MEDIUM-3). When the
        # instance is enumeration-bounded (same cap as the exhaustive enumerator), compute
        # the exhaustive optimum here so the gaps are real; otherwise stay honestly null.
        reference_objective = raw.get("reference_objective")
        reference_source = "persisted" if reference_objective is not None else None
        if reference_objective is None:
            k_opts = qubo.n_qubits // max(1, inst.n)
            if k_opts ** inst.n <= 2_000_000:   # discrete_enum's own exhaustive cap
                from .qpu_batch import optimum_onehot_bits
                _ref = optimum_onehot_bits(
                    inst, qubo, qubo.objective_id, params["n_theta"], params.get("n_q", 1))
                if _ref.get("status") == "optimal" and _ref.get("feasible"):
                    reference_objective = _ref["optimum"]
                    reference_source = "exhaustive_certified"
        summary = decode_samples(
            qubo, raw["counts"], bit_order=raw.get("bit_order", "qiskit"),
            kind=raw.get("kind", "counts"),
            reference_objective=reference_objective)
        from dataclasses import asdict as _asdict
        return {
            "run_id": run_id, "state": run["state"], "available": True,
            "provenance": {"backend": params["backend_requested"], "instance": params["instance"],
                           "config_hash": run["config_hash"],
                           "objective_id": qubo.objective_id,
                           "reference_source": reference_source,
                           # Evidence that a REAL provider job exists. Without it a run is a
                           # local artefact, never "verified hardware" (MoE Expert D, CRITICAL-1).
                           "ibm_job_ids": list(run.get("ibm_job_ids") or []),
                           "has_remote_job": bool(run.get("ibm_job_ids"))},
            "summary": {
                "n_shots": summary.n_shots, "n_unique_total": summary.n_unique_total,
                "feasibility_rate": summary.feasibility_rate, "mean_energy": summary.mean_energy,
                "energy_dispersion": summary.energy_dispersion,
                "optimality_gap": summary.optimality_gap, "absolute_gap": summary.absolute_gap,
                "best_raw": _asdict(summary.best_raw) if summary.best_raw else None,
                "best_feasible": _asdict(summary.best_feasible) if summary.best_feasible else None,
                "modal": _asdict(summary.modal) if summary.modal else None,
                "warnings": list(summary.warnings),
                "top_explanations": list(summary.top_explanations),
            },
        }

    @app.post("/api/qpu/{run_id}/export")
    def qpu_export_route(run_id: str, request: Request) -> dict[str, Any]:
        _guard(request, require_json=False)
        _load(run_id)   # 404/409 for unknown/corrupt
        from . import qpu_export
        export = qpu_export.build_export_auto(run_id, root=runs_dir,
                                              provenance={"exported_by": "acrpq-qpu-api"})
        # also write it atomically next to the run (best-effort; ignore if it exists)
        try:
            dest = str(qpu_store._artefacts_dir(runs_dir, run_id) / "export.json")
            qpu_export.write_export(export, dest)
        except FileExistsError:
            pass
        return export

    return store
