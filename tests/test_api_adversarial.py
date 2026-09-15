"""Adversarial API / concurrency / persistence invariant tests (offline, no network).

Everything here runs against FastAPI's in-process TestClient or the persistence
modules directly — no sockets beyond the ASGI test transport, no IBM, no D-Wave.

Documented findings (xfail strict=True marks a REAL bug at HEAD; the test starts
passing — and the xfail fails loudly — once the bug is fixed):

* FINDING-1 (api.py:1483, 1527): ``applicability_block`` is called OUTSIDE the
  try/except in ``_do_solve`` / ``_do_scientific``, so an unknown instance with a
  preflight-gated solver (qaoa / qubo-exact / qubo-annealing is fine, reference
  is fine) escapes as an unhandled InstanceError -> HTTP 500.
* FINDING-2 (io/loader.py:242, surfaced by api.py error mapping at
  /api/instance/{name} 404 and the ``f"{type}: {exc}"`` 400 details): error
  responses leak the absolute server filesystem path (the loader's data_root).
* FINDING-3 (api.py, /api/solve + /api/scientific + /api/jobs): non-QPU
  state-changing POSTs validate neither Host nor Origin, so a DNS-rebinding page
  can reach them (incl. /api/scientific save=true, which WRITES to disk). The
  QPU routes do check both. (Covered by an xfail on the Host check.)
* FINDING-4 (qpu_api.py:591, 626 vs qpu_ibm_factory.py:26): ``GatewayNotEnabled``
  is a RuntimeError, but /api/qpu/prepare and /api/qpu/dry-run catch only
  ``(ValueError, ACRPError)`` — requesting an ``ibm_*`` backend while the runtime
  factory is disabled crashes with HTTP 500 instead of a clean 4xx/503.

These tests deliberately assert *invariants* (status classes, fail-closed
behaviour, integrity properties), not exact route lists, so additive UI endpoint
changes cannot break them.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import threading
import time
import warnings
from functools import partial
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from acrpq.dashboard.api import create_app  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = str(Path(__file__).resolve().parents[1])

_SNAP = {
    "name": "fake_lagos", "region": "us-east", "family": "Falcon", "physical_qubits": 7,
    "programmable_qubits": 7, "operational": True, "maintenance": False, "pending_jobs": 3,
    "two_qubit_error": 0.006, "readout_error": 0.012, "clops": 2500.0,
    "avg_coupling_degree": 2.0, "data_ts": 0.0,
}
_QPU_BODY = {
    "instance": "CP_3", "n_theta": 2, "shots": 512, "transpiler_seed": 1,
    "backend_requested": "fake_lagos", "gammas": [0.4], "betas": [0.3], "max_jobs": 1,
    "max_quantum_seconds": 20.0, "optimization_level": 0, "snapshots": [_SNAP],
}


@pytest.fixture()
def client():
    rd = tempfile.mkdtemp()
    tc = TestClient(create_app(runs_dir=rd), raise_server_exceptions=False)
    tc.runs_dir = rd
    return tc


@pytest.fixture()
def qpu_client(monkeypatch):
    monkeypatch.setenv("ACRPQ_QPU_ENABLED", "true")
    rd = tempfile.mkdtemp()
    tc = TestClient(create_app(runs_dir=rd), raise_server_exceptions=False)
    tc.runs_dir = rd
    return tc


def _prepare_and_confirm(tc):
    pr = tc.post("/api/qpu/prepare", json=_QPU_BODY)
    assert pr.status_code == 200, pr.text
    p = pr.json()
    ch = p["confirmation"]
    confirm = {
        "nonce": ch["nonce"], "config_hash": p["config_hash"],
        "artefact_sha256": p["artefact_sha256"], "decision_sha256": p["decision_sha256"],
        "backend": p["backend"], "total_shots": p["budget"]["total_shots"],
        "max_jobs": p["budget"]["max_jobs"],
        "max_quantum_seconds": p["budget"]["max_quantum_seconds"],
        "consent": ch["consent_phrase"],
    }
    return p, confirm


# --------------------------------------------------------------------------- #
# 1. Input validation / error handling on the dashboard API
# --------------------------------------------------------------------------- #
def test_solve_rejects_malformed_requests(client):
    cases = [
        ({"solver": "reference"}, "missing instance"),
        ({"instance": "CP_3", "solver": "not-a-solver"}, "unknown solver"),
        ({"instance": "CP_3", "evil_extra": 1}, "unknown field"),
        ({"instance": "CP_3", "n_theta": True}, "bool is not an int"),
        ({"instance": "CP_3", "n_theta": 26}, "grid bound"),
        ({"instance": "CP_3", "n_theta": 13, "n_q": 2}, "grid product cap"),
        ({"instance": "CP_3", "w": 1.5}, "w out of range"),
        ({"instance": "CP_3", "w": "0.5"}, "w must be numeric"),
        ({"instance": "CP_3", "subset": [1, 1]}, "duplicate subset ids"),
        ({"instance": "CP_3", "subset": [True]}, "bool subset id"),
        ({"instance": "CP_3", "seed": 2**64}, "seed out of range"),
        ({"instance": "CP_3", "profile": "citable-please"}, "unknown profile"),
        ({"instance": "CP_3", "mode": "ibm_free_lunch"}, "unknown mode"),
    ]
    for body, why in cases:
        r = client.post("/api/solve", json=body)
        assert r.status_code == 422, f"{why}: expected 422, got {r.status_code} {r.text}"


def test_scientific_rejects_bad_seed_lists(client):
    base = {"instance": "CP_3", "solver": "reference", "n_theta": 3}
    for seeds, why in [
        (None, "missing"), ([], "empty"), ([1, 1], "duplicates"),
        ([True], "bool"), (["1"], "string"), (list(range(13)), "over the 12-seed cap"),
    ]:
        body = dict(base)
        if seeds is not None:
            body["seeds"] = seeds
        r = client.post("/api/scientific", json=body)
        assert r.status_code == 422, f"seeds {why}: got {r.status_code} {r.text}"
    # 'save' must be a real JSON boolean
    r = client.post("/api/scientific", json={**base, "seeds": [1], "save": "yes"})
    assert r.status_code == 422


def test_unknown_instance_is_a_clean_client_error_for_reference(client):
    r = client.post("/api/solve", json={"instance": "NOPE_XX", "solver": "reference"})
    assert 400 <= r.status_code < 500
    r = client.get("/api/instance/NOPE_XX")
    assert r.status_code == 404


def test_unknown_instance_never_500_for_preflight_gated_solvers(client):
    for body in (
        {"instance": "NOPE_XX", "solver": "qaoa"},
        {"instance": "NOPE_XX", "solver": "qubo-exact"},
    ):
        r = client.post("/api/solve", json=body)
        assert r.status_code < 500, f"{body['solver']}: got {r.status_code}"
    r = client.post("/api/scientific",
                    json={"instance": "NOPE_XX", "solver": "qubo-exact", "seeds": [1]})
    assert r.status_code < 500


def test_error_details_do_not_leak_server_filesystem_paths(client):
    responses = [
        client.get("/api/instance/NOPE_XX"),
        client.post("/api/solve", json={"instance": "NOPE_XX", "solver": "reference"}),
        client.get("/api/qubo/NOPE_XX"),
    ]
    for r in responses:
        assert 400 <= r.status_code < 500
        assert REPO_ROOT not in r.text, f"absolute server path leaked: {r.text[:200]}"


def test_scientific_save_refuses_foreign_host_header(client):
    r = client.post(
        "/api/scientific",
        json={"instance": "CP_3", "solver": "reference", "n_theta": 3,
              "seeds": [1], "save": True},
        headers={"host": "evil.example"},
    )
    assert r.status_code == 403, f"expected foreign Host to be refused, got {r.status_code}"


def test_run_read_routes_reject_traversal_and_bad_kinds(client):
    # kind outside the whitelist and malformed run ids must be clean 4xx, never 5xx
    assert client.get("/api/runs/qpu-or-whatever").status_code == 404
    for rid in ("..", "%2e%2e%2fsecrets", "a" * 64, "20200101T000000Z-xxxxxxxx-yyyyyyyy"):
        r = client.get(f"/api/runs/scientific/{rid}")
        assert 400 <= r.status_code < 500, f"{rid}: {r.status_code}"
        v = client.get(f"/api/runs/scientific/{rid}/verify")
        assert 400 <= v.status_code < 500, f"{rid}/verify: {v.status_code}"
    # static mount must not serve files outside static/
    assert client.get("/../pyproject.toml").status_code == 404


# --------------------------------------------------------------------------- #
# 2. Stale state / concurrency on the solve path
# --------------------------------------------------------------------------- #
def test_concurrent_solves_return_results_matching_their_own_config(client):
    """A result computed for config X must never be returned labelled as config Y
    (exercises the shared model/QUBO caches under concurrent requests)."""
    results: dict[int, dict] = {}
    errors: list[str] = []

    def solve(n_theta: int) -> None:
        r = client.post("/api/solve", json={
            "instance": "CP_3", "solver": "reference", "n_theta": n_theta})
        if r.status_code != 200:
            errors.append(f"{n_theta}: {r.status_code}")
            return
        results[n_theta] = r.json()

    threads = [threading.Thread(target=solve, args=(k,)) for k in (3, 5, 7, 3, 5, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for k in (3, 5, 7):
        assert results[k]["solution"]["grid_k"] == k
        assert results[k]["profile"]["params"]["n_theta"] == k
        assert results[k]["name"] == "CP_3"


def test_scientific_aggregate_is_internally_consistent(client):
    out = client.post("/api/scientific", json={
        "instance": "CP_3", "solver": "qubo-annealing", "n_theta": 3, "seeds": [1, 2]})
    assert out.status_code == 200, out.text
    body = out.json()
    agg = body["aggregate"]
    assert agg["n_runs"] == len(body["runs"]) == 2
    assert agg["seeds"] == [1, 2]
    # each per-seed run must carry the seed it was actually computed with
    assert [r["provenance"]["seed"] for r in body["runs"]] == [1, 2]
    assert all(r["profile"]["name"] == "scientific" for r in body["runs"])


def test_job_lifecycle_and_cancellation_semantics(client):
    # invalid work is accepted at submit (validation is deferred into the job) but
    # must surface as a FAILED job, never as a fabricated result
    r = client.post("/api/jobs", json={"instance": "NOPE_XX", "solver": "reference"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    for _ in range(100):
        snap = client.get(f"/api/jobs/{jid}").json()
        if snap["state"] in ("failed", "completed", "cancelled"):
            break
        time.sleep(0.05)
    assert snap["state"] == "failed"
    assert snap["has_result"] is False
    # asking for the result of a non-completed job is a structured 409
    assert client.get(f"/api/jobs/{jid}/result").status_code == 409
    # unknown job -> 404 on all three routes
    assert client.get("/api/jobs/deadbeefdead").status_code == 404
    assert client.get("/api/jobs/deadbeefdead/result").status_code == 404
    assert client.delete("/api/jobs/deadbeefdead").status_code == 404
    # a good job completes and its result matches the submitted config
    r = client.post("/api/jobs", json={"instance": "CP_3", "solver": "reference",
                                       "n_theta": 5})
    jid = r.json()["job_id"]
    for _ in range(200):
        snap = client.get(f"/api/jobs/{jid}").json()
        if snap["state"] in ("failed", "completed", "cancelled"):
            break
        time.sleep(0.05)
    assert snap["state"] == "completed", snap
    result = client.get(f"/api/jobs/{jid}/result").json()
    assert result["solution"]["grid_k"] == 5 and result["name"] == "CP_3"
    # cancelling a finished job is honestly reported, not silently rewound
    c = client.delete(f"/api/jobs/{jid}")
    assert c.status_code == 200 and "already completed" in c.json()["cancel_note"]


def test_job_backlog_is_bounded():
    """The job pool must reject rather than queue unboundedly. Uses the exact
    JobManager class /api/jobs is built on, with controllable blocking work, so
    the bound is asserted deterministically (the HTTP layer maps QueueFullError
    to 429 in api.create_app)."""
    from acrpq.dashboard.jobs import JobManager, QueueFullError

    release = threading.Event()
    mgr = JobManager(max_workers=1, max_queue=3)
    try:
        ids = [mgr.submit("solve", release.wait) for _ in range(3)]
        with pytest.raises(QueueFullError):
            mgr.submit("solve", release.wait)  # 4th active job -> bounded refusal
    finally:
        release.set()
    for jid in ids:
        for _ in range(200):
            if mgr.snapshot(jid)["state"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.01)
    # once the backlog drains, submission is accepted again (no stuck counter)
    jid = mgr.submit("solve", lambda: "ok")
    for _ in range(200):
        if mgr.snapshot(jid)["state"] == "completed":
            break
        time.sleep(0.01)
    assert mgr.result(jid) == "ok"


# --------------------------------------------------------------------------- #
# 3. Persistence: crash-safety, torn files, concurrent writers
# --------------------------------------------------------------------------- #
def test_saved_scientific_run_verify_detects_torn_file(client):
    out = client.post("/api/scientific", json={
        "instance": "CP_3", "solver": "reference", "n_theta": 3,
        "seeds": [1], "save": True})
    assert out.status_code == 200, out.text
    saved = out.json()["saved"]
    rid = saved["run_id"]
    assert client.get(f"/api/runs/scientific/{rid}/verify").json()["json_ok"] is True
    # torn write / tamper: truncate the committed artifact
    path = Path(saved["path"])
    path.write_text(path.read_text(encoding="utf-8")[:100], encoding="utf-8")
    v = client.get(f"/api/runs/scientific/{rid}/verify")
    assert v.status_code == 422 and "integrity check FAILED" in v.text
    # the read route must fail as a clean 4xx (never a 500), and the index survives
    assert 400 <= client.get(f"/api/runs/scientific/{rid}").status_code < 500
    assert client.get("/api/runs/scientific").status_code == 200


def _save_record_worker(root: str, record_json: str, _i: int = 0) -> str:
    from acrpq.dashboard.persist import save_run
    try:
        save_run(json.loads(record_json), root=root)
        return "ok"
    except FileExistsError:
        return "exists"


def _append_obs_worker(root: str, rid: str, i: int):
    from acrpq.dashboard import qpu_runs
    return qpu_runs.record_provider_observation(rid, {"i": i}, root=root)


def _cancel_worker(root: str, rid: str, _i: int) -> str:
    from acrpq.dashboard import qpu_runs
    try:
        qpu_runs.transition(rid, qpu_runs.QpuState.CANCELLED, root=root)
        return "won"
    except qpu_runs.IllegalTransitionError:
        return "refused"


def _fork_pool(n: int):
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - platform without fork
        pytest.skip("fork start method unavailable")
    return ctx.Pool(n)


def test_concurrent_same_run_id_save_has_exactly_one_winner(tmp_path):
    from acrpq.dashboard.persist import build_interactive_record, list_runs

    record = build_interactive_record(
        {"profile": {"name": "demo"}, "solution": {"feasible": True}},
        instance="CP_3", subset=None, config={"solver": "reference"})
    payload = json.dumps(record)
    with _fork_pool(4) as pool:
        outs = pool.map(partial(_save_record_worker, str(tmp_path), payload), range(6))
    assert sorted(outs) == ["exists"] * 5 + ["ok"]
    assert len(list_runs("interactive", root=str(tmp_path))) == 1


def test_concurrent_qpu_event_appends_never_lose_or_clobber_events(tmp_path):
    from acrpq.dashboard import qpu_runs

    rid = qpu_runs.prepare(
        {"mode": "aer_simulation", "instance": "CP_3", "n_theta": 3, "shots": 128},
        root=str(tmp_path))
    with _fork_pool(8) as pool:
        seqs = pool.map(partial(_append_obs_worker, str(tmp_path), rid), range(16))
    assert sorted(seqs) == list(range(1, 17))  # all 16 appended, all seqs distinct
    state = qpu_runs.load(rid, root=str(tmp_path))
    assert state["n_events"] == 17
    assert qpu_runs.verify(rid, root=str(tmp_path))["ok"] is True


def test_concurrent_terminal_transition_has_single_winner(tmp_path):
    from acrpq.dashboard import qpu_runs

    rid = qpu_runs.prepare(
        {"mode": "aer_simulation", "instance": "CP_3", "n_theta": 3, "shots": 128},
        root=str(tmp_path))
    with _fork_pool(4) as pool:
        outs = pool.map(partial(_cancel_worker, str(tmp_path), rid), range(4))
    assert sorted(outs) == ["refused", "refused", "refused", "won"]
    assert qpu_runs.load(rid, root=str(tmp_path))["state"] == "cancelled"
    assert qpu_runs.verify(rid, root=str(tmp_path))["ok"] is True


# --------------------------------------------------------------------------- #
# 4. QPU API: fail-closed flags, confirmation idempotence, HTTP guards
# --------------------------------------------------------------------------- #
def test_qpu_submit_fails_closed_before_anything_else(client):
    """With default flags, /submit is 403 EVEN for an unknown run id: the flag
    gate runs before any disk access, so nothing can be probed while disabled."""
    r = client.post("/api/qpu/20200101T000000000000Z-aaaaaaaa-bbbbbbbb/submit")
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]
    flags = client.get("/api/qpu/flags").json()
    assert flags["submission_allowed"] is False
    assert flags["qpu_enabled"] is False
    assert flags["dry_run_only"] is True


def test_qpu_prepare_disabled_by_default_but_dry_run_stays_available(client):
    assert client.post("/api/qpu/prepare", json=_QPU_BODY).status_code == 403
    r = client.post("/api/qpu/dry-run", json=_QPU_BODY)
    assert r.status_code == 200
    assert r.json()["submittable"] is False


def test_qpu_confirm_is_idempotent_for_the_winner_and_replay_proof(qpu_client):
    prepared, confirm = _prepare_and_confirm(qpu_client)
    rid = prepared["run_id"]
    r1 = qpu_client.post(f"/api/qpu/{rid}/confirm", json=confirm)
    assert r1.status_code == 200
    assert r1.json()["state"] == "awaiting_confirmation"
    assert r1.json()["submission_allowed"] is False  # confirm NEVER submits
    # double-click: the exact same payload is an idempotent success
    r2 = qpu_client.post(f"/api/qpu/{rid}/confirm", json=confirm)
    assert r2.status_code == 200
    assert r2.json()["state"] == "awaiting_confirmation"
    # replay with a different nonce or budget is refused
    assert qpu_client.post(f"/api/qpu/{rid}/confirm",
                           json={**confirm, "nonce": "f" * 32}).status_code == 422
    assert qpu_client.post(f"/api/qpu/{rid}/confirm",
                           json={**confirm, "total_shots": 999999}).status_code == 422
    # wrong consent phrase is refused even with the right nonce
    assert qpu_client.post(f"/api/qpu/{rid}/confirm",
                           json={**confirm, "consent": "yes"}).status_code == 422
    # state is still awaiting_confirmation (the failed replays changed nothing)
    assert qpu_client.get(f"/api/qpu/{rid}").json()["state"] == "awaiting_confirmation"


def test_qpu_confirmed_fake_chain_can_never_ride_submit(qpu_client, monkeypatch):
    """Even with ALL submission flags forced on, a fake-backend chain is refused
    at /submit (409) before any gateway is constructed — no IBM contact."""
    prepared, confirm = _prepare_and_confirm(qpu_client)
    rid = prepared["run_id"]
    assert qpu_client.post(f"/api/qpu/{rid}/confirm", json=confirm).status_code == 200
    monkeypatch.setenv("ACRPQ_IBM_SUBMISSION_ENABLED", "true")
    monkeypatch.setenv("ACRPQ_QPU_DRY_RUN", "false")
    r = qpu_client.post(f"/api/qpu/{rid}/submit")
    assert r.status_code == 409
    assert "not real-submittable" in r.json()["detail"]
    # the run did not advance towards submission
    assert qpu_client.get(f"/api/qpu/{rid}").json()["state"] == "awaiting_confirmation"


def test_qpu_reconcile_fails_closed_when_factory_disabled(qpu_client):
    prepared, confirm = _prepare_and_confirm(qpu_client)
    rid = prepared["run_id"]
    assert qpu_client.post(f"/api/qpu/{rid}/confirm", json=confirm).status_code == 200
    r = qpu_client.post(f"/api/qpu/{rid}/reconcile")
    assert r.status_code == 503
    assert "gateway not available" in r.json()["detail"]


def test_qpu_confirm_refused_after_local_cancel(qpu_client):
    prepared, confirm = _prepare_and_confirm(qpu_client)
    rid = prepared["run_id"]
    assert qpu_client.post(f"/api/qpu/{rid}/cancel").status_code == 200
    r = qpu_client.post(f"/api/qpu/{rid}/confirm", json=confirm)
    assert r.status_code == 409  # cancelled is terminal; confirm cannot resurrect it
    assert qpu_client.get(f"/api/qpu/{rid}").json()["state"] == "cancelled"


def test_qpu_mutations_enforce_host_origin_and_body_limits(qpu_client):
    # foreign Host header -> refused
    r = qpu_client.post("/api/qpu/dry-run", json=_QPU_BODY, headers={"host": "evil.example"})
    assert r.status_code == 403
    # cross-origin -> refused
    r = qpu_client.post("/api/qpu/dry-run", json=_QPU_BODY,
                        headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    # same-origin loopback Origin -> allowed
    r = qpu_client.post("/api/qpu/dry-run", json=_QPU_BODY,
                        headers={"origin": "http://127.0.0.1:8000"})
    assert r.status_code == 200
    # oversized body (real received bytes, not Content-Length) -> 413
    r = qpu_client.post("/api/qpu/dry-run", content=b"x" * 300_000,
                        headers={"content-type": "application/json"})
    assert r.status_code == 413
    # a lying small Content-Length cannot smuggle a big body past the cap
    r = qpu_client.post("/api/qpu/dry-run", content=b"x" * 300_000,
                        headers={"content-type": "application/json",
                                 "content-length": "10"})
    assert r.status_code in (400, 413)


def test_qpu_run_id_traversal_is_rejected(qpu_client):
    for rid in ("..", "manifest", "a" * 80, "20200101T000000000000Z-zzzzzzzz-!!"):
        r = qpu_client.get(f"/api/qpu/{rid}")
        assert r.status_code == 404, f"{rid}: {r.status_code}"


def test_qpu_events_pagination_is_exhaustive(qpu_client):
    prepared, _ = _prepare_and_confirm(qpu_client)
    rid = prepared["run_id"]
    ev = qpu_client.get(f"/api/qpu/{rid}/events").json()
    assert ev["total"] == ev["n_events"] == ev["returned"]
    page = qpu_client.get(f"/api/qpu/{rid}/events?after_seq=-1&limit=1").json()
    assert page["returned"] == 1 and page["total"] == ev["total"]
    assert qpu_client.get(f"/api/qpu/{rid}/events?after_seq=-2").status_code == 422


def test_qpu_rate_limit_is_enforced_per_client_and_path():
    from fastapi import FastAPI

    from acrpq.dashboard.qpu_api import register_qpu_routes
    from acrpq.io.loader import InstanceLoader

    app = FastAPI()
    register_qpu_routes(app, loader=InstanceLoader(), runs_dir=tempfile.mkdtemp(),
                        rate_limit=(3, 60.0))
    tc = TestClient(app, raise_server_exceptions=False)
    codes = [tc.post("/api/qpu/dry-run", json={}).status_code for _ in range(5)]
    assert codes[:3] == [422, 422, 422]      # guard passed; body invalid
    assert codes[3] == 429 and codes[4] == 429


def test_qpu_prepare_rejects_hostile_payloads(qpu_client):
    cases = [
        ({**_QPU_BODY, "extra_field": 1}, "unknown field"),
        ({**_QPU_BODY, "shots": True}, "bool shots"),
        ({**_QPU_BODY, "shots": "512"}, "string shots"),
        ({**_QPU_BODY, "max_quantum_seconds": float("inf")}, "inf budget"),
        ({**_QPU_BODY, "gammas": []}, "empty gammas"),
        ({**_QPU_BODY, "snapshots": []}, "empty snapshots"),
        ({**_QPU_BODY, "snapshots": [{**_SNAP, "synthetic": False, "rogue": 1}]},
         "unknown snapshot field"),
    ]
    for body, why in cases:
        try:
            payload = json.dumps(body)
        except ValueError:
            payload = json.dumps(body, allow_nan=True)  # inf case: raw JSON token
        r = qpu_client.post("/api/qpu/prepare", content=payload,
                            headers={"content-type": "application/json"})
        assert r.status_code == 422, f"{why}: got {r.status_code} {r.text[:120]}"
        # nothing may have been persisted by a refused prepare
    assert qpu_client.get("/api/qpu/runs").json()["runs"] == []


def test_qpu_prepare_with_real_backend_and_disabled_factory_is_not_a_500(qpu_client):
    body = {**_QPU_BODY, "backend_requested": "ibm_real_thing"}
    for route in ("/api/qpu/prepare", "/api/qpu/dry-run"):
        r = qpu_client.post(route, json=body)
        # FIXED (was FINDING-4): a clean, intentional refusal — 403 (flag) or 503
        # (factory disabled, matching _gateway_for) — never an unhandled 500.
        assert r.status_code in (403, 503), f"{route}: got {r.status_code}"
    assert qpu_client.get("/api/qpu/runs").json()["runs"] == []


def test_qpu_snapshot_from_client_is_always_marked_synthetic(qpu_client):
    """A client claiming synthetic=False must not make its snapshot 'real'."""
    body = {**_QPU_BODY, "snapshots": [{**_SNAP, "synthetic": False}]}
    p = qpu_client.post("/api/qpu/prepare", json=body)
    assert p.status_code == 200, p.text
    prepared = p.json()
    assert prepared["submittable"] is False  # fake backend chain stays non-submittable
