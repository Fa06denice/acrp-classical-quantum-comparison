"""Phase 5.4 — budget policy, atomic reservation, single-use confirmation (no IBM).

Includes REAL multi-process tests (spawned interpreters) for the commit-vs-release
transition and for a last-slot race in EACH capped dimension (global / per-backend /
per-instance), plus confirmation replay / concurrent consume / tamper handling.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from acrpq.dashboard import qpu_budget as B

CH = "sha256:" + "1" * 64
JH = "sha256:" + "2" * 64


def _policy(**over):
    base = dict(max_jobs=3, shots_per_job=1024, total_shot_budget=100000,
                qpu_seconds_per_job=10.0, max_total_qpu_seconds=600.0, max_wait_seconds=600.0,
                max_jobs_per_backend=3, max_jobs_per_instance=2, stop_after_consecutive_errors=2)
    base.update(over)
    return B.BudgetPolicy(**base)


def _reserve(led, rid, backend="ibm_x", instance="CP_3"):
    return led.reserve(reservation_id=rid, batch_config_hash=CH, job_config_hash=JH,
                       backend=backend, instance=instance)


# --- policy ---------------------------------------------------------------
def test_within_caps_requires_confirmation_never_silent_allow():
    d = B.evaluate(_policy(), requested_jobs=1, requested_shots_per_job=1024)
    assert d.action == "requires_explicit_confirmation" and not d.allowed


@pytest.mark.parametrize("kw,reason", [
    (dict(requested_jobs=0, requested_shots_per_job=1024), "requested_jobs<1"),
    (dict(requested_jobs=1, requested_shots_per_job=512), "shots_per_job_mismatch"),
    (dict(requested_jobs=99, requested_shots_per_job=1024), "exceeds_effective_max_slots"),
    (dict(requested_jobs=1, requested_shots_per_job=1024, instance_jobs_used=2),
     "exceeds_max_jobs_per_instance"),
    (dict(requested_jobs=1, requested_shots_per_job=1024, consecutive_errors=2),
     "stop_after_consecutive_errors"),
])
def test_blocked_reasons(kw, reason):
    d = B.evaluate(_policy(), **kw)
    assert d.action == "blocked" and reason in d.reasons


def test_effective_max_slots_binds():
    assert _policy(total_shot_budget=5000).effective_max_slots() == 3
    assert _policy(max_total_qpu_seconds=25.0).effective_max_slots() == 2


@pytest.mark.parametrize("bad", [dict(max_jobs=0), dict(shots_per_job=-1),
                                 dict(qpu_seconds_per_job=0.0),
                                 dict(max_total_qpu_seconds=float("inf"))])
def test_invalid_policy_refused(bad):
    with pytest.raises(B.BudgetError):
        _policy(**bad)


# --- reservation single-process -------------------------------------------
def test_reservation_binds_identity_and_exhausts(tmp_path):
    led = B.BudgetLedger(str(tmp_path), _policy(max_jobs=2, total_shot_budget=100000,
                                                max_total_qpu_seconds=600.0))
    r0 = _reserve(led, "a")
    assert r0.batch_config_hash == CH and r0.backend == "ibm_x" and r0.shots == 1024
    _reserve(led, "b")
    with pytest.raises(B.BudgetExhausted, match="global"):
        _reserve(led, "c")


def test_release_returns_budget_then_reservable(tmp_path):
    led = B.BudgetLedger(str(tmp_path), _policy(max_jobs=1, total_shot_budget=100000,
                                                max_total_qpu_seconds=600.0))
    r = _reserve(led, "a")
    led.release(r)
    assert led.slots_used("global") == 0
    _reserve(led, "b")                                    # slot returned -> reservable again


def test_commit_then_release_fails_closed_and_vice_versa(tmp_path):
    led = B.BudgetLedger(str(tmp_path), _policy())
    r = _reserve(led, "a")
    led.commit(r)
    with pytest.raises(B.BudgetError, match="cannot released"):
        led.release(r)
    led.commit(r)                                         # idempotent re-commit ok
    r2 = _reserve(led, "b")
    led.release(r2)
    with pytest.raises(B.BudgetError, match="cannot committed"):
        led.commit(r2)


def test_reservation_record_tamper_fails_closed(tmp_path):
    led = B.BudgetLedger(str(tmp_path), _policy())
    r = _reserve(led, "a")
    p = led._record_path("a")
    obj = json.loads(p.read_text())
    obj["backend"] = "ibm_evil"                           # break the content hash
    p.write_text(json.dumps(obj))
    with pytest.raises(B.BudgetError, match="content hash"):
        led.commit(r)


def test_duplicate_reservation_id_refused(tmp_path):
    led = B.BudgetLedger(str(tmp_path), _policy())
    _reserve(led, "a")
    with pytest.raises(B.BudgetError, match="already exists"):
        _reserve(led, "a")


# --- REAL multi-process races ---------------------------------------------
_RESERVE_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    from acrpq.dashboard import qpu_budget as B
    root, out, gate, backend, instance = sys.argv[1:6]
    pol = B.BudgetPolicy(max_jobs=int(os.environ["MAXJOBS"]), shots_per_job=1024,
        total_shot_budget=100000, qpu_seconds_per_job=10.0, max_total_qpu_seconds=600.0,
        max_wait_seconds=600.0, max_jobs_per_backend=int(os.environ["MAXBK"]),
        max_jobs_per_instance=int(os.environ["MAXINST"]), stop_after_consecutive_errors=2)
    led = B.BudgetLedger(root, pol)
    while not os.path.exists(gate):
        time.sleep(0.001)
    try:
        r = led.reserve(reservation_id="pid-%d" % os.getpid(),
            batch_config_hash="sha256:"+"1"*64, job_config_hash="sha256:"+"2"*64,
            backend=backend, instance=instance)
        res = {"ok": True}
    except B.BudgetExhausted:
        res = {"ok": False}
    json.dump(res, open(out, "w"))
    """
)


def _race_reserve(tmp_path, n, *, maxjobs, maxbk, maxinst, backend="ibm_x", instance="CP_3"):
    import os
    worker = tmp_path / "w.py"
    worker.write_text(_RESERVE_WORKER)
    root = str(tmp_path / "led")
    gate = tmp_path / "GO"
    outs = [str(tmp_path / f"o{i}.json") for i in range(n)]
    env = {**os.environ, "MAXJOBS": str(maxjobs), "MAXBK": str(maxbk), "MAXINST": str(maxinst)}
    procs = [subprocess.Popen([sys.executable, str(worker), root, outs[i], str(gate),
                               backend, instance], env=env) for i in range(n)]
    gate.write_text("go")
    for p in procs:
        assert p.wait(timeout=60) == 0
    return [json.loads(Path(o).read_text()) for o in outs]


def test_global_last_slot_race(tmp_path):
    res = _race_reserve(tmp_path, 4, maxjobs=1, maxbk=9, maxinst=9)
    assert sum(1 for r in res if r["ok"]) == 1            # exactly one wins the single global slot


def test_per_backend_last_slot_race(tmp_path):
    # global allows 5 but the backend cap is 1: same backend -> only one wins
    res = _race_reserve(tmp_path, 4, maxjobs=5, maxbk=1, maxinst=9, backend="ibm_shared")
    assert sum(1 for r in res if r["ok"]) == 1


def test_per_instance_last_slot_race(tmp_path):
    res = _race_reserve(tmp_path, 4, maxjobs=5, maxbk=9, maxinst=1, instance="CP_shared")
    assert sum(1 for r in res if r["ok"]) == 1


_TRANSITION_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    from acrpq.dashboard import qpu_budget as B
    root, out, gate, action = sys.argv[1:5]
    pol = B.BudgetPolicy(max_jobs=3, shots_per_job=1024, total_shot_budget=100000,
        qpu_seconds_per_job=10.0, max_total_qpu_seconds=600.0, max_wait_seconds=600.0,
        max_jobs_per_backend=3, max_jobs_per_instance=3, stop_after_consecutive_errors=2)
    led = B.BudgetLedger(root, pol)
    r = B.Reservation("shared", "sha256:"+"1"*64, "sha256:"+"2"*64, "ibm_x", "CP_3", 1024, 10.0)
    while not os.path.exists(gate):
        time.sleep(0.001)
    try:
        (led.commit if action == "commit" else led.release)(r)
        res = {"ok": True, "action": action}
    except B.BudgetError as e:
        res = {"ok": False, "action": action, "err": str(e)[:40]}
    json.dump(res, open(out, "w"))
    """
)


def test_commit_vs_release_race_is_mutually_exclusive(tmp_path):
    """A commit and a release of the SAME reservation racing in two processes: exactly one
    wins the atomic transition; the other fails closed (no read/check/unlink race)."""
    led = B.BudgetLedger(str(tmp_path / "led"), _policy())
    _reserve(led, "shared")
    worker = tmp_path / "t.py"
    worker.write_text(_TRANSITION_WORKER)
    root = str(tmp_path / "led")
    gate = tmp_path / "GO"
    outs = [str(tmp_path / f"t{i}.json") for i in range(2)]
    procs = [subprocess.Popen([sys.executable, str(worker), root, outs[0], str(gate), "commit"]),
             subprocess.Popen([sys.executable, str(worker), root, outs[1], str(gate), "release"])]
    gate.write_text("go")
    for p in procs:
        assert p.wait(timeout=60) == 0
    res = [json.loads(Path(o).read_text()) for o in outs]
    assert sum(1 for r in res if r["ok"]) == 1            # exactly one transition wins
    assert sum(1 for r in res if not r["ok"]) == 1        # the other fails closed


# --- confirmation ---------------------------------------------------------
def _conf(**over):
    base = dict(batch_config_hash=CH, job_config_hash=JH, instance="CP_3",
                objective_id="maneuver_count_v1", backend="ibm_x", shots=1024, n_qubits=9,
                reserved_shots=1024, reserved_qpu_seconds=10.0, qubo_sha256="sha256:" + "3" * 64,
                isa_sha256="sha256:" + "4" * 64, nonce="nonce-abc", expires_epoch=1000.0)
    base.update(over)
    return B.Confirmation(**base)


@pytest.mark.parametrize("bad", [dict(shots=0), dict(n_qubits=-1), dict(backend=""),
                                 dict(reserved_qpu_seconds=float("inf")),
                                 dict(expires_epoch=float("nan")), dict(nonce="")])
def test_invalid_confirmation_refused(bad):
    with pytest.raises(B.BudgetError):
        _conf(**bad)


def test_confirmation_single_use_and_replay(tmp_path):
    store = B.ConfirmationStore(str(tmp_path))
    c = _conf()
    store.issue(c)
    store.consume(c, now_epoch=100.0)
    with pytest.raises(B.BudgetError, match="already consumed"):
        store.consume(c, now_epoch=100.0)


def test_confirmation_expiry_and_identity_change(tmp_path):
    store = B.ConfirmationStore(str(tmp_path))
    store.issue(_conf(nonce="n1", expires_epoch=50.0))
    with pytest.raises(B.BudgetError, match="expired"):
        store.consume(_conf(nonce="n1", expires_epoch=50.0), now_epoch=100.0)
    store.issue(_conf(nonce="n2"))
    for changed in (_conf(nonce="n2", backend="ibm_other"), _conf(nonce="n2", shots=2048)):
        with pytest.raises(B.BudgetError, match="does not match"):
            store.consume(changed, now_epoch=100.0)


def test_confirmation_pending_tamper_fails_closed(tmp_path):
    store = B.ConfirmationStore(str(tmp_path))
    c = _conf(nonce="n3")
    store.issue(c)
    p = store._pending("n3")
    obj = json.loads(p.read_text())
    obj["expires_epoch"] = 1e18                           # break the content hash
    p.write_text(json.dumps(obj))
    with pytest.raises(B.BudgetError, match="content hash"):
        store.consume(c, now_epoch=100.0)


def test_confirmation_unknown_nonce_refused(tmp_path):
    store = B.ConfirmationStore(str(tmp_path))
    with pytest.raises(B.BudgetError, match="no pending confirmation"):
        store.consume(_conf(nonce="never"), now_epoch=100.0)


_CONSUME_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    from acrpq.dashboard import qpu_budget as B
    root, out, gate = sys.argv[1:4]
    store = B.ConfirmationStore(root)
    c = B.Confirmation("sha256:"+"1"*64, "sha256:"+"2"*64, "CP_3", "maneuver_count_v1",
        "ibm_x", 1024, 9, 1024, 10.0, "sha256:"+"3"*64, "sha256:"+"4"*64, "shared-nonce", 1000.0)
    while not os.path.exists(gate):
        time.sleep(0.001)
    try:
        store.consume(c, now_epoch=100.0)
        res = {"ok": True}
    except B.BudgetError:
        res = {"ok": False}
    json.dump(res, open(out, "w"))
    """
)


def test_concurrent_consume_only_one_succeeds(tmp_path):
    store = B.ConfirmationStore(str(tmp_path / "c"))
    store.issue(_conf(nonce="shared-nonce"))
    worker = tmp_path / "cw.py"
    worker.write_text(_CONSUME_WORKER)
    gate = tmp_path / "GO"
    outs = [str(tmp_path / f"c{i}.json") for i in range(3)]
    procs = [subprocess.Popen([sys.executable, str(worker), str(tmp_path / "c"), outs[i], str(gate)])
             for i in range(3)]
    gate.write_text("go")
    for p in procs:
        assert p.wait(timeout=60) == 0
    res = [json.loads(Path(o).read_text()) for o in outs]
    assert sum(1 for r in res if r["ok"]) == 1            # exactly one consume wins
