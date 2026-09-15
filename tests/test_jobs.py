"""Bounded async job manager: concurrency, queue bound, cancel, TTL."""

from __future__ import annotations

import threading
import time

import pytest

from acrpq.dashboard.jobs import JobManager, QueueFullError


def _wait(mgr, jid, states, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        snap = mgr.snapshot(jid)
        if snap and snap["state"] in states:
            return snap
        time.sleep(0.01)
    raise AssertionError(f"job {jid} did not reach {states}: {mgr.snapshot(jid)}")


def test_completes_and_returns_result():
    mgr = JobManager(max_workers=2, max_queue=8)
    jid = mgr.submit("t", lambda: 21 * 2)
    _wait(mgr, jid, {"completed"})
    assert mgr.result(jid) == 42


def test_failure_recorded_not_raised():
    mgr = JobManager()
    def boom():
        raise ValueError("nope")
    jid = mgr.submit("t", boom)
    snap = _wait(mgr, jid, {"failed"})
    assert "ValueError: nope" in snap["error"]


def test_queue_bound_rejects_overflow():
    mgr = JobManager(max_workers=1, max_queue=2)
    gate = threading.Event()
    # occupy the manager with blocking jobs up to the bound
    ids = [mgr.submit("t", lambda: gate.wait(2)) for _ in range(2)]
    with pytest.raises(QueueFullError):
        mgr.submit("t", lambda: 1)
    gate.set()
    for jid in ids:
        _wait(mgr, jid, {"completed"})


def test_cancel_while_queued():
    mgr = JobManager(max_workers=1, max_queue=8)
    gate = threading.Event()
    running = mgr.submit("t", lambda: gate.wait(2))  # occupies the single worker
    _wait(mgr, running, {"running"})
    queued = mgr.submit("t", lambda: 1)  # sits behind the worker
    snap = mgr.cancel(queued)
    assert snap["cancel_requested"] is True
    gate.set()
    final = _wait(mgr, queued, {"cancelled", "completed"})
    # it was cancelled before starting
    assert final["state"] == "cancelled"


def test_cancel_running_is_honest():
    mgr = JobManager(max_workers=1, max_queue=8)
    gate = threading.Event()
    jid = mgr.submit("t", lambda: gate.wait(2))
    _wait(mgr, jid, {"running"})
    snap = mgr.cancel(jid)
    assert "cannot be interrupted" in snap["cancel_note"]
    gate.set()


def test_ttl_sweep():
    mgr = JobManager(ttl_s=0.0)  # everything expires immediately once finished
    jid = mgr.submit("t", lambda: 1)
    _wait(mgr, jid, {"completed"})
    time.sleep(0.02)
    mgr.submit("t", lambda: 2)  # triggers a sweep of the expired first job
    assert mgr.snapshot(jid) is None
