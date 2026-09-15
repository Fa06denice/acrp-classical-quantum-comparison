"""Bounded asynchronous job manager for long dashboard computations.

A small, dependency-free job runner so a long solve (e.g. QAOA with many
iterations, or a real-hardware submission) does not block the HTTP request. It
is deliberately *bounded* and *honest*:

* a fixed worker pool (``max_workers``) and a bounded backlog (``max_queue``);
  submitting when the backlog is full is rejected with ``QueueFullError`` — the
  server never spawns unbounded threads;
* coarse, truthful states (``queued`` / ``running`` / ``completed`` / ``failed``
  / ``cancelled``); progress is elapsed time and the current state, never a
  fabricated percentage;
* cooperative cancellation: a *queued* job is cancelled before it starts; a job
  already *running* a synchronous solve cannot be interrupted mid-flight, and a
  submission already sent to external hardware cannot be recalled — both are
  reported honestly rather than pretended;
* finished jobs expire after ``ttl_s`` and are swept, so memory is bounded.

State is guarded by a single lock; the manager is safe under concurrent HTTP
requests.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueueFullError(RuntimeError):
    """Raised when the bounded backlog is full (map to HTTP 429)."""


_TERMINAL = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.QUEUED
    submitted_ts: float = 0.0
    started_ts: float | None = None
    finished_ts: float | None = None
    result: Any = None
    error: str | None = None
    cancel_requested: bool = False

    def snapshot(self, now: float) -> dict[str, Any]:
        started = self.started_ts
        finished = self.finished_ts
        if self.state in _TERMINAL and started is not None and finished is not None:
            elapsed = finished - started
        elif started is not None:
            elapsed = now - started
        else:
            elapsed = 0.0
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "queued_for_s": round((started or now) - self.submitted_ts, 3),
            "elapsed_s": round(elapsed, 3),
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "has_result": self.result is not None,
        }


@dataclass
class JobManager:
    max_workers: int = 2
    max_queue: int = 8
    ttl_s: float = 900.0  # sweep finished jobs after 15 min

    _jobs: dict[str, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pool: ThreadPoolExecutor | None = None

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="acrpq-job")

    # -- submission ------------------------------------------------------- #
    def submit(self, kind: str, fn: Callable[[], Any]) -> str:
        now = time.time()
        with self._lock:
            self._sweep(now)
            active = sum(1 for j in self._jobs.values() if j.state not in _TERMINAL)
            if active >= self.max_queue:
                raise QueueFullError(
                    f"job backlog full ({active}/{self.max_queue}); retry later"
                )
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, submitted_ts=now)
            self._jobs[job.id] = job
        assert self._pool is not None
        self._pool.submit(self._run, job.id, fn)
        return job.id

    def _run(self, job_id: str, fn: Callable[[], Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_requested:  # cancelled while still queued
                job.state = JobState.CANCELLED
                job.finished_ts = time.time()
                return
            job.state = JobState.RUNNING
            job.started_ts = time.time()
        try:
            result = fn()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                # honour a cancel that arrived during the run, but do not pretend
                # the underlying work stopped — the result is simply discarded
                if job.cancel_requested:
                    job.state = JobState.CANCELLED
                else:
                    job.state = JobState.COMPLETED
                    job.result = result
                job.finished_ts = time.time()
        except Exception as exc:  # noqa: BLE001 - report any solver failure as a failed job
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.state = JobState.FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.finished_ts = time.time()

    # -- queries / control ------------------------------------------------ #
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot(now) if job else None

    def result(self, job_id: str) -> Any:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.result if job else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        """Best-effort cancel. Effective only if the job has not started."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == JobState.QUEUED:
                job.cancel_requested = True  # _run will finalise as CANCELLED
                note = "cancelled before start"
            elif job.state == JobState.RUNNING:
                job.cancel_requested = True
                note = "cancel requested; a running solve cannot be interrupted mid-flight"
            else:
                note = f"already {job.state.value}"
            snap = job.snapshot(time.time())
            snap["cancel_note"] = note
            return snap

    def _sweep(self, now: float) -> None:
        expired = [
            jid for jid, j in self._jobs.items()
            if j.state in _TERMINAL and j.finished_ts is not None and (now - j.finished_ts) > self.ttl_s
        ]
        for jid in expired:
            del self._jobs[jid]

    def stats(self) -> dict[str, int]:
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j.state not in _TERMINAL)
            return {"total": len(self._jobs), "active": active,
                    "max_queue": self.max_queue, "max_workers": self.max_workers}
