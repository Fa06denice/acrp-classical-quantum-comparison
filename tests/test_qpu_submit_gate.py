"""Codex coordinated integration — point-of-use hardware submission gate (fake only).

Proves that a run_sampler placed BEHIND authorize_hardware_submission fires exactly once
when all four invariants hold, and CANNOT fire when any single invariant is neutered
(identity / consumed-confirmation / committed-budget / authoritative dedup). No IBM, no
real gateway — a fake gateway counts calls.
"""
from __future__ import annotations

import warnings as _w

import pytest

from acrpq.dashboard import qpu_budget as B
from acrpq.dashboard import qpu_dedup as D
from acrpq.dashboard import qpu_runs
from acrpq.dashboard import qpu_submit_gate as G

_w.filterwarnings("ignore")

CFG = "sha256:" + "a" * 64          # batch_config_hash
JOB = "sha256:" + "b" * 64          # job_config_hash


class FakeGateway:
    """Stand-in for RuntimeGateway — counts run_sampler calls; never contacts IBM."""

    def __init__(self):
        self.run_calls = 0

    def run_sampler(self, **_k):
        self.run_calls += 1
        return "fake-job-id"


def _guarded_run_sampler(gw: FakeGateway, gate_kwargs: dict):
    """The integration pattern: the gate is the last check before run_sampler."""
    G.authorize_hardware_submission(**gate_kwargs)      # raises if ANY invariant is absent
    return gw.run_sampler(isa="fake")                   # only reachable if authorized


def _params():
    return {"mode": "hardware_validation", "instance": "CP_3", "n_theta": 3, "n_q": 1,
            "shots": 128, "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
            "qubo_sha256": "sha256:" + "c" * 64, "backend_requested": "fake"}


def _policy():
    return B.BudgetPolicy(max_jobs=3, shots_per_job=1024, total_shot_budget=100000,
                          qpu_seconds_per_job=10.0, max_total_qpu_seconds=600.0,
                          max_wait_seconds=600.0, max_jobs_per_backend=3,
                          max_jobs_per_instance=3, stop_after_consecutive_errors=2)


def _conf(nonce="n1"):
    return B.Confirmation(batch_config_hash=CFG, job_config_hash=JOB, instance="CP_3",
                          objective_id="maneuver_count_v1", backend="ibm_x", shots=1024,
                          n_qubits=9, reserved_shots=1024, reserved_qpu_seconds=10.0,
                          qubo_sha256="sha256:" + "d" * 64, isa_sha256="sha256:" + "e" * 64,
                          nonce=nonce, expires_epoch=1000.0)


def _all_satisfied(tmp_path, *, skip=None):
    """Build the four states; ``skip`` neuters exactly one invariant."""
    runs = str(tmp_path / "runs")
    # 1. identity bound in the authoritative chain (unless skipped -> legacy run)
    rid = qpu_runs.prepare(_params(), root=runs)
    if skip != "identity":
        qpu_runs.record_batch_identity(rid, CFG, root=runs)
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs)
    # 2. confirmation issued + consumed (unless skipped)
    store = B.ConfirmationStore(str(tmp_path / "conf"))
    c = _conf()
    store.issue(c)
    if skip != "confirmation":
        store.consume(c, now_epoch=100.0)
    # 3. budget reserved + committed (unless skipped)
    ledger = B.BudgetLedger(str(tmp_path / "led"), _policy())
    res = ledger.reserve(reservation_id="r1", batch_config_hash=CFG, job_config_hash=JOB,
                         backend="ibm_x", instance="CP_3")
    if skip != "budget":
        ledger.commit(res)
    # 4. authoritative dedup decision (submit_allowed unless skipped -> a remote job exists)
    tag = D.job_tag(CFG, JOB)
    jobs = [] if skip != "dedup" else [D.RemoteJobRef("existing", (tag,), JOB)]
    decision = D.decide_submission(batch_config_hash=CFG, job_config_hash=JOB,
                                   local_job_id=None, search=D.FakeRemoteJobService(jobs=jobs))
    return dict(root=runs, local_run_id=rid, batch_config_hash=CFG, job_config_hash=JOB,
                confirmation=c, confirmation_store=store, ledger=ledger, reservation=res,
                dedup_decision=decision)


def test_all_invariants_present_fires_run_sampler_once(tmp_path):
    gw = FakeGateway()
    kw = _all_satisfied(tmp_path)
    assert _guarded_run_sampler(gw, kw) == "fake-job-id"
    assert gw.run_calls == 1


@pytest.mark.parametrize("neuter", ["identity", "confirmation", "budget", "dedup"])
def test_neutering_any_single_guard_blocks_run_sampler(tmp_path, neuter):
    gw = FakeGateway()
    kw = _all_satisfied(tmp_path, skip=neuter)
    with pytest.raises(G.SubmissionForbidden) as ei:
        _guarded_run_sampler(gw, kw)
    assert gw.run_calls == 0                             # run_sampler NEVER executed
    assert neuter in str(ei.value) or {                 # the message names the missing invariant
        "identity": "identity", "confirmation": "confirmation",
        "budget": "budget", "dedup": "dedup"}[neuter] in str(ei.value)


def test_gate_cross_binds_confirmation_and_reservation_to_identity(tmp_path):
    """Red-team #1: a confirmation/reservation/dedup minted for a DIFFERENT config must not
    authorise this run even if each is individually valid."""
    gw = FakeGateway()
    kw = _all_satisfied(tmp_path)                        # everything valid for (CFG, JOB)
    other_job = "sha256:" + "9" * 64
    # ask the gate to authorise a DIFFERENT job identity than the confirmation/reservation bind
    kw["job_config_hash"] = other_job
    with pytest.raises(G.SubmissionForbidden) as ei:
        _guarded_run_sampler(gw, kw)
    assert gw.run_calls == 0
    msg = str(ei.value)
    assert "confirmation_job_mismatch" in msg and "budget_job_mismatch" in msg
    assert "dedup_tag_mismatch" in msg                  # the dedup tag is for JOB, not other_job


def test_gate_rejects_forged_dedup_decision(tmp_path):
    """Red-team #4: a duck-typed object with may_submit=True but action!=submit_allowed is
    refused — the gate requires a real SubmissionDecision with action==submit_allowed."""
    class Forged:
        may_submit = True
        action = "refuse"
        tag = None
    gw = FakeGateway()
    kw = _all_satisfied(tmp_path)
    kw["dedup_decision"] = Forged()
    with pytest.raises(G.SubmissionForbidden, match="dedup_not_submit_allowed"):
        _guarded_run_sampler(gw, kw)
    assert gw.run_calls == 0


def test_gate_module_builds_no_service_and_imports_no_real_path():
    """The gate constructs no service and imports no real path (run_sampler appears only in
    its docstrings). The neuter test above proves it never fires run_sampler."""
    import inspect
    src = inspect.getsource(G)
    assert "QiskitRuntimeService(" not in src
    assert "import qpu_api" not in src and "quantum.backends" not in src
    assert "QISKIT_IBM_TOKEN" not in src
